# CODEX 行动参考文档：少样本 32×32 细胞核 5 分类冠军冲刺版

> 交给 Codex 执行。当前已拿到最终测试集，可以做无标签测试集适配、LoRA、伪标签、Label Propagation。目标是最大化 `macro-F1` 与 `balanced accuracy`，不是普通 accuracy。

## 0. 约束与原则

1. 只允许老师给的 `train_few_shot/Class_0~Class_4` 共 250 张图作为有标签监督来源。
2. 最终测试集只能作为无标签图像使用：允许 TTA、特征提取、聚类、Label Propagation、high-confidence pseudo-label、test-time adaptation。
3. 不得使用 PanNuke、MoNuSAC、MoNuSeg、PathMNIST、LC25000 等外部带标签图像数据训练或验证。
4. 不要从零训练 CNN 做主力，不要强 CutMix / RandomResizedCrop，不要让 LLM/VLM 直接决定类别。
5. 主线是：多 foundation backbone frozen feature + few-shot head + OOF 融合 + LoRA 增益 + 保守伪标签。

## 1. P0：先修正工程基础

### 1.1 固定 folds

新增：

```text
src/folds.py
scripts/00c_make_folds.py
models/fold_indices.json
```

使用：

```python
RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
```

所有模型、所有 head、所有 LoRA、所有 OOF 融合必须复用同一份 fold 文件，避免 OOF 不可比。

### 1.2 路径一致性检查

新增：

```text
cache/canonical_train_paths.json
cache/canonical_test_paths.json
```

每个 `.npz` 都必须保存 `paths`。融合前必须：

```python
assert list(paths_this_model) == list(canonical_paths)
```

否则不同模型概率会错位。

### 1.3 live_score 只融合 OOF 选中的模型

修改 `scripts/06_live_score.py`：

- 默认读取 `models/fusion_config.json`；
- 只使用 `selected_models`；
- 使用 `weights`、`temperatures`、`class_bias`；
- selected model 缺失时降级，但必须打印 warning；
- 禁止 full 模式简单平均所有可用模型。

## 2. 模型池升级

新增 `safe_plus` mode：

```yaml
modes:
  safe_plus:
    backbones: [dinobloom_l, phikon_v2, h0_mini, biomedclip, dinov2_g, eva02_large]
    heads: [logreg, ridge, linearsvc, centroid, cache]
    variants: [rgb, clahe]
    tta: safe
```

新增 `full_plus` mode：

```yaml
modes:
  full_plus:
    backbones: [dinobloom_l, dinobloom_g, phikon_v2, h0_mini, biomedclip, dinov2_g, dinov2_l, eva02_large, convnextv2_large, openclip_vitl, cell_dino]
    heads: [logreg, ridge, linearsvc, knn, centroid, cache]
    variants: [rgb, clahe, gray3]
    tta: safe
```

优先实现 loader：

```text
dinobloom_l      # 32×32 nucleus/single-cell morphology 关键分支
dinobloom_g      # 显存允许则启用
phikon_v2        # H&E histology 主力
h0_mini          # 轻量 histology foundation model
biomedclip       # biomedical/microscopy/histology 补充
dinov2_g         # 通用自监督强基线
eva02_large      # 与 DINO/医学模型互补
```

DinoBloom 优先尝试：

```python
timm.create_model("hf-hub:1aurent/vit_large_patch14_224.dinobloom", pretrained=True, num_classes=0)
timm.create_model("hf-hub:1aurent/vit_giant_patch14_224.dinobloom", pretrained=True, num_classes=0)
```

H0-mini 以模型卡官方加载方式为准。任何 backbone 加载失败都必须 graceful skip，不能使全流程崩溃。

## 3. TTA 必须训练/验证/推理一致

当前不要只在 test 做 TTA feature average。必须比较三种方式：

```text
A. identity feature train + identity test
B. TTA-averaged feature train + TTA-averaged test
C. identity feature train + 每个 TTA view 分别 predict_proba + prob average
```

保存特征命名：

```text
cache/features/{backbone}_{variant}_{split}_{feature_mode}.npz
feature_mode in [identity, tta8_featavg]
```

D4 TTA：

```text
identity, hflip, vflip, rot90, rot180, rot270, transpose, anti_transpose
```

实现 batch TTA：把 `[B, V, C, H, W]` reshape 到 `[B*V, C, H, W]` 一次推理，再 reshape 回 `[B, V, D]`。

## 4. 新增 morphology / HED 分支

新增：

```text
src/morphology.py
scripts/01b_extract_morphology.py
scripts/02b_train_morphology_heads.py
```

提取特征：

```text
RGB/HSV/LAB 每通道 mean/std/min/max/p10/p25/p50/p75/p90/skew/kurtosis
HED 或 H&E color deconvolution：hematoxylin/eosin mean/std/H-E ratio
Otsu nucleus mask：area/perimeter/eccentricity/solidity/extent/axis ratio/components
LBP histogram、GLCM texture、Laplacian blur、Sobel edge magnitude
```

训练：

```text
morphology + StandardScaler + LogisticRegression / LinearSVC / Ridge
```

即使单独分数不高，只要能提升某个弱类 recall，就低权重加入融合。

## 5. 分类头升级

现有 logreg/ridge/knn 外，新增：

```text
linearsvc
centroid
cache
```

### 5.1 LinearSVC

```python
LinearSVC(C=..., class_weight="balanced") + CalibratedClassifierCV(cv=3, method="sigmoid")
C = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
```

### 5.2 nearest centroid

```python
prototype_c = normalize(mean(normalize(X_train[y == c])))
score_c = cosine(z, prototype_c)
prob = softmax(score / T)
T = [0.03, 0.05, 0.07, 0.1, 0.2, 0.5]
```

### 5.3 cache classifier / Tip-Adapter-style

```text
sim_i = cosine(z_test, z_train_i)
score_c = sum_i exp(beta * sim_i) * onehot(y_i=c)
```

搜索：

```text
beta = [5, 10, 20, 40, 80, 120]
top_k = [3, 5, 7, 11, 15, 25, all]
```

注意：LogReg/Ridge/SVC 使用 StandardScaler；kNN/centroid/cache 使用原始 L2-normalized embedding，不要对 cosine head 使用 StandardScaler 后的特征。

## 6. OOF 融合升级

每个分支输出：

```text
oof_probs/{model_key}.npz
  probs: [250, 5]
  labels: [250]
  paths: [250]
  model_key: str
```

`model_key`：

```text
{backbone}_{variant}_{feature_mode}_{head}
example: dinobloom_l_rgb_tta8_featavg_cache
```

融合采用 log-prob weighted fusion：

```python
logits = 0
for p, w, T in zip(probs_list, weights, temperatures):
    logits += w * np.log(np.clip(p, 1e-8, 1.0)) / T
logits += class_bias[None, :]
p_final = softmax(logits, axis=1)
```

搜索：

```text
greedy forward selection
+ random simplex / Dirichlet search
+ optional scipy differential_evolution
+ class_bias search in [-0.5, 0.5]
```

目标：

```python
score = 0.5 * macro_f1 + 0.5 * balanced_accuracy
```

保存：

```json
models/fusion_config.json
{
  "selected_models": [],
  "weights": [],
  "temperatures": [],
  "class_bias": [],
  "fusion_type": "logprob_weighted",
  "oof_macro_f1": 0.0,
  "oof_balanced_accuracy": 0.0,
  "oof_combined": 0.0
}
```

新增：

```text
scripts/04_cv_evaluate.py
reports/cv_results/leaderboard.csv
reports/cv_results/fusion_report.md
reports/cv_results/per_class_metrics.csv
reports/cv_results/confusion_matrix.png
```

## 7. LoRA 计划

LoRA 分两阶段：

```text
A. supervised LoRA：只用 250 张训练图，做 5-fold OOF。
B. pseudo-label LoRA：只在超高置信伪标签上低权重训练，默认可关闭。
```

只对 OOF 前 2 名且错误互补的 backbone 做：

```text
phikon_v2
h0_mini
dinobloom_l
dinov2_l 或 dinov2_g
```

新增：

```text
src/lora_utils.py
scripts/07_train_lora.py
scripts/07_predict_lora.py
```

推荐参数：

```yaml
rank: [4, 8]
alpha: [8, 16]
dropout: 0.05
epochs: 80
early_stopping_patience: 12
batch_size: 16
head_lr: 1e-3
lora_lr: [1e-5, 3e-5, 5e-5]
weight_decay: 0.05
label_smoothing: 0.05
augment: D4 + mild color jitter + tiny gaussian noise
forbid: random crop, cutmix, full fine-tune
```

LoRA 进入 final 的条件：

```text
1. LoRA OOF combined >= 对应 frozen branch + 0.005；或
2. 提升最弱类 recall 且 precision 不崩；或
3. 加入融合后 OOF combined 提升。
```

否则丢弃。

## 8. 最终测试集伪标签与 Label Propagation

### 8.1 先跑 supervised base

```bash
python scripts/06_live_score.py \
  --test_dir test \
  --mode safe_plus \
  --output submissions/submission_safe_plus.csv \
  --use_selected_fusion
```

### 8.2 生成伪标签

新增：

```text
scripts/08_make_pseudo_labels.py
```

规则：

```text
pmax >= 0.98
margin = top1 - top2 >= 0.20
不要强制每类均匀
每类最多取 min(该类预测数量的 20%, 1000)；若测试集小则最多 200/class
weight = 0.1 + 0.2 * clip((pmax - threshold)/(1-threshold), 0, 1)
```

输出：

```text
pseudo/pseudo_round1.csv
filename,pseudo_label,pred_idx,pmax,margin,weight,source
```

### 8.3 伪标签重训轻量 head

新增：

```text
scripts/09_train_pseudo_heads.py
```

只重训 logreg/cache：

```text
训练集 = 250 labeled + selected pseudo-labeled test
真实训练样本 weight = 1.0
伪标签样本 weight = 0.1~0.3
```

默认不对 Ridge/SVC 做 weighted pseudo-label，除非实现稳定。

### 8.4 Label Propagation

新增：

```text
src/transductive.py
scripts/10_label_propagation.py
```

用 selected backbone normalized embeddings 拼接：

```text
dinobloom_l, phikon_v2, h0_mini, biomedclip, dinov2_g
```

实现：

```python
LabelSpreading(kernel="knn", n_neighbors=10 or 15, alpha=0.2)
```

输出：

```text
test_probs/label_propagation_probs.npz
```

建议融合：

```text
p_transductive = 0.85 * p_supervised + 0.10 * p_pseudo_head + 0.05 * p_label_propagation
```

若 Label Propagation 预测分布异常或置信度低，权重设为 0。

### 8.5 pseudo-label LoRA

默认关闭。只有满足以下条件才开：

```text
pmax >= 0.995
margin >= 0.30
每类最多取 min(300, 0.2 * predicted_count_class)
pseudo sample weight = 0.05~0.15
只训练 10~30 epoch
LoRA lr <= supervised LoRA lr / 2
```

若 compared with supervised 预测改变比例 > 15%，回滚。

## 9. 最终 ensemble 与提交

新增：

```text
scripts/11_final_ensemble.py
scripts/12_compare_submissions.py
scripts/99_validate_submission.py
```

生成：

```text
submissions/submission_safe_plus.csv
submissions/submission_full_supervised.csv
submissions/submission_transductive_light.csv
submissions/submission_final.csv
```

选择规则：

```text
1. LoRA OOF 有可靠提升，优先 full_supervised。
2. pseudo/LP 诊断正常，使用 transductive_light。
3. transductive 改变比例 > 15%，默认回滚到 full_supervised。
4. 任何复杂模块失败，保底 safe_plus。
```

提交校验：

```text
列名：filename,label
行数等于测试图像数
filename 唯一且与 test basename 一致
label in Class_0~Class_4
概率无 NaN/Inf
```

## 10. 执行命令顺序

```bash
git checkout -b champion-final
pip install -r requirements.txt
pip install peft accelerate umap-learn optuna

python scripts/00_eda.py --test_dir test
python scripts/00c_make_folds.py --train_dir train_few_shot --out models/fold_indices.json

python scripts/01_extract_features.py --split train --models dinobloom_l,phikon_v2,h0_mini,biomedclip,dinov2_g,eva02_large --variants rgb,clahe --feature_modes identity,tta8_featavg --device cuda:0
python scripts/01_extract_features.py --split test --test_dir test --models dinobloom_l,phikon_v2,h0_mini,biomedclip,dinov2_g,eva02_large --variants rgb,clahe --feature_modes identity,tta8_featavg --device cuda:0

python scripts/01b_extract_morphology.py --split train --data_dir train_few_shot
python scripts/01b_extract_morphology.py --split test --data_dir test

python scripts/02_train_heads.py --models dinobloom_l,phikon_v2,h0_mini,biomedclip,dinov2_g,eva02_large --variants rgb,clahe --feature_modes identity,tta8_featavg --heads logreg,ridge,linearsvc,centroid,cache --folds models/fold_indices.json
python scripts/02b_train_morphology_heads.py --heads logreg,linearsvc,ridge --folds models/fold_indices.json

python scripts/03_tune_fusion.py --oof_dir oof_probs --method logprob_weighted --max_models 20 --out models/fusion_config.json
python scripts/04_cv_evaluate.py --fusion_config models/fusion_config.json --out reports/cv_results

python scripts/07_train_lora.py --backbone phikon_v2 --train_dir train_few_shot --folds models/fold_indices.json --rank 4,8 --lora_lr 1e-5,3e-5,5e-5 --epochs 80 --early_stop 12 --output models/lora/phikon_v2
python scripts/07_predict_lora.py --backbone phikon_v2 --test_dir test --ckpt_dir models/lora/phikon_v2 --output test_probs/lora_phikon_v2_probs.npz

python scripts/06_live_score.py --test_dir test --mode safe_plus --output submissions/submission_safe_plus.csv --use_selected_fusion
python scripts/11_final_ensemble.py --base_probs submissions/submission_safe_plus.probs.npz --lora_probs test_probs/lora_phikon_v2_probs.npz --fusion_config models/fusion_config.json --output submissions/submission_full_supervised.csv

python scripts/08_make_pseudo_labels.py --probs submissions/submission_full_supervised.probs.npz --threshold 0.98 --margin 0.20 --max_per_class auto --output pseudo/pseudo_round1.csv
python scripts/09_train_pseudo_heads.py --pseudo_csv pseudo/pseudo_round1.csv --models selected_from_fusion --heads logreg,cache --pseudo_weight 0.2 --output test_probs/pseudo_heads_probs.npz
python scripts/10_label_propagation.py --train_features selected_from_fusion --test_features selected_from_fusion --n_neighbors 10,15,25 --alpha 0.2,0.4 --output test_probs/label_propagation_probs.npz
python scripts/11_final_ensemble.py --base_probs submissions/submission_full_supervised.probs.npz --pseudo_probs test_probs/pseudo_heads_probs.npz --labelprop_probs test_probs/label_propagation_probs.npz --transductive_weights 0.85,0.10,0.05 --output submissions/submission_transductive_light.csv

python scripts/12_compare_submissions.py --submissions submissions/submission_safe_plus.csv,submissions/submission_full_supervised.csv,submissions/submission_transductive_light.csv --out reports/final_compare.md
cp submissions/submission_transductive_light.csv submission.csv
python scripts/99_validate_submission.py --csv submission.csv --test_dir test
```

## 11. 回滚规则

```text
如果 LoRA OOF 不提升：不用 LoRA。
如果 pseudo-label 平均置信度 < 0.985：不用 pseudo。
如果 transductive 相比 supervised 改变 > 15%：回滚 full_supervised。
如果任意复杂模块失败：提交 safe_plus。
```

最终优先级：

```text
DinoBloom + Phikon-v2 + H0-mini + BioMedCLIP + DINOv2
> cache/centroid few-shot head
> morphology/HED branch
> OOF log-prob weighted fusion + class bias
> supervised LoRA
> strict pseudo-label + Label Propagation
```
