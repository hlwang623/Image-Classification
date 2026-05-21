# 少样本细胞核图像 5 分类 — 强方案实现计划

## Context

课程期末大作业：32x32 RGB 细胞核显微图像 5 分类（H&E 染色）。训练集仅 250 张（每类 50 张），测试集由老师现场提供、现场跑分。评估指标为 macro-F1 和 balanced accuracy。硬件：RTX Pro 6000 + RTX 5090。

本方案融合了两套方案的核心思想：
- **方案 A**（简洁路线）：多 backbone 冻结特征 + 集成 + TTA
- **方案 B**（PDF 手册）：医学域模型、Cache 分类器、OOF 融合、三档模式、防泄漏 CV

---

## 核心策略

**冻结多 backbone 特征提取 + 多分类头 + D4 TTA + OOF 学融合权重**

不做端到端训练。把 250 张训练图当作 support set，站在多个 foundation model 的 embedding 空间上做少样本分类。

---

## 项目文件结构

```
e:\少样本图像分类\
├── train_few_shot/              # 训练数据 (250 张)
│   ├── Class_0/ ... Class_4/
├── test/                        # 现场提供
│
├── configs/
│   └── config.yaml              # 中央配置：backbone、head、TTA、融合、模式
│
├── src/
│   ├── __init__.py
│   ├── registry.py              # Backbone 注册表：名称 → (加载函数, 预处理, 特征维度)
│   ├── dataset.py               # NucleiDataset, 输入变体 (RGB, CLAHE, Gray3), resize
│   ├── extract.py               # 特征提取引擎 (批量 GPU, 缓存到磁盘)
│   ├── traditional.py           # HOG, LBP, 颜色直方图
│   ├── heads.py                 # 分类头：LogReg, Ridge, kNN, Cache（含 CV 调参）
│   ├── tta.py                   # D4 TTA (8 变换)
│   ├── fusion.py                # OOF 融合、温度缩放、class bias
│   └── utils.py                 # I/O、指标(macro_f1, balanced_acc)、seed
│
├── scripts/
│   ├── 00_eda.py                # EDA 可视化
│   ├── 01_extract_features.py   # 提取 & 缓存所有 backbone 特征
│   ├── 02_train_heads.py        # 训练分类头 + 收集 OOF 预测
│   ├── 03_tune_fusion.py        # 在 OOF 上学融合权重
│   ├── 04_cv_evaluate.py        # 完整 CV 评估报告
│   ├── 05_predict.py            # 离线生成 submission.csv
│   └── 06_live_score.py         # 现场跑分脚本 (fast/safe/full)
│
├── cache/
│   └── features/                # {backbone}_{variant}_{split}.npz
├── oof_probs/                   # OOF 概率矩阵
├── models/                      # 训练好的 head、融合权重、scaler
├── submissions/                 # 输出 CSV
├── reports/
│   ├── eda/                     # EDA 图
│   └── cv_results/              # CV 结果表、混淆矩阵
└── requirements.txt
```

---

## Backbone 优先级

### 第一梯队（必做，最高收益）

| 模型 | 为什么 | 参数量 | 特征维度 | 来源 |
|------|--------|--------|----------|------|
| **DINOv2-ViT-g/14-reg** | 最强通用自监督特征 | 1.1B | 1536 | `facebook/dinov2-giant` |
| **Phikon-v2** | ViT-L 在 4.6 亿病理切片上训练，与任务域完全匹配 | 307M | 1024 | `owkin/phikon-v2` |

### 第二梯队（强补充）

| 模型 | 为什么 | 参数量 | 特征维度 | 来源 |
|------|--------|--------|----------|------|
| **DINOv3-ViT-L** | 2025 年 Meta 最新，Gram anchoring 技术 | ~300M | 1024 | `facebook/dinov3` |
| **EVA-02-Large** | CLIP init + MAE，和 DINOv2 互补 | 304M | 1024 | `timm: eva02_large_patch14_448` |
| **BioMedCLIP** | 微软生物医学 CLIP，1500 万图文对 | 86M | 512 | `microsoft/BiomedCLIP` |

### 第三梯队（时间充足再做）

| 模型 | 为什么 | 来源 |
|------|--------|------|
| **ConvNeXtV2-Large** | CNN 归纳偏置，和 ViT 互补 | timm |
| **OpenCLIP ViT-L/14** | 不同预训练范式 | open_clip |
| **HOG + LBP + 颜色统计** | 传统特征，零成本，可解释 baseline | skimage/opencv |

---

## 分类头策略

每个 backbone 训练 3 种主力分类头：

| 分类头 | 超参搜索范围 | 说明 |
|--------|-------------|------|
| **LogisticRegression** | C=[0.01,0.1,1,10,100], max_iter=5000, class_weight='balanced' | 主力，有 predict_proba |
| **RidgeClassifier** | alpha=[0.1,1,10,100] | 闭式解，极快，需 CalibratedClassifierCV 转概率 |
| **kNN (cosine)** | k=[3,5,7,11] | 非参数，不需训练，直接检索 support set |

所有特征在训练前做 `StandardScaler` + `L2 normalize`。

### 可选分类头

| 分类头 | 说明 | 优先级 |
|--------|------|--------|
| **Cache classifier (Tip-Adapter 风格)** | exp(beta * cosine) 加权投票，beta=[5,10,20,40,80] | 中：本质是带温度的 kNN，OOF 有提升则保留 |
| **LinearSVC** | C=[0.01,0.1,1,10]，高维 embedding 上常很强 | 低：需 CalibratedClassifierCV 转概率 |
| **MLP head** | 1 hidden layer, dropout>=0.3 | 低：250 张极易过拟合，只作消融 |

---

## 输入变体

| 变体 | 做法 | 优先级 |
|------|------|--------|
| **RGB** | 原图 bicubic 上采样到 224×224，用模型自带 normalize | 必做：所有 backbone 的默认输入 |
| **CLAHE** | LAB 空间 L 通道 CLAHE (clipLimit=2.0)，增强核边界对比度 | 建议：OOF 有收益时保留 |
| **Gray3** | 转灰度后复制 3 通道 | 可选：降低染色变化影响，突出形态；但 H&E 颜色有诊断价值，可能丢信息 |

---

## D4 TTA 策略

细胞核无固定方向，D4 群变换（8 种）是零风险增强：

```
identity, hflip, vflip, rot180, rot90, rot270, transpose, anti-transpose
```

- **训练时**：不用 TTA，只用原图特征
- **推理时**：每张图 8 个视角 → 分别提特征 → 分别过分类头 → 概率取平均
- **三档控制**：fast=3 视角(identity+hflip+vflip), safe=D4(8 视角), full=D4+CLAHE 变体

---

## OOF 融合

### 交叉验证
- `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`
- 每个 (backbone, variant, head) 组合输出 OOF 概率矩阵 `[250, 5]`
- 所有模型用同一套 fold，保证可融合
- 增强只在 fold 内生成，防泄漏

### 融合搜索
1. 收集所有分支的 OOF 概率，堆叠为 `[250, num_models × 5]`
2. 贪心前向选择：逐个加入模型，OOF score 提升则保留
3. 学习 stacking 权重（LogisticRegression meta-classifier）
4. 温度缩放：优化 T 使 OOF NLL 最小，搜索范围 [0.5, 3.0]
5. Class bias：小范围 [-0.3, 0.3] 搜索，只在 OOF 上调

### 评估目标
```
score = 0.5 * macro_F1 + 0.5 * balanced_accuracy
```

### 融合后诊断
- 每类 recall/F1：balanced accuracy 低通常说明某类 recall 崩了
- 每类 precision：macro-F1 低但 balanced accuracy 高，可能是某类报太多
- 混淆矩阵：找最混淆的类别对
- 低置信样本 top 20：人工观察是否模糊、标注噪声

---

## 现场跑分三档模式

| 模式 | Backbone | Head | TTA | 预计时间(1000张) | 何时用 |
|------|----------|------|-----|-----------------|--------|
| **fast** | DINOv2-G + Phikon-v2 | LogReg | 3 视角 | ~30s | 环境出问题/时间极紧 |
| **safe** | 第 1+2 梯队 (5 模型) | LogReg+Ridge | D4 (8 视角) | ~3min | 默认提交模式 |
| **full** | 所有模型 + 全部 head | 全部 | D4 + CLAHE | ~15min | 时间充足 |

**容错机制**：
- 每个 backbone 用 try/except 包裹，加载失败则跳过并 warning
- 先跑 fast 确保有 submission.csv，再跑 safe/full 覆盖
- 输出前断言：列名正确、filename 唯一、label ∈ {Class_0,...,Class_4}
- 保存原始概率矩阵，便于事后分析

---

## 实现阶段

### Phase 0: 环境 & EDA（30 分钟）
- `requirements.txt`: torch, timm>=1.0, transformers, open-clip-torch, scikit-learn, scikit-image, opencv-python, pyyaml, tqdm
- `scripts/00_eda.py`: 每类 montage、RGB 通道统计、颜色直方图分布、t-SNE 可视化
- 检查：250 张 32x32 RGB，无损坏文件，每类 50 张
- 近重复图检测（像素 hash / embedding 余弦），防 fold 泄漏
- 输出到 `reports/eda/`

### Phase 1: 最小可用 pipeline（2-3 小时）
- 实现 `src/registry.py`（backbone 注册表）
- 实现 `src/dataset.py`（数据集 + 预处理）
- 实现 `src/extract.py`（GPU 批量特征提取 + 磁盘缓存）
- 实现 `src/heads.py`（3 种分类头 + CV 调参）
- `scripts/01_extract_features.py`: 先只跑 DINOv2-G + RGB
- `scripts/02_train_heads.py`: 5-fold CV，得到第一个 baseline 分数
- **验证**：单模型 macro-F1 预期 0.85-0.92

### Phase 2: 多 backbone + TTA（3-4 小时）
- 扩展 registry：加入 Phikon-v2, DINOv3, EVA-02, BioMedCLIP
- 实现 `src/tta.py`（D4 变换）
- 实现 `src/traditional.py`（HOG + LBP + 颜色直方图）
- 加入 CLAHE 输入变体
- 重新跑 01 + 02，收集所有分支的 OOF 概率
- **验证**：Phikon-v2 预期排名 top-2（病理域先验优势）

### Phase 3: OOF 融合（2-3 小时）
- 实现 `src/fusion.py`（堆叠、温度缩放、class bias）
- `scripts/03_tune_fusion.py`: 贪心选择 + meta-classifier + 温度
- `scripts/04_cv_evaluate.py`: 完整报告（每类 P/R/F1、混淆矩阵、消融表）
- 保存融合权重到 `models/`
- **验证**：融合分数 > 最佳单模型

### Phase 4: 现场脚本固化（1-2 小时）
- `scripts/06_live_score.py`: fast/safe/full 三档 + 容错 + CSV 校验
- 模拟测试：用 train_few_shot 平铺后做 mock 测试
- 离线模型权重缓存测试（HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1）
- `configs/config.yaml`: 最终配置锁定
- **验证**：三种模式都能产出合法 CSV，计时符合预期

### Phase 5: 可选增强（视时间和 OOF 收益决定）

| 增强项 | 预期收益 | 风险 | 优先级 |
|--------|---------|------|--------|
| 多尺度特征（224 + 336 concat） | 中 | 低：增加 ~50% 提取时间 | 中 |
| 跨 backbone 特征拼接后训练单头 | 中 | 低 | 中 |
| LoRA 微调 top-2 backbone (rank=4) | 中-高 | 高：250 张过拟合风险大，CV 方差大 | 低 |
| Gray3 输入变体 | 低-中 | 低 | 低 |
| Cache classifier (Tip-Adapter) | 低-中 | 低 | 低 |
| 高置信伪标签 (threshold=0.95, 1 轮) | 中 | 中：确认性偏差 | 低，做成开关 |
| Label propagation (kNN graph) | 中 | 中：需要规则允许 | 低，做成开关 |
| CLIP prompt (需先人工判断类别语义) | 低 | 高：Class_0~4 无语义，prompt 效果不可控 | 最低 |
| RAG / LLM few-shot 推理 | 低 | 高：32x32 太小，推理慢，现场不稳定 | 最低 |

---

## GPU 分工

| GPU | 负责 | 原因 |
|-----|------|------|
| RTX Pro 6000 | DINOv2-G, DINOv3, Phikon-v2 特征提取 | 显存大，跑大模型 |
| RTX 5090 | EVA-02, BioMedCLIP, OpenCLIP, 传统特征 | 并行提特征 |
| CPU | LogReg/Ridge/kNN 训练、融合、CSV 检查 | 计算量小 |

---

## 验证清单

| 阶段 | 验证内容 |
|------|---------|
| Phase 0 | EDA 图正常，250 张 32x32 RGB，无损坏文件 |
| Phase 1 | 单模型 CV macro-F1 在合理范围 (0.80-0.92)，缓存文件存在 |
| Phase 2 | 多模型排行有区分度，Phikon-v2 表现突出 |
| Phase 3 | 融合 >= 最佳单模型，温度在 [0.5, 3.0] |
| Phase 4 | 三种模式都输出合法 CSV，计时符合预期 |
| 最终 | submission.csv 格式：`filename,label`，filename 唯一，label ∈ {Class_0,...,Class_4} |

## 烟雾测试

```bash
# 用训练集模拟测试集，验证全流程
python scripts/06_live_score.py --test_dir train_few_shot_flat/ --mode safe --output submissions/smoke_test.csv
# 预期：准确率接近 1.0（因为是训练样本）
```
