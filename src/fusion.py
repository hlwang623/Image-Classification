import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from .utils import compute_metrics


def load_oof_probs(oof_dir):
    from pathlib import Path
    oof_dir = Path(oof_dir)
    results = {}
    for f in sorted(oof_dir.glob("*.npz")):
        data = np.load(f)
        name = f.stem
        results[name] = {
            "probs": data["probs"],
            "labels": data["labels"],
        }
    return results


def temperature_scale(probs, T, eps=1e-9):
    logits = np.log(np.clip(probs, eps, 1.0))
    scaled = softmax(logits / T, axis=1)
    return scaled


def find_best_temperature(probs, labels, T_range=(0.1, 10.0)):
    def nll_loss(T):
        scaled = temperature_scale(probs, T)
        return -np.mean(np.log(scaled[np.arange(len(labels)), labels] + 1e-10))
    result = minimize_scalar(nll_loss, bounds=T_range, method="bounded")
    return result.x


def fuse_predictions(probs_list, weights=None, temperatures=None,
                     class_bias=None, eps=1e-9):
    n_models = len(probs_list)
    if weights is None:
        weights = np.ones(n_models) / n_models
    weights = np.array(weights)
    weights = weights / weights.sum()

    fused = np.zeros_like(probs_list[0])
    for i, p in enumerate(probs_list):
        if temperatures is not None and temperatures[i] != 1.0:
            p = temperature_scale(p, temperatures[i])
        fused += weights[i] * p

    if class_bias is not None:
        logits = np.log(np.clip(fused, eps, 1.0))
        logits += np.array(class_bias)[None, :]
        fused = softmax(logits, axis=1)

    return fused


def greedy_forward_selection(all_oof_probs, labels, max_models=15):
    n_models = len(all_oof_probs)
    model_names = list(all_oof_probs.keys())
    selected = []
    remaining = list(range(n_models))
    best_global_score = -1

    while remaining and len(selected) < max_models:
        best_idx = None
        best_score = -1

        for idx in remaining:
            trial = selected + [idx]
            trial_probs = [all_oof_probs[model_names[i]]["probs"] for i in trial]
            fused = np.mean(trial_probs, axis=0)
            pred = np.argmax(fused, axis=1)
            metrics = compute_metrics(labels, pred)
            score = metrics["combined_score"]

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score > best_global_score:
            best_global_score = best_score
            selected.append(best_idx)
            remaining.remove(best_idx)
            print(f"  Selected {model_names[best_idx]}: score={best_score:.4f}")
        else:
            break

    selected_names = [model_names[i] for i in selected]
    return selected_names, best_global_score


def learn_stacking_weights(all_oof_probs, labels, selected_names, n_splits=5, seed=42):
    probs_list = [all_oof_probs[name]["probs"] for name in selected_names]
    stacked = np.hstack(probs_list)

    scaler = StandardScaler()
    stacked_scaled = scaler.fit_transform(stacked)

    best_C = 1.0
    best_score = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        fold_scores = []
        for train_idx, val_idx in skf.split(stacked_scaled, labels):
            meta = LogisticRegression(C=C, max_iter=5000, solver="lbfgs",
                                     multi_class="multinomial")
            meta.fit(stacked_scaled[train_idx], labels[train_idx])
            pred = meta.predict(stacked_scaled[val_idx])
            metrics = compute_metrics(labels[val_idx], pred)
            fold_scores.append(metrics["combined_score"])
        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_C = C

    meta = LogisticRegression(C=best_C, max_iter=5000, solver="lbfgs",
                              multi_class="multinomial")
    meta.fit(stacked_scaled, labels)

    return meta, scaler, best_C, best_score
