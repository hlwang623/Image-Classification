"""Tune fusion weights on OOF predictions."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import numpy as np
import joblib
from pathlib import Path

from src.utils import load_config, set_seed, ensure_dir, compute_metrics, print_classification_report
from src.fusion import load_oof_probs, greedy_forward_selection, learn_stacking_weights, find_best_temperature


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    root = Path(__file__).parent.parent
    oof_dir = root / "oof_probs"
    models_dir = ensure_dir(root / "models")

    all_oof = load_oof_probs(oof_dir)
    if not all_oof:
        print("No OOF predictions found. Run 02_train_heads.py first.")
        return

    first_key = list(all_oof.keys())[0]
    labels = all_oof[first_key]["labels"]

    print(f"Found {len(all_oof)} model OOF predictions")
    print(f"Labels shape: {labels.shape}")

    # Individual model scores
    print(f"\n{'='*60}")
    print("Individual Model OOF Scores")
    print(f"{'='*60}")
    individual_scores = []
    for name, data in all_oof.items():
        pred = np.argmax(data["probs"], axis=1)
        metrics = compute_metrics(labels, pred)
        individual_scores.append((name, metrics["combined_score"], metrics["macro_f1"], metrics["balanced_accuracy"]))
    individual_scores.sort(key=lambda x: x[1], reverse=True)
    for name, combined, mf1, bacc in individual_scores:
        print(f"  {name:<40} combined={combined:.4f}  F1={mf1:.4f}  bAcc={bacc:.4f}")

    # Greedy forward selection
    print(f"\n{'='*60}")
    print("Greedy Forward Selection")
    print(f"{'='*60}")
    selected_names, greedy_score = greedy_forward_selection(all_oof, labels)
    print(f"\nSelected {len(selected_names)} models, greedy score: {greedy_score:.4f}")

    # Simple average fusion
    print(f"\n{'='*60}")
    print("Simple Average Fusion (selected models)")
    print(f"{'='*60}")
    avg_probs = np.mean([all_oof[n]["probs"] for n in selected_names], axis=0)
    avg_pred = np.argmax(avg_probs, axis=1)
    avg_metrics = compute_metrics(labels, avg_pred)
    print(f"  combined={avg_metrics['combined_score']:.4f}  F1={avg_metrics['macro_f1']:.4f}  bAcc={avg_metrics['balanced_accuracy']:.4f}")
    print_classification_report(labels, avg_pred)

    # Learn stacking weights
    print(f"\n{'='*60}")
    print("Stacking Meta-Classifier")
    print(f"{'='*60}")
    meta, scaler, best_C, stacking_score = learn_stacking_weights(
        all_oof, labels, selected_names, n_splits=cfg["cv"]["n_splits"], seed=cfg["seed"]
    )
    print(f"  Best C: {best_C}")
    print(f"  Stacking CV score: {stacking_score:.4f}")

    # Temperature scaling on fused predictions
    print(f"\n{'='*60}")
    print("Temperature Scaling")
    print(f"{'='*60}")
    best_T = find_best_temperature(avg_probs, labels)
    print(f"  Optimal temperature: {best_T:.3f}")

    # Save fusion artifacts
    fusion_config = {
        "selected_models": selected_names,
        "greedy_score": greedy_score,
        "stacking_score": stacking_score,
        "temperature": best_T,
        "stacking_C": best_C,
    }
    import json
    with open(models_dir / "fusion_config.json", "w") as f:
        json.dump(fusion_config, f, indent=2)

    joblib.dump(meta, models_dir / "fusion_meta.joblib")
    joblib.dump(scaler, models_dir / "fusion_scaler.joblib")

    print(f"\nFusion artifacts saved to {models_dir}/")
    print(f"  fusion_config.json")
    print(f"  fusion_meta.joblib")
    print(f"  fusion_scaler.joblib")


if __name__ == "__main__":
    main()
