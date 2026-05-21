"""Train classification heads on extracted features and collect OOF predictions."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import argparse
import numpy as np
from pathlib import Path

from src.utils import load_config, set_seed, ensure_dir, compute_metrics, print_classification_report
from src.heads import tune_and_train, collect_oof_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None, help="Comma-separated backbone names")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--heads", default=None, help="Comma-separated head types")
    parser.add_argument("--n_repeats", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    root = Path(__file__).parent.parent
    cache_dir = root / "cache" / "features"
    oof_dir = ensure_dir(root / "oof_probs")
    models_dir = ensure_dir(root / "models")

    if args.models:
        backbone_names = args.models.split(",")
    else:
        backbone_names = [
            name for name, bcfg in cfg["backbones"].items()
            if bcfg.get("enabled", False) and bcfg.get("source") != "traditional"
        ]

    variants = args.variants.split(",") if args.variants else cfg.get("input_variants", ["rgb"])

    if args.heads:
        head_types = args.heads.split(",")
    else:
        head_types = [
            name for name, hcfg in cfg["heads"].items()
            if hcfg.get("enabled", False)
        ]

    print(f"Backbones: {backbone_names}")
    print(f"Variants: {variants}")
    print(f"Heads: {head_types}")
    print()

    leaderboard = []

    for bb_name in backbone_names:
        for variant in variants:
            cache_file = cache_dir / f"{bb_name}_{variant}_train.npz"
            if not cache_file.exists():
                print(f"SKIP {bb_name}/{variant}: cache not found at {cache_file}")
                continue

            data = np.load(cache_file)
            X = data["features"]
            y = data["labels"]
            print(f"\n{'='*60}")
            print(f"Backbone: {bb_name} | Variant: {variant} | Features: {X.shape}")
            print(f"{'='*60}")

            for head_type in head_types:
                print(f"\n  --- {head_type} ---")

                head, best_param, best_score, fold_scores = tune_and_train(
                    head_type, X, y, n_splits=cfg["cv"]["n_splits"], seed=cfg["seed"]
                )

                print(f"  Best param: {best_param}")
                print(f"  CV score: {best_score:.4f} (std={np.std(fold_scores):.4f})")
                print(f"  Per-fold: {[f'{s:.4f}' for s in fold_scores]}")

                oof_probs = collect_oof_predictions(
                    head_type, X, y, best_param,
                    n_splits=cfg["cv"]["n_splits"],
                    n_repeats=args.n_repeats,
                    seed=cfg["seed"]
                )

                oof_pred = np.argmax(oof_probs, axis=1)
                oof_metrics = compute_metrics(y, oof_pred)
                print(f"  OOF macro_F1:  {oof_metrics['macro_f1']:.4f}")
                print(f"  OOF bAcc:      {oof_metrics['balanced_accuracy']:.4f}")
                print(f"  OOF combined:  {oof_metrics['combined_score']:.4f}")

                model_key = f"{bb_name}_{variant}_{head_type}"
                np.savez(
                    oof_dir / f"{model_key}.npz",
                    probs=oof_probs, labels=y, model_key=model_key
                )

                import joblib
                joblib.dump(head, models_dir / f"{model_key}.joblib")

                leaderboard.append({
                    "backbone": bb_name,
                    "variant": variant,
                    "head": head_type,
                    "best_param": best_param,
                    "cv_score": best_score,
                    "oof_macro_f1": oof_metrics["macro_f1"],
                    "oof_bacc": oof_metrics["balanced_accuracy"],
                    "oof_combined": oof_metrics["combined_score"],
                })

    print(f"\n\n{'='*80}")
    print("LEADERBOARD")
    print(f"{'='*80}")
    leaderboard.sort(key=lambda x: x["oof_combined"], reverse=True)
    print(f"{'#':<4} {'Backbone':<18} {'Var':<8} {'Head':<10} {'Param':>8} "
          f"{'macro_F1':>9} {'bAcc':>9} {'Combined':>9}")
    for i, entry in enumerate(leaderboard):
        print(f"{i+1:<4} {entry['backbone']:<18} {entry['variant']:<8} {entry['head']:<10} "
              f"{str(entry['best_param']):>8} {entry['oof_macro_f1']:9.4f} "
              f"{entry['oof_bacc']:9.4f} {entry['oof_combined']:9.4f}")


if __name__ == "__main__":
    main()
