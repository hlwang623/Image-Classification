"""Live scoring script with fast/safe/full modes."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import argparse
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter

from src.utils import load_config, set_seed, ensure_dir
from src.extract import extract_features
from src.tta import get_tta_transforms
from src.extract import extract_features_with_tta

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Live scoring for nuclei classification")
    parser.add_argument("--test_dir", required=True, help="Path to test images directory")
    parser.add_argument("--mode", default="safe", choices=["fast", "safe", "full"])
    parser.add_argument("--output", default="submissions/submission.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_stacking", action="store_true", help="Use stacking meta-classifier")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    root = Path(__file__).parent.parent
    models_dir = root / "models"
    mode_cfg = cfg["modes"][args.mode]

    print(f"{'='*60}")
    print(f"LIVE SCORING - Mode: {args.mode}")
    print(f"{'='*60}")
    print(f"Test dir: {args.test_dir}")
    print(f"Device: {args.device}")

    test_dir = Path(args.test_dir)
    test_files = sorted(list(test_dir.glob("*.png")))
    if not test_files:
        test_files = sorted(list(test_dir.glob("**/*.png")))
    print(f"Found {len(test_files)} test images")
    assert len(test_files) > 0, "No test images found!"

    backbones = mode_cfg["backbones"]
    head_types = mode_cfg["heads"]
    variants = mode_cfg["variants"]
    tta_mode = mode_cfg["tta"]
    tta_views = get_tta_transforms(tta_mode)

    print(f"Backbones: {backbones}")
    print(f"Heads: {head_types}")
    print(f"Variants: {variants}")
    print(f"TTA: {tta_mode} ({len(tta_views)} views)")

    all_probs = []
    model_names_used = []
    t_start = time.time()

    for bb_name in backbones:
        if bb_name == "traditional":
            continue

        for variant in variants:
            print(f"\n--- {bb_name} / {variant} ---")

            try:
                if len(tta_views) > 1:
                    features, paths = extract_features_with_tta(
                        bb_name, split="test", variant=variant,
                        data_dir=args.test_dir, tta_views=tta_views,
                        device=args.device, batch_size=args.batch_size,
                        force=True
                    )
                else:
                    features, paths = extract_features(
                        bb_name, split="test", variant=variant,
                        data_dir=args.test_dir, device=args.device,
                        batch_size=args.batch_size, force=True
                    )
            except Exception as e:
                print(f"  FAILED to extract features: {e}")
                continue

            for head_type in head_types:
                model_key = f"{bb_name}_{variant}_{head_type}"
                model_path = models_dir / f"{model_key}.joblib"

                if not model_path.exists():
                    print(f"  SKIP {model_key}: model file not found")
                    continue

                try:
                    head = joblib.load(model_path)
                    probs = head.predict_proba(features)
                    all_probs.append(probs)
                    model_names_used.append(model_key)
                    print(f"  {model_key}: OK (shape={probs.shape})")
                except Exception as e:
                    print(f"  {model_key}: FAILED ({e})")

    if not all_probs:
        print("\nERROR: No models produced predictions!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"FUSION ({len(all_probs)} models)")
    print(f"{'='*60}")

    # Try stacking meta-classifier if available
    fusion_config_path = models_dir / "fusion_config.json"
    meta_path = models_dir / "fusion_meta.joblib"
    scaler_path = models_dir / "fusion_scaler.joblib"

    if args.use_stacking and fusion_config_path.exists() and meta_path.exists():
        try:
            with open(fusion_config_path) as f:
                fusion_cfg = json.load(f)
            meta = joblib.load(meta_path)
            scaler = joblib.load(scaler_path)

            selected = fusion_cfg["selected_models"]
            selected_probs = []
            for name in selected:
                if name in model_names_used:
                    idx = model_names_used.index(name)
                    selected_probs.append(all_probs[idx])

            if len(selected_probs) == len(selected):
                stacked = np.hstack(selected_probs)
                stacked_scaled = scaler.transform(stacked)
                fused_probs = meta.predict_proba(stacked_scaled)
                print("  Using stacking meta-classifier")
            else:
                print(f"  Stacking requires {len(selected)} models, only {len(selected_probs)} available")
                print("  Falling back to simple average")
                fused_probs = np.mean(all_probs, axis=0)
        except Exception as e:
            print(f"  Stacking failed: {e}, using simple average")
            fused_probs = np.mean(all_probs, axis=0)
    else:
        fused_probs = np.mean(all_probs, axis=0)
        print("  Using simple average fusion")

    # Temperature scaling
    if fusion_config_path.exists():
        with open(fusion_config_path) as f:
            fusion_cfg = json.load(f)
        T = fusion_cfg.get("temperature", 1.0)
        if T != 1.0:
            from src.fusion import temperature_scale
            fused_probs = temperature_scale(fused_probs, T)
            print(f"  Temperature scaling: T={T:.3f}")

    preds = np.argmax(fused_probs, axis=1)
    class_names = [f"Class_{i}" for i in range(5)]

    # Build submission
    filenames = [Path(p).name for p in paths]
    df = pd.DataFrame({
        "filename": filenames,
        "label": [class_names[p] for p in preds]
    })

    # Sanity checks
    print(f"\n{'='*60}")
    print("SANITY CHECKS")
    print(f"{'='*60}")
    assert list(df.columns) == ["filename", "label"], f"Bad columns: {list(df.columns)}"
    assert df["filename"].is_unique, "Duplicate filenames!"
    assert df["label"].isin(class_names).all(), f"Invalid labels: {set(df['label']) - set(class_names)}"
    print(f"  Columns: OK")
    print(f"  Unique filenames: OK ({len(df)})")
    print(f"  Valid labels: OK")

    pred_dist = Counter(df["label"])
    print(f"\nPrediction distribution:")
    for cls in class_names:
        count = pred_dist.get(cls, 0)
        pct = count / len(df) * 100
        print(f"  {cls}: {count:>6} ({pct:5.1f}%)")

    confidence = fused_probs.max(axis=1)
    print(f"\nConfidence stats:")
    print(f"  Mean: {confidence.mean():.3f}")
    print(f"  Min:  {confidence.min():.3f}")
    print(f"  Max:  {confidence.max():.3f}")

    # Save
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Rows: {len(df)}")

    # Save raw probabilities
    prob_path = output_path.with_suffix(".probs.npz")
    np.savez(prob_path, probs=fused_probs, filenames=np.array(filenames),
             preds=preds, model_names=np.array(model_names_used))
    print(f"Probabilities saved to {prob_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    print("DONE!")


if __name__ == "__main__":
    main()
