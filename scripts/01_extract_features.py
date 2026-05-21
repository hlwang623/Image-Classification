"""Extract features from all enabled backbones and cache to disk."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import argparse
import time
from pathlib import Path

from src.utils import load_config, set_seed, get_device
from src.extract import extract_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--test_dir", default=None)
    parser.add_argument("--models", default=None, help="Comma-separated backbone names, or 'all'")
    parser.add_argument("--variants", default=None, help="Comma-separated variants")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    root = Path(__file__).parent.parent

    if args.split == "train":
        data_dir = root / cfg["train_dir"]
    else:
        data_dir = Path(args.test_dir) if args.test_dir else root / cfg["test_dir"]

    if args.models:
        if args.models == "all":
            backbone_names = [
                name for name, bcfg in cfg["backbones"].items()
                if bcfg.get("enabled", False)
            ]
        else:
            backbone_names = args.models.split(",")
    else:
        backbone_names = [
            name for name, bcfg in cfg["backbones"].items()
            if bcfg.get("enabled", False)
        ]

    variants = args.variants.split(",") if args.variants else cfg.get("input_variants", ["rgb"])

    print(f"Split: {args.split}")
    print(f"Data dir: {data_dir}")
    print(f"Backbones: {backbone_names}")
    print(f"Variants: {variants}")
    print(f"Device: {args.device}")
    print()

    results = []
    for bb_name in backbone_names:
        for variant in variants:
            t0 = time.time()
            try:
                if args.split == "train":
                    features, labels, paths = extract_features(
                        bb_name, split="train", variant=variant,
                        data_dir=data_dir, device=args.device,
                        batch_size=args.batch_size, force=args.force
                    )
                    elapsed = time.time() - t0
                    results.append((bb_name, variant, features.shape, elapsed, "OK"))
                    print(f"  -> {bb_name}/{variant}: shape={features.shape}, time={elapsed:.1f}s\n")
                else:
                    features, paths = extract_features(
                        bb_name, split="test", variant=variant,
                        data_dir=data_dir, device=args.device,
                        batch_size=args.batch_size, force=args.force
                    )
                    elapsed = time.time() - t0
                    results.append((bb_name, variant, features.shape, elapsed, "OK"))
                    print(f"  -> {bb_name}/{variant}: shape={features.shape}, time={elapsed:.1f}s\n")
            except Exception as e:
                elapsed = time.time() - t0
                results.append((bb_name, variant, None, elapsed, str(e)))
                print(f"  -> {bb_name}/{variant}: FAILED ({e})\n")

    print("\n=== Summary ===")
    print(f"{'Backbone':<20} {'Variant':<10} {'Shape':<20} {'Time':>8} {'Status'}")
    for bb, var, shape, t, status in results:
        shape_str = str(shape) if shape is not None else "N/A"
        print(f"{bb:<20} {var:<10} {shape_str:<20} {t:7.1f}s {status}")


if __name__ == "__main__":
    main()
