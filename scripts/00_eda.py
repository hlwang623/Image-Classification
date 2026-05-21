"""EDA: Exploratory Data Analysis for nuclei images."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import Counter

from src.utils import load_config, ensure_dir


def main():
    cfg = load_config()
    root = Path(__file__).parent.parent
    train_dir = root / cfg["train_dir"]
    report_dir = ensure_dir(root / "reports" / "eda")

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"Classes: {classes}")

    all_images = {}
    counts = Counter()

    for cls in classes:
        cls_dir = train_dir / cls
        imgs = []
        for p in sorted(cls_dir.glob("*.png")):
            img = Image.open(p).convert("RGB")
            assert img.size == (32, 32), f"Unexpected size: {p} -> {img.size}"
            imgs.append((p.name, np.array(img)))
            counts[cls] += 1
        all_images[cls] = imgs

    print(f"Counts: {dict(counts)}")
    assert all(counts[f"Class_{i}"] == 50 for i in range(5)), "Not 50 per class!"

    # === Montage ===
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    fig.suptitle("Sample Montage (10 per class)", fontsize=16)
    for row, cls in enumerate(classes):
        for col in range(10):
            ax = axes[row, col]
            ax.imshow(all_images[cls][col][1])
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=12, rotation=0, labelpad=50)
    plt.tight_layout()
    plt.savefig(report_dir / "montage.png", dpi=150)
    plt.close()
    print(f"Saved montage to {report_dir / 'montage.png'}")

    # === Per-class RGB statistics ===
    print("\nPer-class RGB statistics:")
    print(f"{'Class':<10} {'R_mean':>8} {'G_mean':>8} {'B_mean':>8} {'R_std':>8} {'G_std':>8} {'B_std':>8}")
    class_stats = {}
    for cls in classes:
        pixels = np.stack([img for _, img in all_images[cls]], axis=0).astype(np.float32)
        means = pixels.mean(axis=(0, 1, 2))
        stds = pixels.std(axis=(0, 1, 2))
        class_stats[cls] = {"mean": means, "std": stds}
        print(f"{cls:<10} {means[0]:8.2f} {means[1]:8.2f} {means[2]:8.2f} "
              f"{stds[0]:8.2f} {stds[1]:8.2f} {stds[2]:8.2f}")

    # === Color histogram distributions ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    channel_names = ["Red", "Green", "Blue"]
    colors_plot = ["red", "green", "blue"]
    for ch in range(3):
        ax = axes[ch]
        for cls in classes:
            all_vals = np.concatenate([img[:, :, ch].flatten() for _, img in all_images[cls]])
            ax.hist(all_vals, bins=64, range=(0, 256), alpha=0.4, label=cls, density=True)
        ax.set_title(f"{channel_names[ch]} Channel Distribution")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(report_dir / "color_distributions.png", dpi=150)
    plt.close()
    print(f"Saved color distributions to {report_dir / 'color_distributions.png'}")

    # === t-SNE on raw pixels ===
    try:
        from sklearn.manifold import TSNE
        all_pixels = []
        all_labels = []
        for cls_idx, cls in enumerate(classes):
            for _, img in all_images[cls]:
                all_pixels.append(img.flatten())
                all_labels.append(cls_idx)
        X = np.array(all_pixels, dtype=np.float32)
        y = np.array(all_labels)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        for cls_idx, cls in enumerate(classes):
            mask = y == cls_idx
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=cls, alpha=0.7, s=30)
        ax.legend()
        ax.set_title("t-SNE of Raw Pixels (32x32x3)")
        plt.tight_layout()
        plt.savefig(report_dir / "tsne_raw_pixels.png", dpi=150)
        plt.close()
        print(f"Saved t-SNE to {report_dir / 'tsne_raw_pixels.png'}")
    except Exception as e:
        print(f"t-SNE failed: {e}")

    print("\nEDA complete!")


if __name__ == "__main__":
    main()
