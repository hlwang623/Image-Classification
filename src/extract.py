import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from .registry import load_backbone
from .dataset import NucleiDataset, TestDataset
from .utils import ensure_dir, get_device


def extract_features(backbone_name, split="train", variant="rgb",
                     data_dir=None, device="cuda", batch_size=32,
                     cache_dir=None, force=False):
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "cache" / "features"
    cache_dir = ensure_dir(cache_dir)

    cache_file = cache_dir / f"{backbone_name}_{variant}_{split}.npz"
    if cache_file.exists() and not force:
        print(f"  Loading cached features from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        if split == "train":
            return data["features"], data["labels"], data["paths"]
        else:
            return data["features"], data["paths"]

    print(f"  Extracting {backbone_name} features ({variant}, {split})...")
    model, transform, feat_dim, forward_fn = load_backbone(backbone_name, device)

    if split == "train":
        dataset = NucleiDataset(data_dir, variant=variant, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    else:
        dataset = TestDataset(data_dir, variant=variant, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    all_features = []
    all_labels = []
    all_paths = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"  {backbone_name}/{variant}/{split}"):
            if split == "train":
                images, labels, paths = batch
                all_labels.append(labels.numpy())
            else:
                images, paths = batch

            images = images.to(device, non_blocking=True)
            feats = forward_fn(model, images)

            if feats.dim() > 2:
                feats = feats.mean(dim=1)

            feats = torch.nn.functional.normalize(feats, dim=1)
            all_features.append(feats.cpu().numpy())
            all_paths.extend(paths)

    features = np.concatenate(all_features, axis=0)
    paths_arr = np.array(all_paths)

    if split == "train":
        labels = np.concatenate(all_labels, axis=0)
        np.savez(cache_file, features=features, labels=labels, paths=paths_arr)
        print(f"  Saved: {cache_file} | shape={features.shape}")
        return features, labels, paths_arr
    else:
        np.savez(cache_file, features=features, paths=paths_arr)
        print(f"  Saved: {cache_file} | shape={features.shape}")
        return features, paths_arr


def extract_features_with_tta(backbone_name, split, variant, data_dir,
                              tta_views, device="cuda", batch_size=32,
                              cache_dir=None, force=False):
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "cache" / "features"
    cache_dir = ensure_dir(cache_dir)

    tta_name = f"tta{len(tta_views)}"
    cache_file = cache_dir / f"{backbone_name}_{variant}_{split}_{tta_name}.npz"
    if cache_file.exists() and not force:
        print(f"  Loading cached TTA features from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        if split == "train":
            return data["features"], data["labels"], data["paths"]
        else:
            return data["features"], data["paths"]

    print(f"  Extracting {backbone_name} TTA features ({variant}, {split}, {len(tta_views)} views)...")
    model, transform, feat_dim, forward_fn = load_backbone(backbone_name, device)

    if split == "train":
        dataset = NucleiDataset(data_dir, variant=variant, transform=transform)
    else:
        dataset = TestDataset(data_dir, variant=variant, transform=transform)

    all_features = []
    all_labels = []
    all_paths = []

    with torch.inference_mode():
        for idx in tqdm(range(len(dataset)), desc=f"  TTA {backbone_name}"):
            if split == "train":
                img_tensor, label, path = dataset[idx]
                all_labels.append(label)
            else:
                img_tensor, path = dataset[idx]
            all_paths.append(path)

            view_feats = []
            for tta_fn in tta_views:
                augmented = tta_fn(img_tensor)
                augmented = augmented.unsqueeze(0).to(device, non_blocking=True)
                feat = forward_fn(model, augmented)
                if feat.dim() > 2:
                    feat = feat.mean(dim=1)
                feat = torch.nn.functional.normalize(feat, dim=1)
                view_feats.append(feat.cpu().numpy())

            avg_feat = np.mean(view_feats, axis=0)
            avg_feat = avg_feat / (np.linalg.norm(avg_feat, axis=1, keepdims=True) + 1e-8)
            all_features.append(avg_feat)

    features = np.concatenate(all_features, axis=0)
    paths_arr = np.array(all_paths)

    if split == "train":
        labels = np.array(all_labels)
        np.savez(cache_file, features=features, labels=labels, paths=paths_arr)
    else:
        np.savez(cache_file, features=features, paths=paths_arr)

    print(f"  Saved: {cache_file} | shape={features.shape}")
    if split == "train":
        return features, labels, paths_arr
    return features, paths_arr
