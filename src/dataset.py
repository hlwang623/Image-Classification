import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import cv2


class NucleiDataset(Dataset):
    def __init__(self, root_dir, variant="rgb", transform=None):
        self.root_dir = Path(root_dir)
        self.variant = variant
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_names = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self._apply_variant(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx], str(img_path)

    def _apply_variant(self, img):
        if self.variant == "rgb":
            return img
        elif self.variant == "clahe":
            return self._apply_clahe(img)
        elif self.variant == "gray3":
            return self._apply_gray3(img)
        return img

    def _apply_clahe(self, img):
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)

    def _apply_gray3(self, img):
        gray = img.convert("L")
        return Image.merge("RGB", [gray, gray, gray])


class TestDataset(Dataset):
    def __init__(self, test_dir, variant="rgb", transform=None):
        self.test_dir = Path(test_dir)
        self.variant = variant
        self.transform = transform
        self.samples = sorted(list(self.test_dir.glob("*.png")))
        if not self.samples:
            self.samples = sorted(list(self.test_dir.glob("**/*.png")))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self._apply_variant(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, str(img_path)

    def _apply_variant(self, img):
        if self.variant == "rgb":
            return img
        elif self.variant == "clahe":
            arr = np.array(img)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(result)
        elif self.variant == "gray3":
            gray = img.convert("L")
            return Image.merge("RGB", [gray, gray, gray])
        return img
