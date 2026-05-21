import os
import random
import numpy as np
import yaml
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root():
    return Path(__file__).parent.parent


def compute_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import f1_score, balanced_accuracy_score
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bacc = balanced_accuracy_score(y_true, y_pred)
    combined = 0.5 * macro_f1 + 0.5 * bacc
    result = {
        "macro_f1": macro_f1,
        "balanced_accuracy": bacc,
        "combined_score": combined,
    }
    return result


def print_classification_report(y_true, y_pred, class_names=None):
    from sklearn.metrics import classification_report, confusion_matrix
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(5)]
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    return cm


def get_device(gpu_id=0):
    import torch
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
