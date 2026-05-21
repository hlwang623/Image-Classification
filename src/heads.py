import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from .utils import compute_metrics


class HeadWrapper:
    def __init__(self, name, pipeline):
        self.name = name
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


def create_logreg(C=1.0, class_weight="balanced"):
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C, solver="lbfgs", max_iter=5000,
            multi_class="multinomial", class_weight=class_weight
        )
    )
    return HeadWrapper("logreg", pipeline)


def create_ridge(alpha=1.0):
    base = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=alpha, class_weight="balanced")
    )
    pipeline = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    return HeadWrapper("ridge", pipeline)


def create_knn(k=5, metric="cosine"):
    pipeline = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=k, metric=metric, weights="distance"
        )
    )
    return HeadWrapper("knn", pipeline)


HEAD_FACTORY = {
    "logreg": create_logreg,
    "ridge": create_ridge,
    "knn": create_knn,
}

HYPERPARAM_GRID = {
    "logreg": {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "ridge": {"alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "knn": {"k": [3, 5, 7, 11, 15]},
}


def tune_and_train(head_type, X, y, n_splits=5, seed=42):
    grid = HYPERPARAM_GRID.get(head_type, {})
    param_name = list(grid.keys())[0] if grid else None
    param_values = list(grid.values())[0] if grid else [None]

    best_score = -1
    best_param = None
    best_scores_per_fold = None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for param_val in param_values:
        fold_scores = []
        for train_idx, val_idx in skf.split(X, y):
            kwargs = {param_name: param_val} if param_name else {}
            head = HEAD_FACTORY[head_type](**kwargs)
            head.fit(X[train_idx], y[train_idx])
            pred = head.predict(X[val_idx])
            metrics = compute_metrics(y[val_idx], pred)
            fold_scores.append(metrics["combined_score"])

        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_param = param_val
            best_scores_per_fold = fold_scores

    kwargs = {param_name: best_param} if param_name else {}
    final_head = HEAD_FACTORY[head_type](**kwargs)
    final_head.fit(X, y)

    return final_head, best_param, best_score, best_scores_per_fold


def collect_oof_predictions(head_type, X, y, best_param, n_splits=5, n_repeats=10, seed=42):
    from sklearn.model_selection import RepeatedStratifiedKFold

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    param_name = list(HYPERPARAM_GRID.get(head_type, {}).keys())
    param_name = param_name[0] if param_name else None

    oof_probs = np.zeros((len(y), 5), dtype=np.float64)
    oof_counts = np.zeros(len(y), dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        kwargs = {param_name: best_param} if param_name else {}
        head = HEAD_FACTORY[head_type](**kwargs)
        head.fit(X[train_idx], y[train_idx])
        prob = head.predict_proba(X[val_idx])
        oof_probs[val_idx] += prob
        oof_counts[val_idx] += 1

    oof_probs /= oof_counts[:, None]
    return oof_probs
