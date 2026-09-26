"""Reference baselines:

  Persistence (oracle),
  LR, single window          - required by the PS,
  LR, stacked history,
  Gradient boosting, stacked - sklearn HistGradientBoosting,
  Random forest, stacked     - RF on rolling-window features,
  LSTM classifier,
  Stage LR per horizon       - direct multi-horizon stage classifier (stronger than persistence)
Features, scaler and labels are identical to the world model's so any gap is attributable to the model.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..constants import EXPLOIT_STAGES
from ..features.sequences import stacked_features


class PersistenceBaseline:
    """Repeat the current label: alert if the current window is malicious.

    Uses *ground-truth* current labels, i.e. it is an oracle detector - a strong,
    optimistic baseline for during-attack windows but blind before first contact."""

    name = "Persistence (oracle current label)"

    def __init__(self, any_attack: bool = True):
        self.any_attack = any_attack

    def score(self, stage: np.ndarray) -> np.ndarray:
        s = np.asarray(stage)
        return (s > 0 if self.any_attack else np.isin(s, EXPLOIT_STAGES)).astype(np.float32)


def subsample(y: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """Keep every positive, sample negatives, total <= cap rows."""
    idx = np.arange(len(y))
    if len(y) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    pos, neg = idx[y > 0], idx[y <= 0]
    k = max(cap - len(pos), cap // 2)
    return np.sort(np.r_[pos[: cap // 2] if len(pos) > cap // 2 else pos, rng.choice(neg, min(k, len(neg)), replace=False)])


class SklearnBaseline:
    def __init__(self, name: str, est, stack: int = 1):
        self.name, self.est, self.stack = name, est, stack
        self.background = None

    def design(self, X, seg):
        return X if self.stack == 1 else stacked_features(X, seg, self.stack)

    def fit(self, X, seg, y, cap: int, seed: int = 0):
        D = self.design(X, seg)
        rows = subsample(y, cap, seed)
        self.est.fit(D[rows], y[rows])
        self.background = D[rows].mean(0)
        return self

    def score(self, X, seg):
        return self.est.predict_proba(self.design(X, seg))[:, 1]

    def linear_shap(self, x_row):
        """Exact SHAP of the logit for a linear model (feature independence)."""
        return (np.atleast_2d(x_row) - self.background) * self.est.coef_[0]


def make_baselines(stack: int, seed: int = 0) -> list[SklearnBaseline]:
    return [
        SklearnBaseline("Logistic Regression (single window)",
                        LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000), 1),
        SklearnBaseline(f"Logistic Regression (stacked {stack})",
                        LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000), stack),
        SklearnBaseline(f"Gradient Boosting (stacked {stack})",
                        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced",
                                                       random_state=seed), stack),
        SklearnBaseline(f"Random Forest (stacked {stack})",
                        RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features="sqrt",
                                               class_weight="balanced_subsample", n_jobs=-1, random_state=seed), stack),
    ]


# kept for backwards compatibility with older bundles/tests
class LogisticBaseline(SklearnBaseline):
    name = "Logistic Regression"

    def __init__(self, C: float = 0.5):
        super().__init__("Logistic Regression", LogisticRegression(C=C, class_weight="balanced", max_iter=3000), 1)

    def fit(self, X, y):  # type: ignore[override]
        self.est.fit(X, y)
        self.background = X.mean(0)
        self.clf = self.est
        return self

    def score(self, X):  # type: ignore[override]
        return self.est.predict_proba(X)[:, 1]


class StageHorizonLR:
    """One multinomial LR per horizon k predicting stage[t+k] from stacked history."""

    name = "Stage LR per horizon (stacked)"

    def __init__(self, K: int, stack: int):
        self.K, self.stack = K, stack
        self.models: dict[int, LogisticRegression | int] = {}

    def fit(self, X, seg, stage, cap: int, seed: int = 0, horizons=None):
        D = stacked_features(X, seg, self.stack)
        for k in horizons or range(1, self.K + 1):
            tgt, ok = future_stage(stage, seg, k)
            rows = np.flatnonzero(ok)
            y = tgt[rows]
            rows = rows[subsample((y > 0).astype(int), cap, seed + k)]
            y = tgt[rows]
            if len(np.unique(y)) < 2:
                self.models[k] = int(y[0]) if len(y) else 0
                continue
            self.models[k] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=300, tol=1e-3).fit(D[rows], y)
        return self

    def predict(self, X, seg, k: int) -> np.ndarray:
        m = self.models[k]
        if isinstance(m, int):
            return np.full(len(X), m)
        return m.predict(stacked_features(X, seg, self.stack))


def future_stage(stage: np.ndarray, seg: np.ndarray, k: int):
    """stage[t+k] within the same segment, plus a validity mask."""
    n = len(stage)
    tgt = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    if k < n:
        same = seg[k:] == seg[:-k]
        tgt[:-k] = stage[k:]
        ok[:-k] = same
    return tgt, ok
