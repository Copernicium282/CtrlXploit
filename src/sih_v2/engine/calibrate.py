"""Decision threshold calibration under a false-positive budget."""
from __future__ import annotations

import numpy as np


def choose_threshold(scores, y, max_fpr: float = 0.05) -> float:
    """Maximise F1 on validation subject to FPR <= max_fpr (SOC analyst budget)."""
    s, y = np.asarray(scores, float), np.asarray(y).astype(bool)
    cands = np.unique(np.quantile(s, np.linspace(0, 1, 501)))
    best, best_f1, fallback, best_fpr = 0.5, -1.0, 0.5, 2.0
    neg = max((~y).sum(), 1)
    for t in cands:
        p = s >= t
        tp, fp = (p & y).sum(), (p & ~y).sum()
        fpr = fp / neg
        f1 = 2 * tp / max(2 * tp + fp + (~p & y).sum(), 1)
        if fpr <= max_fpr and f1 > best_f1:
            best, best_f1 = t, f1
        if fpr < best_fpr:
            fallback, best_fpr = t, fpr
    return float(best if best_f1 >= 0 else fallback)


def tune_fusion(frame, step: float = 0.1) -> tuple[dict, float]:
    """Grid-search simplex weights (direct, rollout, markov) maximising validation
    early-warning PR-AUC (windows not yet under exploitation)."""
    from sklearn.metrics import average_precision_score

    m = (frame["exploit_now"] == 0).to_numpy()
    y = frame["y_future"].to_numpy()[m]
    if y.min() == y.max():
        return None, float("nan")
    comps = frame[["risk_direct", "risk_rollout", "risk_markov"]].to_numpy()[m]
    best, best_w = -1.0, None
    grid = np.round(np.arange(0, 1 + 1e-9, step), 3)
    for a in grid:
        for b in grid:
            c = round(1 - a - b, 3)
            if c < -1e-9:
                continue
            ap = average_precision_score(y, comps @ np.array([a, b, max(c, 0.0)]))
            if ap > best + 1e-6:
                best, best_w = ap, {"direct": float(a), "rollout": float(b), "markov": float(max(c, 0.0))}
    return best_w, float(best)


def fit_ensemble(val_frame, val_X, val_seg, pack: dict, max_fpr: float, select_metric: str = "mix",
                 step: float = 0.1) -> dict | None:
    """Validation-only blend of the world model's fused risk with the best tree-ensemble baseline.

    The tree ensembles on stacked history are the strongest *binary* forecasters on real CTU-13
    traffic; the world model adds stage trajectories, ETA and uncertainty. The blend weight, the
    partner model and the threshold are all chosen on validation - the test split is never seen."""
    from ..models.trainer import selection_score

    y = val_frame["y_future"].to_numpy()
    stage = val_frame["stage"].to_numpy()
    cands = [e for e in pack.get("sklearn", []) if "Forest" in e["model"].name or "Boosting" in e["model"].name]
    if not cands or y.min() == y.max():
        return None
    best = None
    for e in cands:
        s_tree = e["model"].score(val_X, val_seg)
        for w in np.round(np.arange(0, 1 + 1e-9, step), 3):
            s = w * val_frame["risk"].to_numpy() + (1 - w) * s_tree
            sc = selection_score(y, s, stage, select_metric)
            if best is None or sc > best["val_select"] + 1e-6:
                best = {"partner": e["model"].name, "w_world_model": float(w), "val_select": float(sc)}
    e = next(c for c in cands if c["model"].name == best["partner"])
    s = best["w_world_model"] * val_frame["risk"].to_numpy() + (1 - best["w_world_model"]) * e["model"].score(val_X, val_seg)
    best["threshold"] = choose_threshold(s, y, max_fpr)
    return best
