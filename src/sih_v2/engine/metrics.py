"""Forecasting metrics: F1/P/R/FPR, ROC/PR AUC, phase-stratified rates and early-warning lead time."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve


def binary_metrics(y, pred) -> dict:
    y, p = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, fp, fn, tn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
            "fpr": float(fp / max(fp + tn, 1)), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def ranking_metrics(y, score) -> dict:
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score))}


def curves(y, score, n: int = 120) -> dict:
    fpr, tpr, _ = roc_curve(y, score)
    pr, rc, _ = precision_recall_curve(y, score)
    pick = lambda a: np.asarray(a)[np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))].tolist()
    return {"roc": {"fpr": pick(fpr), "tpr": pick(tpr)}, "pr": {"precision": pick(pr), "recall": pick(rc)}}


def phase_metrics(frame: pd.DataFrame, pred) -> dict:
    """Separate TRUE early warning (pre-attack) from detection of ongoing attacks."""
    p = np.asarray(pred).astype(bool)
    out = {}
    for ph in ("pre_attack", "during_attack", "benign"):
        m = (frame["phase"] == ph).to_numpy()
        out[ph] = {"windows": int(m.sum()), "alert_rate": float(p[m].mean()) if m.any() else float("nan")}
    return out


def lead_times(frame: pd.DataFrame, pred, horizon: int, window_seconds: float) -> dict:
    """Incident = exploitation onset with no exploitation in the previous H windows.
    Lead time = number of consecutive alerted windows immediately before onset (capped at H)."""
    p = np.asarray(pred).astype(bool)
    leads, rows = [], []
    for seg, idx in frame.groupby("segment", sort=False).indices.items():
        ex = frame["exploit_now"].to_numpy()[idx].astype(bool)
        pp = p[idx]
        for o in np.flatnonzero(ex):
            if ex[max(0, o - horizon):o].any() or o == 0:
                continue
            k = 0
            while k < horizon and o - k - 1 >= 0 and pp[o - k - 1]:
                k += 1
            leads.append(k)
            rows.append({"segment": int(seg), "onset_index": int(o), "lead_windows": k,
                         "alert_at_onset": bool(pp[o])})
    leads = np.asarray(leads, float)
    mins = leads * window_seconds / 60.0
    return {
        "incidents": int(len(leads)),
        "warned_before_onset": float((leads > 0).mean()) if len(leads) else float("nan"),
        "mean_lead_min": float(mins.mean()) if len(leads) else float("nan"),
        "median_lead_min": float(np.median(mins)) if len(leads) else float("nan"),
        "mean_lead_min_when_warned": float(mins[leads > 0].mean()) if (leads > 0).any() else 0.0,
        "detail": rows,
    }


def evaluate_method(name, frame, score, threshold, horizon, window_seconds) -> dict:
    y = frame["y_future"].to_numpy()
    pred = np.asarray(score) >= threshold
    r = {"method": name, "threshold": float(threshold), **binary_metrics(y, pred), **ranking_metrics(y, score)}
    ew = (frame["exploit_now"] == 0).to_numpy()   # early-warning subset: not yet compromised
    ewm = ranking_metrics(y[ew], np.asarray(score)[ew])
    r["ew_pr_auc"], r["ew_roc_auc"] = ewm["pr_auc"], ewm["roc_auc"]
    r["phases"] = phase_metrics(frame, pred)
    r["lead_time"] = lead_times(frame, pred, horizon, window_seconds)
    r["curves"] = curves(y, score) if len(np.unique(y)) > 1 else None
    return r


# ----------------------------------------------------------------------------- stage forecasting
def macro_f1_present(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    labels = np.unique(y_true)
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) if len(labels) else float("nan")


def stage_horizon_table(frame: pd.DataFrame, predictors: dict, horizons, restrict_malicious: bool = True) -> pd.DataFrame:
    """Stage forecasting at t+k:

    predictors: name -> callable(k) returning predicted stage ids for every row.
    Scored on host-slots that ever carry malicious traffic (benign cells would
    otherwise make every method look perfect). 'transition acc' is accuracy on
    cells whose stage at t+k differs from the stage at t - persistence scores 0
    there by construction, so it isolates genuine dynamics."""
    from ..models.baselines import future_stage

    seg = frame["segment"].to_numpy()
    stage = frame["stage"].to_numpy()
    mal_seg = frame.groupby("segment")["stage"].transform("max").to_numpy() > 0
    rows = []
    for k in horizons:
        tgt, ok = future_stage(stage, seg, k)
        m = ok & (mal_seg if restrict_malicious else True)
        if not m.any():
            continue
        trans = m & (tgt != stage)
        for name, fn in predictors.items():
            p = np.asarray(fn(k))
            rows.append({"k": k, "method": name, "macro_f1": macro_f1_present(tgt[m], p[m]),
                         "accuracy": float((tgt[m] == p[m]).mean()),
                         "transition_acc": float((tgt[trans] == p[trans]).mean()) if trans.any() else float("nan"),
                         "n": int(m.sum()), "n_transitions": int(trans.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- host triage
def host_triage(frame: pd.DataFrame, threshold: float, z_thr: float, sign: float = 1.0) -> dict:
    """Host-slot level, two independent channels:
    risk    - any window of the host-slot crosses the calibrated risk threshold
    anomaly - the host-slot's mean surprise is > z_thr robust deviations above the
              median host-slot of the same capture (label-free)"""
    from .simulate import robust_z

    g = frame.groupby("segment")
    h = pd.DataFrame({"capture": g["capture"].first() if "capture" in frame else 0,
                      "host": g["host"].first() if "host" in frame else g["segment"].first(),
                      "infected": g["stage"].max() > 0, "max_risk": g["risk"].max(),
                      "surprise": g["surprise_obs"].mean(), "cells": g.size()})
    h["surprise_z"] = sign * robust_z(h["surprise"], h["capture"])
    h["by_risk"] = h["max_risk"] >= threshold
    h["by_anomaly"] = h["surprise_z"] >= z_thr
    h["flagged"] = h["by_risk"] | h["by_anomaly"]
    inf, ben = h[h["infected"]], h[~h["infected"]]
    return {
        "host_slots": int(len(h)), "infected": int(len(inf)),
        "caught": int(inf["flagged"].sum()), "caught_by_risk": int(inf["by_risk"].sum()),
        "caught_only_by_anomaly": int((inf["by_anomaly"] & ~inf["by_risk"]).sum()),
        "false_alarms": int(ben["flagged"].sum()), "benign": int(len(ben)),
        "false_alarm_rate": float(ben["flagged"].mean()) if len(ben) else float("nan"),
        "risk_only_false_alarms": int(ben["by_risk"].sum()),
        "table": h.reset_index(),
    }


# ----------------------------------------------------------------------------- uncertainty
def bootstrap_ci(frame: pd.DataFrame, score, threshold: float, B: int = 200, seed: int = 0) -> dict:
    """95 % CI by resampling whole host-slots/segments (cells within a segment are correlated)."""
    rng = np.random.default_rng(seed)
    segs = frame["segment"].to_numpy()
    uniq, inv = np.unique(segs, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    y = frame["y_future"].to_numpy()
    s = np.asarray(score)
    vals = {"f1": [], "pr_auc": [], "recall": [], "fpr": []}
    for _ in range(B):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        yy, ss = y[pick], s[pick]
        bm = binary_metrics(yy, ss >= threshold)
        vals["f1"].append(bm["f1"]); vals["recall"].append(bm["recall"]); vals["fpr"].append(bm["fpr"])
        vals["pr_auc"].append(ranking_metrics(yy, ss)["pr_auc"])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}
