"""Head-to-head evaluation under one protocol.

Every method sees identical features, scaler, labels, splits and threshold rule
(F1-optimal subject to FPR <= max_fpr on validation, frozen before test)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from ..config import seed_paths
from ..constants import EXPLOIT_STAGES
from ..models.baselines import PersistenceBaseline
from ..models.trainer import SeqClassifier, direct_scores
from . import leakage
from .bundle import Bundle, load_bundle
from .metrics import bootstrap_ci, evaluate_method, host_triage, stage_horizon_table
from .simulate import ForecastEngine, ForecastResult

log = logging.getLogger(__name__)

WM_VARIANTS = [("NetWorldModel - direct head", "risk_direct", "direct"),
               ("NetWorldModel - K-step rollout", "risk_rollout", "rollout"),
               ("Markov kill-chain prior", "risk_markov", "markov"),
               ("NetWorldModel - fused (ours)", "risk", "fused")]
OURS = "NetWorldModel - fused (ours)"
ENSEMBLE = "NetWorldModel ⊕ tree ensemble (ours, ensemble)"


def _lstm(entry) -> SeqClassifier:
    m = SeqClassifier(**entry["hparams"])
    m.load_state_dict(entry["state_dict"])
    return m.eval()


def aggregate(per_seed: list[dict]) -> dict:
    """mean and std across seeds for the scalar metrics of one method."""
    keys = ["f1", "precision", "recall", "fpr", "roc_auc", "pr_auc", "ew_pr_auc"]
    out = {k: float(np.nanmean([r[k] for r in per_seed])) for k in keys}
    out.update({f"{k}_std": float(np.nanstd([r[k] for r in per_seed])) for k in keys})
    for ph in ("pre_attack", "during_attack", "benign"):
        out[f"{ph}_alert_rate"] = float(np.nanmean([r["phases"][ph]["alert_rate"] for r in per_seed]))
    for k in ("warned_before_onset", "mean_lead_min"):
        out[k] = float(np.nanmean([r["lead_time"][k] for r in per_seed]))
    out["incidents"] = per_seed[0]["lead_time"]["incidents"]
    out["seeds"] = len(per_seed)
    return out


def evaluate_protocol(cfg: dict, test: pd.DataFrame, train: pd.DataFrame, bundle: Bundle, pack: dict,
                      seeds: list[int]) -> tuple[dict, ForecastResult]:
    H, W = cfg["forecast"]["horizon"], cfg["features"]["window_seconds"]
    L = cfg["forecast"]["context"]
    primary = ForecastEngine(bundle, seed=int(bundle.meta.get("seed", 0))).run(test)
    fr = primary.frame
    X, seg = primary.X, test["segment"].to_numpy()
    per_method: dict[str, list[dict]] = {}
    scores: dict[str, tuple] = {}

    def add(name, score, thr, keep=True):
        r = evaluate_method(name, fr, score, thr, H, W)
        per_method.setdefault(name, []).append(r)
        if keep and name not in scores:
            scores[name] = (np.asarray(score), thr)

    add(PersistenceBaseline.name, PersistenceBaseline().score(fr["stage"]), 0.5)
    for e in pack["sklearn"]:
        add(e["model"].name, e["model"].score(X, seg), e["threshold"])
    for sd in seeds:
        if sd in pack.get("lstm", {}):
            ent = pack["lstm"][sd]
            add("LSTM classifier (no world model)", direct_scores(_lstm(ent), X, seg, L), ent["threshold"],
                keep=sd == seeds[0])
        p = seed_paths(cfg, sd)["model_bundle"]
        if not p.exists():
            continue
        b = load_bundle(p)
        f = fr if sd == bundle.meta.get("seed") else ForecastEngine(b, seed=sd).run(test).frame
        for name, col, key in WM_VARIANTS:
            add(name, f[col], b.thresholds[key], keep=sd == bundle.meta.get("seed"))
        ens = b.meta.get("ensemble")
        if ens:
            tree = next(e for e in pack["sklearn"] if e["model"].name == ens["partner"])
            s_ens = ens["w_world_model"] * f["risk"].to_numpy() + (1 - ens["w_world_model"]) * tree["model"].score(X, seg)
            add(ENSEMBLE, s_ens, ens["threshold"], keep=sd == bundle.meta.get("seed"))

    summary = []
    for name, runs in per_method.items():
        agg = aggregate(runs)
        s, thr = scores[name]
        agg["ci95"] = bootstrap_ci(fr, s, thr, B=cfg.get("bootstrap", 200))
        agg["method"] = name
        agg["curves"] = runs[0]["curves"]
        agg["lead_detail"] = runs[0]["lead_time"]["detail"]
        summary.append(agg)
        fr[f"score::{name}"] = s

    # ---------------------------------------------------------------- stage forecasting per horizon
    horizons = [k for k in (1, 2, 3, 5, 7, 10) if k <= primary.rollout_mean.shape[1]]
    stage_lr = pack.get("stage_lr")
    preds = {
        "NetWorldModel rollout (ours)": lambda k: primary.rollout_mean[:, k - 1].argmax(-1),
        "Markov kill-chain prior": lambda k: primary.markov_path[:, k - 1].argmax(-1),
        "Persistence (inferred current stage)": lambda k: primary.stage_now.argmax(-1),
        "Persistence (oracle current stage)": lambda k: fr["stage"].to_numpy(),
    }
    if stage_lr is not None:
        preds["Stage LR per horizon (stacked)"] = lambda k: stage_lr.predict(X, seg, k) if k in stage_lr.models \
            else np.zeros(len(X), int)
    stage_tab = stage_horizon_table(fr, preds, horizons)

    # ---------------------------------------------------------------- surprise / anomaly channel
    sign = float(bundle.meta.get("surprise_sign", 1.0))
    mal = (fr["stage"] > 0).to_numpy()
    surprise = {"sign": sign, "val_auc_raw": bundle.meta.get("surprise_val_auc")}
    if 0 < mal.mean() < 1:
        surprise["window_auc"] = float(roc_auc_score(mal, sign * fr["surprise_obs"]))
        surprise["window_auc_kl"] = float(roc_auc_score(mal, sign * fr["surprise_kl"]))
    triage = host_triage(fr, bundle.threshold, float(cfg["engine"].get("surprise_z", 3.0)), sign)
    tri_tab = triage.pop("table")
    if tri_tab["infected"].nunique() > 1:
        surprise["host_auc"] = float(roc_auc_score(tri_tab["infected"], tri_tab["surprise_z"]))
        surprise["host_auc_risk"] = float(roc_auc_score(tri_tab["infected"], tri_tab["max_risk"]))

    # ---------------------------------------------------------------- integrity checks
    ours = next(r for r in summary if r["method"] == OURS)
    checks = [leakage.split_integrity(pd.concat([train.assign(_s="train"), test.assign(_s="test")]), "_s",
                                      cfg["protocol"]),
              leakage.schedule_probe(train, test, ours["pr_auc"]),
              leakage.identity_probe(train, test),
              leakage.permutation_test(scores[OURS][0], test["y_future"].to_numpy()),
              leakage.no_peeking(bundle.model, X.shape[1], L, cfg["engine"]["rollout_steps"])]
    ben = test[test.groupby("segment")["stage"].transform("max") == 0]
    if len(ben):
        segs = ben["segment"].drop_duplicates().sample(min(40, ben["segment"].nunique()), random_state=0)
        checks.append(leakage.ood_benign(ForecastEngine(bundle, particles=8), ben[ben["segment"].isin(segs)], cfg))
    for c in checks:
        c.pop("_features", None)
    return {"summary": summary, "stage_horizon": stage_tab.to_dict("records"), "triage": triage,
            "triage_table": tri_tab, "surprise": surprise, "checks": checks}, primary


def summary_table(rows: list[dict]) -> pd.DataFrame:
    def pm(r, k):
        return r.get(k, float("nan"))
    return pd.DataFrame([{
        "Method": r["method"], "Seeds": r.get("seeds", 1),
        "F1": pm(r, "f1"), "F1 ±": pm(r, "f1_std"), "F1 95% CI": r.get("ci95", {}).get("f1"),
        "Precision": pm(r, "precision"), "Recall": pm(r, "recall"), "FPR": pm(r, "fpr"),
        "ROC-AUC": pm(r, "roc_auc"), "PR-AUC": pm(r, "pr_auc"), "PR-AUC ±": pm(r, "pr_auc_std"),
        "EW PR-AUC": pm(r, "ew_pr_auc"),
        "Pre-attack recall": pm(r, "pre_attack_alert_rate"), "During-attack recall": pm(r, "during_attack_alert_rate"),
        "Incidents warned": pm(r, "warned_before_onset"), "Mean lead (min)": pm(r, "mean_lead_min"),
    } for r in rows])


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]

    def fmt(v):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return f"[{v[0]:.3f}, {v[1]:.3f}]"
        if isinstance(v, (float, np.floating)):
            return "–" if v != v else f"{v:.3f}"
        return str(v)
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def exploit_mask(stage) -> np.ndarray:
    return np.isin(np.asarray(stage), EXPLOIT_STAGES)


__all__ = ["evaluate_protocol", "summary_table", "to_markdown", "OURS", "WM_VARIANTS", "torch"]
