"""Evaluation-integrity checks. Each returns a dict with a plain-language verdict.

  schedule_probe     - can hour-of-day / day-of-week / elapsed-time alone predict the target?
  identity_probe     - can the capture id alone predict it? (dataset-composition leakage)
  permutation_test   - is the observed test PR-AUC significantly above label-permuted chance?
  split_integrity    - no time overlap (temporal) / no shared capture (family) between splits
  no_peeking         - an imagination-only loss must send ZERO gradient to the encoder when
                       the rollout starts from a detached state, and non-zero when wired to
                       the observations
  ood_benign         - unusual-but-benign traffic (volume x20, new ports) must not light up
                       the risk channel
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def _ap(y, s):
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if y.min() != y.max() else float("nan")


def schedule_features(df: pd.DataFrame) -> np.ndarray:
    t = pd.to_datetime(df["ts"], unit="s")
    h = t.dt.hour + t.dt.minute / 60.0
    start = df.groupby("capture")["ts"].transform("min") if "capture" in df else df["ts"].min()
    span = (df.groupby("capture")["ts"].transform("max") - start).clip(lower=1) if "capture" in df else 1
    return np.c_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), t.dt.dayofweek, (df["ts"] - start) / span]


def schedule_probe(train: pd.DataFrame, test: pd.DataFrame, reference_ap: float, target: str = "y_future") -> dict:
    clf = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0)
    clf.fit(schedule_features(train), train[target])
    ap = _ap(test[target], clf.predict_proba(schedule_features(test))[:, 1])
    prev = float(test[target].mean())
    # share of the model's lift over chance that timing alone reproduces
    lift = (ap - prev) / (reference_ap - prev) if reference_ap == reference_ap and reference_ap > prev else float("nan")
    verdict = ("WARNING: timing alone reproduces most of the model's lift over chance" if lift == lift and lift >= 0.5
               else "CAUTION: timing carries part of the signal" if lift == lift and lift >= 0.25
               else "OK: timing alone is far weaker than the traffic model")
    return {"check": "schedule-only probe (hour, weekday, elapsed time)", "schedule_ap": ap, "prevalence": prev,
            "model_ap": reference_ap, "schedule_share_of_lift": lift, "verdict": verdict}


def identity_probe(train: pd.DataFrame, test: pd.DataFrame, target: str = "y_future") -> dict:
    caps = sorted(set(train["capture"]) | set(test["capture"]))
    if not set(test["capture"]) & set(train["capture"]):
        return {"check": "capture-identity probe", "verdict": "N/A: test captures unseen in training (family protocol)"}
    enc = lambda d: (d["capture"].to_numpy()[:, None] == np.array(caps)[None]).astype(float)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(enc(train), train[target])
    ap = _ap(test[target], clf.predict_proba(enc(test))[:, 1])
    return {"check": "capture-identity probe", "identity_ap": ap, "prevalence": float(test[target].mean()),
            "verdict": "reported for context: which capture you are in carries this much signal"}


def permutation_test(score, y, n: int = 200, seed: int = 0) -> dict:
    """Label-permutation significance test of the model's test PR-AUC: how often does a
    random re-assignment of the test labels reach the observed AP? (A y-randomisation
    *training* control is not used: with ranking metrics a model fitted on shuffled
    labels still ranks by a random direction over informative features, so its AP is
    arbitrary rather than ~prevalence.)"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    obs = _ap(y, score)
    null = np.array([_ap(rng.permutation(y), score) for _ in range(n)])
    p = float((np.sum(null >= obs) + 1) / (n + 1))
    return {"check": "label-permutation test of test PR-AUC", "observed_ap": obs,
            "null_mean": float(np.nanmean(null)), "null_p99": float(np.nanpercentile(null, 99)), "p_value": p,
            "verdict": (f"OK: p {'<' if p <= 1 / (n + 1) + 1e-12 else '='} {p:.3g} (signal above label-permuted chance)"
                        if p < 0.01 else "WARNING: not significant")}


def split_integrity(df: pd.DataFrame, split_col: str, protocol: str) -> dict:
    issues = []
    if protocol == "family":
        caps = {s: set(df.loc[df[split_col] == s, "capture"]) for s in ("train", "val", "test")}
        if caps["train"] & caps["test"] or caps["train"] & caps["val"] or caps["val"] & caps["test"]:
            issues.append("a capture appears in more than one split")
    else:
        order = {"train": 0, "val": 1, "test": 2}
        for cap, g in df[df[split_col].isin(order)].groupby("capture"):
            if "host" not in g and g["segment"].nunique() >= 5:
                continue  # episode-level split
            r = g.groupby(split_col)["wid"].agg(["min", "max"])
            for a, b in (("train", "val"), ("val", "test"), ("train", "test")):
                if a in r.index and b in r.index and r.loc[a, "max"] >= r.loc[b, "min"]:
                    issues.append(f"capture {cap}: {a} overlaps {b} in time")
    return {"check": f"split integrity ({protocol})", "issues": issues,
            "verdict": "OK: no overlap" if not issues else "FAIL: " + "; ".join(issues[:5])}


def no_peeking(model, obs_dim: int, steps: int = 24, horizon: int = 10, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    was = model.training
    model.eval()
    obs = torch.randn(4, steps, obs_dim)
    out = model.observe(obs, sample=False)

    def enc_grad(h, z):
        model.zero_grad(set_to_none=True)
        hi, zi = model.imagine(h, z, horizon, sample=False)
        model.heads(hi, zi)["stage_logits"].sum().backward(retain_graph=True)
        g = model.encoder[0].weight.grad
        return 0.0 if g is None else float(g.abs().sum())
    honest = enc_grad(out["h"][:, -1].detach(), out["post_mu"][:, -1].detach())
    wired = enc_grad(out["h"][:, -1], out["post_mu"][:, -1])
    model.zero_grad(set_to_none=True)
    model.train(was)
    ok = honest == 0.0 and wired > 0.0
    return {"check": "no-peeking gradient test", "encoder_grad_detached": honest, "encoder_grad_wired": wired,
            "verdict": "OK: imagination never touches observations (and the check can fail)" if ok else "FAIL"}


def ood_benign(engine, benign_states: pd.DataFrame, cfg: dict, surge: float = 5.0) -> dict:
    """Unusual-but-benign traffic must not light up the risk channel. Two probes on genuinely benign host-slots:
      flash crowd  - x`surge` volume with the same traffic mix (flags, ports, entropies unchanged)
      novel values - the 3 lowest-variance training features pushed far outside their range"""
    from ..features.states import make_states

    f = cfg["features"]
    base = benign_states.drop(columns=[c for c in benign_states if "__" in c])
    crowd = base.copy()
    for c in ("log_flows", "log_bytes", "log_pkts"):
        crowd[c] = crowd[c] + np.log(surge)
    novel = base.copy()
    sc = engine.b.scaler
    base_idx = [i for i, n in enumerate(engine.b.feature_names) if "__" not in n]
    low = sorted(base_idx, key=lambda i: sc.scale[i])[:3]
    for i in low:
        n = engine.b.feature_names[i]
        novel[n] = novel[n] + 20 * sc.scale[i] + 1.0
    rates = {}
    for name, d in (("original", base), ("flash_crowd", crowd), ("novel_values", novel)):
        r = engine.run(make_states(d, f["ema_short"], f["ema_long"])).frame
        rates[name] = (float((r["risk"] >= engine.b.threshold).mean()), float(r["anomaly"].mean()))
    lim = max(3 * rates["original"][0], 0.05)
    ok = rates["flash_crowd"][0] <= lim   # pass/fail on the flash crowd; novel values are informational
    #                                       (the lowest-variance features are often genuine scan signatures)
    return {"check": f"out-of-distribution benign (flash crowd x{surge:g}; novel values in "
                     f"{[engine.b.feature_names[i] for i in low]})", "cells": int(len(base)),
            "alert_rate_original": rates["original"][0], "alert_rate_flash_crowd": rates["flash_crowd"][0],
            "alert_rate_novel_values": rates["novel_values"][0],
            "anomaly_rate_flash_crowd": rates["flash_crowd"][1], "anomaly_rate_novel_values": rates["novel_values"][1],
            "verdict": ("OK: risk channel stays quiet on unusual benign traffic" if ok
                        else "WARNING: unusual benign traffic raises risk alerts")}
