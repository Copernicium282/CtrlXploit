"""Leakage-safe split protocols.

temporal - deployment shape: within every capture, the first 70 % of *time* trains,
           the next 15 % validates, the last 15 % tests; `purge` windows around each
           cut are discarded so no context window or target horizon straddles a
           boundary (purging: López de Prado, 2018).
family   - generalisation shape: whole captures (malware families / attack days)
           are held out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(w: pd.DataFrame, fractions: dict, purge: int) -> np.ndarray:
    out = np.full(len(w), "purged", dtype=object)
    tr, va = fractions["train"], fractions["train"] + fractions["val"]
    # many independent network-mode episodes in one capture -> split whole episodes in time order
    if "host" not in w and w.groupby("capture")["segment"].nunique().min() >= 5:
        for _, idx in w.groupby("capture", sort=False).indices.items():
            seg = w["segment"].to_numpy()[idx]
            order = list(dict.fromkeys(seg[np.argsort(w["ts"].to_numpy()[idx], kind="stable")]))
            n = len(order)
            a, b = max(int(round(n * tr)), 1), max(min(int(round(n * va)), n - 1), 1)
            lab = {s: ("train" if i < a else "val" if i < b else "test") for i, s in enumerate(order)}
            out[idx] = [lab[s] for s in seg]
        return out
    wid = w["wid"].to_numpy()
    for _, idx in w.groupby("capture", sort=False).indices.items():
        lo, hi = wid[idx].min(), wid[idx].max() + 1
        c1, c2 = lo + int((hi - lo) * tr), lo + int((hi - lo) * va)
        x = wid[idx]
        lab = np.where(x < c1, "train", np.where(x < c2, "val", "test")).astype(object)
        for c in (c1, c2):
            lab[(x >= c - purge) & (x < c)] = "purged"
        out[idx] = lab
    return out


def family_split(w: pd.DataFrame, groups: dict) -> np.ndarray:
    cap = w["capture"].astype(str).to_numpy()
    out = np.full(len(w), "unused", dtype=object)
    for name in ("train", "val", "test"):
        out[np.isin(cap, [str(g) for g in groups.get(name, [])])] = name
    return out


def assign_splits(w: pd.DataFrame, fractions: dict, purge: int, protocol: str = "temporal",
                  groups: dict | None = None) -> pd.Series:
    if "capture" not in w:
        w = w.assign(capture="0")
    if protocol == "family":
        if not groups:
            raise ValueError("family protocol needs datasets.<name>.family_split")
        return pd.Series(family_split(w, groups), index=w.index)
    return pd.Series(temporal_split(w, fractions, purge), index=w.index)
