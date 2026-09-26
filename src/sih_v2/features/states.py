"""Multi-scale temporal state vectors and forecasting targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import EXPLOIT_STAGES
from .windows import BASE_FEATURES

SCALES = ("", "__dev_short", "__trend_long")


def state_feature_names() -> list[str]:
    return [f + s for s in SCALES for f in BASE_FEATURES]


def make_states(w: pd.DataFrame, ema_short: int, ema_long: int) -> pd.DataFrame:
    """State s_t = [x_t, x_t - EMA_short(x)_{t-1}, EMA_long(x)_t] per segment (causal)."""
    out = []
    for _, part in w.groupby("segment", sort=False):
        x = part[BASE_FEATURES].astype("float64")
        short = x.ewm(span=ema_short, adjust=False).mean().shift(1).fillna(x.iloc[0])
        long = x.ewm(span=ema_long, adjust=False).mean()
        dev = (x - short).add_suffix("__dev_short")
        trend = long.add_suffix("__trend_long")
        out.append(pd.concat([dev, trend], axis=1))
    extra = pd.concat(out).loc[w.index]
    return pd.concat([w, extra.astype("float32")], axis=1)


def add_targets(w: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """y_future[t] = 1 if an exploitation-stage window occurs in (t, t+H].

    phase[t]: 'benign'  - no exploitation now or within H
              'pre_attack'   - not yet compromised but exploitation within H (true early warning)
              'during_attack' - active exploitation in window t
    """
    w = w.copy()
    exploit = w["stage"].isin(EXPLOIT_STAGES).to_numpy()
    y = np.zeros(len(w), dtype=np.int8)
    tta = np.full(len(w), -1, dtype=np.int32)   # windows until next exploitation (within H)
    for _, idx in w.groupby("segment", sort=False).indices.items():
        e = exploit[idx]
        n = len(idx)
        nxt = np.full(n, np.iinfo(np.int32).max)
        last = np.iinfo(np.int32).max
        for i in range(n - 1, -1, -1):
            nxt[i] = last
            if e[i]:
                last = i
        d = nxt - np.arange(n)
        hit = d <= horizon
        y[idx] = hit
        tta[idx] = np.where(hit, d, -1)
    w["y_future"] = y
    w["time_to_attack"] = tta
    w["exploit_now"] = exploit.astype(np.int8)
    w["phase"] = np.where(exploit, "during_attack", np.where(y == 1, "pre_attack", "benign"))
    return w
