"""End-to-end glue: raw telemetry file(s) -> window states -> forecasts."""
from __future__ import annotations

import pandas as pd

from ..features.states import add_targets, make_states
from ..features.windows import DEFAULT_INTERNAL, build_windows_from_files
from .bundle import Bundle
from .simulate import ForecastEngine, ForecastResult


def windows_to_states(w: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    w = make_states(w, f["ema_short"], f["ema_long"])
    return add_targets(w, cfg["forecast"]["horizon"])


def states_from_files(paths, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Raw files -> state cells at the granularity the model was trained on (network or per-host).
    For uploads in host mode, private address ranges count as monitored hosts in addition to the
    training network's prefixes."""
    d = cfg.get("data", {"mode": "network"})
    host = d.get("mode") == "host"
    internal = tuple(d.get("internal_prefixes", ())) + DEFAULT_INTERNAL
    w, stats = build_windows_from_files(
        paths, cfg["features"]["window_seconds"], cfg["ingest"]["chunksize"], cfg["ingest"]["segment_gap_windows"],
        by_host=host, host_prefixes=internal, internal_prefixes=internal,
        max_benign_hosts=d.get("max_benign_hosts") if host else None,
        min_active_windows=d.get("min_active_windows", 1), stage_rule=d.get("stage_rule", "dominant"))
    if w.empty:
        raise ValueError("No valid flows could be parsed from the input")
    return windows_to_states(w, cfg), stats


def forecast_files(paths, bundle: Bundle, **engine_kw) -> tuple[ForecastResult, pd.DataFrame, dict]:
    states, stats = states_from_files(paths, bundle.config)
    return ForecastEngine(bundle, **engine_kw).run(states), states, stats
