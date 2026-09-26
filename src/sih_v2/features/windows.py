"""Aggregation of canonical flows into fixed-length telemetry state cells.

Two granularities:
  * network mode - one cell per (capture, window): the whole monitored network
  * host mode    - one cell per (capture, source host, window): per-host state
                   trajectories, so a compromised machine's history is not diluted
                   by thousands of benign neighbours

Out-of-core design: pass 1 streams chunks and spills rows into hourly parquet
buckets (so arbitrarily unsorted files are handled exactly); pass 2 loads one
bucket at a time and aggregates it. Peak memory = one chunk + one hour of flows.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import LATERAL_PORTS, N_STAGES
from ..ingest.csv_reader import iter_flow_chunks, list_inputs

log = logging.getLogger(__name__)

BASE_FEATURES = [
    "log_flows", "log_bytes", "log_pkts", "log_mean_duration", "log_bytes_per_flow",
    "n_src_ips", "n_dst_ips", "n_dst_ports", "dst_port_entropy", "dst_ip_entropy",
    "syn_ratio", "ack_ratio", "fin_ratio", "rst_ratio", "psh_ratio", "urg_ratio",
    "flag_bitmask_entropy", "ttl_mean", "ttl_std", "init_win_mean", "init_win_std",
    "log_iat_mean", "log_iat_std", "retrans_rate", "small_flow_frac", "log_out_in_ratio",
    "lateral_frac", "sensitive_port_frac", "udp_frac", "dns_frac", "beacon_score",
]
STAGE_COUNT_COLS = [f"cnt_{i}" for i in range(N_STAGES)]
SENSITIVE_PORTS = [21, 22, 23, 135, 139, 445, 1433, 3306, 3389, 5900, 5985]
DEFAULT_INTERNAL = ("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2", "172.30.", "172.31.")

# Window stage rules.
#  dominant: the malicious stage with the most flows (HowSuyash/AttackForecast's argument:
#            a spam bot emits thousands of spam flows and a few C2 flows per window, so a
#            "most severe present" rule makes the kill chain teleport to its end and freeze)
#  priority: most salient stage present (useful when a rare, high-impact stage such as a
#            handful of exfiltration flows co-occurs with continuous C2 beaconing)
STAGE_PRIORITY = [6, 5, 3, 2, 4, 1]  # Impact > Exfil > Lateral > InitialAccess > C2 > Recon


def is_internal(ip: pd.Series, prefixes=DEFAULT_INTERNAL) -> np.ndarray:
    s = ip.astype(str)
    out = np.zeros(len(s), dtype=bool)
    for p in prefixes:
        out |= s.str.startswith(p).to_numpy()
    return out


def _entropy(df: pd.DataFrame, keys: list[str], col: str) -> pd.Series:
    c = df.groupby([*keys, col], sort=False, observed=True).size()
    lv = list(range(len(keys)))
    p = c / c.groupby(level=lv).transform("sum")
    return (-(p * np.log2(p))).groupby(level=lv).sum()


def compute_window_features(df: pd.DataFrame, window_seconds: float, by_host: bool = False,
                            internal_prefixes=DEFAULT_INTERNAL) -> pd.DataFrame:
    """Canonical flows -> one row per cell with BASE_FEATURES + per-stage flow counts."""
    keys = ["capture", "host", "wid"] if by_host else ["capture", "wid"]
    if df.empty:
        return pd.DataFrame(columns=[*keys, *BASE_FEATURES, *STAGE_COUNT_COLS])
    df = df.copy()
    if "capture" not in df:
        df["capture"] = "0"
    if by_host:
        df["host"] = df["src_ip"]
    df["wid"] = np.floor(df["ts"].to_numpy() / window_seconds).astype(np.int64)
    pk = df["fwd_pkts"] + df["bwd_pkts"]
    df["pkts"] = pk
    df["bytes"] = df["fwd_bytes"] + df["bwd_bytes"]
    df["small"] = (pk <= 2).astype("float32")
    df["udp"] = (df["proto"] == 17).astype("float32")
    df["dns"] = (df["dst_port"] == 53).astype("float32")
    df["lateral"] = (is_internal(df["src_ip"], internal_prefixes) & is_internal(df["dst_ip"], internal_prefixes)
                     & ~df["dst_port"].isin([53, 123]).to_numpy()).astype("float32")
    df["sensitive"] = df["dst_port"].isin(SENSITIVE_PORTS).astype("float32")
    df["bitmask"] = ((df["syn"] > 0) * 1 + (df["ack"] > 0) * 2 + (df["fin"] > 0) * 4
                     + (df["rst"] > 0) * 8 + (df["psh"] > 0) * 16 + (df["urg"] > 0) * 32).astype("int8")
    # behavioural rule: malicious internal->internal remote-service traffic is Lateral Movement (T1021)
    lat = ((df["stage"] > 0).to_numpy() & (df["lateral"] > 0).to_numpy()
           & df["dst_port"].isin(LATERAL_PORTS).to_numpy())
    if lat.any():
        df.loc[lat, "stage"] = 3
    tcp = df["proto"] == 6
    df["win_tcp"] = df["init_win"].where(tcp & (df["init_win"] > 0))
    df["ttl_sq"] = df["ttl_std"] ** 2

    g = df.groupby(keys, sort=True)
    s = g[["pkts", "bytes", "fwd_bytes", "bwd_bytes", "syn", "ack", "fin", "rst", "psh", "urg",
           "retrans", "small", "udp", "dns", "lateral", "sensitive"]].sum()
    n = g.size()
    m = g[["duration", "iat_mean", "iat_std", "ttl_mean", "ttl_sq", "win_tcp"]].mean()
    ttl_var_between = g["ttl_mean"].var(ddof=0)
    win_std = g["win_tcp"].std(ddof=0)
    nun = g[["src_ip", "dst_ip", "dst_port"]].nunique()
    tp = s["pkts"].clip(lower=1)

    out = pd.DataFrame(index=n.index)
    out["log_flows"] = np.log1p(n)
    out["log_bytes"] = np.log1p(s["bytes"])
    out["log_pkts"] = np.log1p(s["pkts"])
    out["log_mean_duration"] = np.log1p(m["duration"])
    out["log_bytes_per_flow"] = np.log1p(s["bytes"] / n)
    out["n_src_ips"] = np.log1p(nun["src_ip"])
    out["n_dst_ips"] = np.log1p(nun["dst_ip"])
    out["n_dst_ports"] = np.log1p(nun["dst_port"])
    out["dst_port_entropy"] = _entropy(df, keys, "dst_port")
    out["dst_ip_entropy"] = _entropy(df, keys, "dst_ip")
    for f in ("syn", "ack", "fin", "rst", "psh", "urg"):
        out[f"{f}_ratio"] = s[f] / tp
    out["flag_bitmask_entropy"] = _entropy(df, keys, "bitmask")
    # TTL: law of total variance = within-flow variance + between-flow variance
    out["ttl_mean"] = m["ttl_mean"]
    out["ttl_std"] = np.sqrt(m["ttl_sq"].fillna(0) + ttl_var_between.fillna(0))
    out["init_win_mean"] = np.log1p(m["win_tcp"])
    out["init_win_std"] = np.log1p(win_std)
    out["log_iat_mean"] = np.log1p(m["iat_mean"] * 1e3)   # ms
    out["log_iat_std"] = np.log1p(m["iat_std"] * 1e3)
    out["retrans_rate"] = s["retrans"] / tp
    out["small_flow_frac"] = s["small"] / n
    out["log_out_in_ratio"] = np.log1p(s["fwd_bytes"]) - np.log1p(s["bwd_bytes"])
    out["lateral_frac"] = s["lateral"] / n
    out["sensitive_port_frac"] = s["sensitive"] / n
    out["udp_frac"] = s["udp"] / n
    out["dns_frac"] = s["dns"] / n
    out["beacon_score"] = _beacon_score(df, keys)
    out = out.reindex(columns=BASE_FEATURES).fillna(0.0).astype("float32")
    out["ttl_missing"] = m["ttl_mean"].isna().astype("int8")  # CICFlowMeter / Argus have no TTL

    cnt = df.groupby([*keys, "stage"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(columns=range(N_STAGES), fill_value=0)
    cnt.columns = STAGE_COUNT_COLS
    out = out.join(cnt).fillna(0)
    out["n_flows_raw"] = n
    return out.reset_index()


def _beacon_score(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Periodicity of repeated (src,dst,dport) conversations: 1/(1+CV of start-time gaps)."""
    conv = ["capture", "src_ip", "dst_ip", "dst_port"]
    d = df[list(dict.fromkeys([*keys, *conv, "ts"]))].sort_values("ts")
    d["gap"] = d.groupby(conv, sort=False)["ts"].diff()
    d = d[d["gap"].notna() & (d["gap"] > 0.5)]
    if d.empty:
        return pd.Series(dtype="float32")
    a = d.groupby(list(dict.fromkeys([*keys, *conv[1:]])), sort=False)["gap"].agg(["mean", "std", "count"])
    a = a[a["count"] >= 3]
    score = (1.0 / (1.0 + a["std"] / a["mean"])) * np.minimum(a["count"] / 6.0, 1.0)
    return score.groupby(level=list(range(len(keys)))).max()


def stage_from_counts(w: pd.DataFrame, rule: str = "dominant") -> np.ndarray:
    atk = w[STAGE_COUNT_COLS[1:]].to_numpy()
    if rule == "dominant":
        return np.where(atk.sum(1) > 0, atk.argmax(1) + 1, 0).astype(np.int64)
    stage = np.zeros(len(w), dtype=np.int64)
    for s in reversed(STAGE_PRIORITY):
        stage = np.where(w[f"cnt_{s}"].to_numpy() > 0, s, stage)
    return stage


def select_hosts(w: pd.DataFrame, max_benign_hosts: int | None, min_active_windows: int) -> pd.DataFrame:
    """Keep every host that ever carries malicious traffic + the most active benign hosts."""
    if "host" not in w or max_benign_hosts is None:
        return w
    agg = w.groupby(["capture", "host"]).agg(active=("wid", "size"), flows=("n_flows_raw", "sum"),
                                              mal=("n_flows_raw", lambda s: 0))
    mal = (w[STAGE_COUNT_COLS[1:]].sum(1) > 0).groupby([w["capture"], w["host"]]).any()
    agg["mal"] = mal
    keep = [agg[agg["mal"]]]                               # infected hosts are always kept
    agg = agg[agg["active"] >= min_active_windows]
    for cap, part in agg[~agg["mal"]].groupby(level=0):
        keep.append(part.sort_values("flows", ascending=False).head(max_benign_hosts))
    idx = pd.concat(keep).index
    return w.set_index(["capture", "host"]).loc[lambda d: d.index.isin(idx)].reset_index()


def finalize_windows(w: pd.DataFrame, window_seconds: float, segment_gap: int,
                     stage_rule: str = "dominant") -> pd.DataFrame:
    """Merge duplicate cells, fill empty windows inside segments, derive stage & segment ids.

    A segment is a contiguous activity run of one (capture[, host]); a silence longer
    than `segment_gap` windows starts a new segment."""
    if w.empty:
        return w
    if "capture" not in w:
        w = w.assign(capture="0")
    gkeys = ["capture", "host"] if "host" in w else ["capture"]
    if w.duplicated([*gkeys, "wid"]).any():
        w = w.groupby([*gkeys, "wid"], as_index=False).first()
    w = w.sort_values([*gkeys, "wid"]).reset_index(drop=True)
    frames, sid = [], 0
    feat_cols = [c for c in w.columns if c not in (*gkeys, "wid")]
    for key, grp in w.groupby(gkeys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        brk = (grp["wid"].diff().fillna(1) > segment_gap).cumsum().to_numpy()
        for _, part in grp.groupby(brk, sort=True):
            full = pd.RangeIndex(part["wid"].min(), part["wid"].max() + 1)
            part = part.set_index("wid")[feat_cols].reindex(full).fillna(0.0)
            part.index.name = "wid"
            part = part.reset_index()
            for k, v in zip(gkeys, key):
                part[k] = v
            part["segment"] = sid
            sid += 1
            frames.append(part)
    w = pd.concat(frames, ignore_index=True)
    w["ts"] = w["wid"] * window_seconds
    w["stage"] = stage_from_counts(w, stage_rule)
    return w


def build_windows_from_frame(flows: pd.DataFrame, window_seconds: float, segment_gap: int = 10,
                             by_host: bool = False, stage_rule: str = "dominant",
                             internal_prefixes=DEFAULT_INTERNAL) -> pd.DataFrame:
    return finalize_windows(compute_window_features(flows, window_seconds, by_host, internal_prefixes),
                            window_seconds, segment_gap, stage_rule)


def build_windows_from_files(paths, window_seconds: float, chunksize: int = 200_000,
                             segment_gap: int = 10, bucket_windows: int = 60,
                             tmp_dir: str | None = None, by_host: bool = False,
                             host_prefixes=None, internal_prefixes=DEFAULT_INTERNAL,
                             max_benign_hosts: int | None = None, min_active_windows: int = 1,
                             stage_rule: str = "dominant", capture_names: dict | None = None,
                             ) -> tuple[pd.DataFrame, dict]:
    """Two-pass out-of-core cell builder over any number of large files.

    host mode keeps only flows whose source address starts with `host_prefixes`
    (the monitored network) before anything is written to disk."""
    files = list_inputs(paths)
    if not files:
        raise FileNotFoundError(f"No telemetry files found in {paths}")
    work = Path(tempfile.mkdtemp(prefix="sih_buckets_", dir=tmp_dir))
    stats = {"files": [str(f) for f in files], "flows": 0, "flows_kept": 0, "chunks": 0, "skipped_files": []}
    hp = tuple(host_prefixes) if host_prefixes else tuple(internal_prefixes)
    try:
        part = 0
        for f in files:
            cap = (capture_names or {}).get(str(f), Path(f).stem)
            try:
                for chunk in iter_flow_chunks(f, chunksize):
                    if chunk.empty:
                        continue
                    stats["chunks"] += 1
                    stats["flows"] += len(chunk)
                    if by_host:
                        chunk = chunk[is_internal(chunk["src_ip"], hp)]
                    stats["flows_kept"] += len(chunk)
                    chunk = chunk.assign(capture=cap)
                    b = (np.floor(chunk["ts"] / window_seconds) // bucket_windows).astype(np.int64)
                    for bid, sub in chunk.groupby(b):
                        d = work / f"b={bid}"
                        d.mkdir(exist_ok=True)
                        sub.to_parquet(d / f"part{part:06d}.parquet", index=False)
                        part += 1
            except Exception as exc:  # one broken file must not kill a multi-day build
                log.error("skipping %s: %s", f, exc)
                stats["skipped_files"].append(f"{f}: {exc}")
        feats = []
        for d in sorted(work.iterdir(), key=lambda p: int(p.name[2:])):
            flows = pd.concat([pd.read_parquet(p) for p in sorted(d.iterdir())], ignore_index=True)
            feats.append(compute_window_features(flows, window_seconds, by_host, internal_prefixes))
        w = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if by_host and len(w):
        w = select_hosts(w, max_benign_hosts, min_active_windows)
    w = finalize_windows(w, window_seconds, segment_gap, stage_rule)
    stats["windows"] = int(len(w))
    stats["segments"] = int(w["segment"].nunique()) if len(w) else 0
    if by_host and len(w):
        stats["hosts"] = int(w.groupby(["capture", "host"]).ngroups)
    return w, stats
