"""Canonical flow schema + alias mapping for CSE-CIC-IDS2017/2018, CTU-13, UNSW-NB15 and PCAP flows.

Every ingested source is normalised into the same per-flow frame so that the
window builder, models and dashboard never care where telemetry came from.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..constants import map_label

CANON_COLUMNS = [
    "ts", "src_ip", "dst_ip", "src_port", "dst_port", "proto", "duration",
    "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
    "syn", "ack", "fin", "rst", "psh", "urg",
    "ttl_mean", "ttl_std", "init_win", "iat_mean", "iat_std", "retrans", "label",
]
NUMERIC = [c for c in CANON_COLUMNS if c not in ("ts", "src_ip", "dst_ip", "label")]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# normalised raw name -> (canonical name, multiplicative scale to canonical unit)
ALIASES: dict[str, tuple[str, float]] = {}


def _alias(canon: str, scale: float, *names: str) -> None:
    for n in names:
        ALIASES[_key(n)] = (canon, scale)


_alias("ts", 1.0, "Timestamp", "StartTime", "Stime", "ts", "time", "first_seen")
_alias("src_ip", 1.0, "Src IP", "Source IP", "SrcAddr", "saddr", "srcip")
_alias("dst_ip", 1.0, "Dst IP", "Destination IP", "DstAddr", "daddr", "dstip")
_alias("src_port", 1.0, "Src Port", "Source Port", "Sport", "srcport")
_alias("dst_port", 1.0, "Dst Port", "Destination Port", "Dport", "dsport", "dstport")
_alias("proto", 1.0, "Protocol", "Proto")
_alias("duration", 1e-6, "Flow Duration")                  # CICFlowMeter: microseconds
_alias("duration", 1.0, "Dur", "Duration")                 # CTU-13 / UNSW: seconds
_alias("fwd_pkts", 1.0, "Tot Fwd Pkts", "Total Fwd Packets", "Spkts", "SrcPkts")
_alias("bwd_pkts", 1.0, "Tot Bwd Pkts", "Total Backward Packets", "Dpkts", "DstPkts")
_alias("tot_pkts", 1.0, "TotPkts")
_alias("fwd_bytes", 1.0, "TotLen Fwd Pkts", "Total Length of Fwd Packets", "SrcBytes", "sbytes")
_alias("bwd_bytes", 1.0, "TotLen Bwd Pkts", "Total Length of Bwd Packets", "DstBytes", "dbytes")
_alias("tot_bytes", 1.0, "TotBytes")
_alias("state", 1.0, "State")
for _f in ("syn", "ack", "fin", "rst", "psh", "urg"):
    _alias(_f, 1.0, f"{_f} Flag Cnt", f"{_f} Flag Count", f"{_f}_count", _f)
_alias("ttl_mean", 1.0, "TTL Mean", "sttl", "TTL", "ttl_mean")
_alias("ttl_std", 1.0, "TTL Std", "ttl_std")
_alias("init_win", 1.0, "Init Fwd Win Byts", "Init_Win_bytes_forward", "swin", "init_win")
_alias("iat_mean", 1e-6, "Flow IAT Mean")                  # microseconds
_alias("iat_std", 1e-6, "Flow IAT Std")
_alias("iat_mean", 1.0, "iat_mean")
_alias("iat_std", 1.0, "iat_std")
_alias("retrans", 1.0, "Retrans", "Retransmissions", "retrans")
_alias("label", 1.0, "Label", "attack_cat", "label")

PROTO_NAMES = {"tcp": 6, "udp": 17, "icmp": 1, "ipv6-icmp": 58, "igmp": 2, "arp": 0}

_TS_FORMATS = [
    "%d/%m/%Y %H:%M:%S",        # CSE-CIC-IDS2018
    "%d/%m/%Y %I:%M:%S %p",
    "%Y/%m/%d %H:%M:%S.%f",     # CTU-13 binetflow
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M",           # CSE-CIC-IDS2017
    "%m/%d/%Y %H:%M:%S",
]


def column_mapping(columns) -> dict[str, tuple[str, float]]:
    """raw column -> (canonical, scale). First alias wins for duplicate canonicals."""
    out, seen = {}, set()
    for c in columns:
        hit = ALIASES.get(_key(c))
        if hit and hit[0] not in seen:
            out[c] = hit
            seen.add(hit[0])
    return out


def parse_timestamps(s: pd.Series) -> pd.Series:
    """Return float epoch seconds; unparseable -> NaN (row later dropped)."""
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").astype("float64")
        med = np.nanmedian(v.values) if v.notna().any() else 0
        if med > 1e17:
            v = v / 1e9
        elif med > 1e14:
            v = v / 1e6
        elif med > 1e11:
            v = v / 1e3
        return v
    s = s.astype(str).str.strip()
    sample = s[s.str.match(r"^\d")].head(50)
    for fmt in _TS_FORMATS:
        try:
            pd.to_datetime(sample, format=fmt)
        except (ValueError, TypeError):
            continue
        dt = pd.to_datetime(s, format=fmt, errors="coerce")
        break
    else:
        dt = pd.to_datetime(s, errors="coerce", format="mixed")
    out = pd.Series(np.nan, index=s.index)
    ok = dt.notna()
    out[ok] = dt[ok].astype("datetime64[ns]").astype("int64") / 1e9
    return out


def _state_flags(state: pd.Series) -> dict[str, np.ndarray]:
    """CTU-13 Argus 'State' (e.g. 'FSPA_FSPA', 'S_RA') -> TCP flag counts."""
    st = state.astype(str).str.upper()
    letters = {"syn": "S", "fin": "F", "rst": "R", "psh": "P", "ack": "A", "urg": "U"}
    return {k: st.str.count(v).to_numpy(dtype="float32") for k, v in letters.items()}


def normalize_frame(raw: pd.DataFrame, mapping: dict | None = None) -> pd.DataFrame:
    """Normalise any supported raw flow frame to the canonical schema (fault tolerant)."""
    mapping = mapping or column_mapping(raw.columns)
    n = len(raw)
    out: dict[str, object] = {}
    extra: dict[str, pd.Series] = {}
    for rc, (canon, scale) in mapping.items():
        col = raw[rc]
        if canon in ("ts",):
            out["ts"] = parse_timestamps(col)
        elif canon in ("src_ip", "dst_ip", "label", "state"):
            (extra if canon == "state" else out)[canon] = col.astype(str)
        elif canon == "proto":
            if pd.api.types.is_numeric_dtype(col):
                out["proto"] = pd.to_numeric(col, errors="coerce")
            else:
                low = col.astype(str).str.strip().str.lower()
                out["proto"] = pd.to_numeric(low.map(PROTO_NAMES).fillna(pd.to_numeric(low, errors="coerce")), errors="coerce")
        elif canon in ("tot_pkts", "tot_bytes"):
            extra[canon] = pd.to_numeric(col, errors="coerce")
        else:
            # hex ports in CTU-13 (e.g. '0x0303') are coerced to NaN -> 0
            out[canon] = pd.to_numeric(col, errors="coerce") * scale

    if "ts" not in out:
        raise ValueError("No timestamp column found - cannot build a time series. "
                         f"Columns seen: {list(raw.columns)[:20]}")
    df = pd.DataFrame(out, index=raw.index)

    if "tot_pkts" in extra and "fwd_pkts" not in df:
        df["fwd_pkts"] = extra["tot_pkts"]
    if "tot_bytes" in extra:
        if "fwd_bytes" in df and "bwd_bytes" not in df:
            df["bwd_bytes"] = (extra["tot_bytes"] - df["fwd_bytes"]).clip(lower=0)
        elif "fwd_bytes" not in df:
            df["fwd_bytes"] = extra["tot_bytes"]
    if "state" in extra and "syn" not in df:
        for k, v in _state_flags(extra["state"]).items():
            df[k] = v

    for c in CANON_COLUMNS:
        if c not in df:
            if c in ("src_ip", "dst_ip"):
                df[c] = "0.0.0.0"
            elif c == "label":
                df[c] = "Benign"
            elif c in ("ttl_mean", "ttl_std"):
                df[c] = np.nan            # genuinely unavailable (e.g. CICFlowMeter)
            else:
                df[c] = 0.0
    df = df[CANON_COLUMNS]
    num = df[NUMERIC].replace([np.inf, -np.inf], np.nan)
    ttl = num[["ttl_mean", "ttl_std"]]
    num = num.fillna(0.0).clip(lower=0)
    num[["ttl_mean", "ttl_std"]] = ttl.clip(lower=0)
    df[NUMERIC] = num.astype("float32")
    df = df[df["ts"].notna()].copy()
    df["label"] = df["label"].fillna("Benign").astype(str).str.strip()
    cache = {u: map_label(u) for u in df["label"].unique()}
    df["stage"] = df["label"].map(cache).astype("int8")
    return df.reset_index(drop=True)
