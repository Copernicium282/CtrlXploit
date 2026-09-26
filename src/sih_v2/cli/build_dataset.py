"""Build the state-cell dataset for one dataset.

  python -m sih_v2 build_dataset --dataset ctu13        # real CTU-13 (streams 1.9 GB, keeps only cells)
  python -m sih_v2 build_dataset --dataset cic2018      # CSE-CIC-IDS2018 day files, one at a time
  python -m sih_v2 build_dataset --dataset synthetic    # controlled generated campaigns
  python -m sih_v2 build_dataset --dataset ctu13 --inputs /data/CTU-13/   # already-downloaded files

Raw files are converted to compact per-capture cell parquet files and (unless
keep_raw) deleted immediately, so the footprint stays within paths.disk_budget_gb.
"""
from __future__ import annotations

import copy
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, resolve
from ..engine.pipeline import windows_to_states
from ..features.splits import assign_splits
from ..features.windows import build_windows_from_files
from ..ingest import synth
from ..ingest.csv_reader import list_inputs
from .common import base_parser, dump_json, setup

log = logging.getLogger("build_dataset")


def disk_usage_gb(*dirs) -> float:
    total = 0
    for d in dirs:
        d = resolve(d)
        if d.exists():
            total += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return total / 1e9


def check_budget(cfg):
    used = disk_usage_gb("data", "models", "reports")
    budget = cfg["paths"].get("disk_budget_gb", 7.0)
    log.info("project data footprint %.2f GB (budget %.1f GB)", used, budget)
    if used > budget:
        raise RuntimeError(f"disk budget exceeded: {used:.2f} GB > {budget} GB")
    return used


def build_cells(files, cfg, capture: str) -> pd.DataFrame:
    d, ing = cfg["data"], cfg["ingest"]
    w, stats = build_windows_from_files(
        files, cfg["features"]["window_seconds"], ing["chunksize"], ing["segment_gap_windows"],
        by_host=d["mode"] == "host", host_prefixes=d.get("host_prefixes"),
        internal_prefixes=tuple(d.get("internal_prefixes", ("10.", "192.168.", "172.16."))),
        max_benign_hosts=d.get("max_benign_hosts"), min_active_windows=d.get("min_active_windows", 1),
        stage_rule=d.get("stage_rule", "dominant"), capture_names={str(f): capture for f in files})
    log.info("capture %s: %d flows -> %d kept -> %d cells, %d segments%s", capture, stats["flows"],
             stats["flows_kept"], stats["windows"], stats["segments"],
             f", {stats['hosts']} hosts" if "hosts" in stats else "")
    return w


# ----------------------------------------------------------------------------- CTU-13
def ctu_scenario(path: Path) -> str:
    # normalise separators first: on Windows, str(path) uses '\\' and the capture
    # number would never match, so every cell fell back to the file stem and the
    # family split (keyed on scenario numbers) went unused.
    m = re.search(r"CTU-13-Dataset/(\d+)/", str(path).replace("\\", "/"))
    return m.group(1) if m else path.stem


def _shquote(value) -> str:
    """Quote one argument for the platform shell.

    POSIX shells understand single quotes; `cmd.exe` does not treat `'` as a
    quoting character at all.
    """
    s = str(value)
    return f'"{s}"' if os.name == "nt" else shlex.quote(s)


def fetch_ctu13(cfg, raw: Path):
    raw.mkdir(parents=True, exist_ok=True)
    log.info("streaming CTU-13 (1.9 GB) and extracting only the labelled .binetflow files -> %s", raw)
    cmd = (f"curl -sL {_shquote(cfg['data']['url'])} | tar -xjf - -C {_shquote(raw)} "
           f"--include={_shquote('*.binetflow')}")
    subprocess.run(cmd, shell=True, check=True)


def build_ctu13(cfg, inputs) -> list[Path]:
    raw, cells = resolve(cfg["paths"]["raw_dir"]), resolve(cfg["paths"]["cells_dir"])
    cells.mkdir(parents=True, exist_ok=True)
    files = list_inputs(inputs) if inputs else sorted(raw.rglob("*.binetflow"))
    have = {p.stem for p in cells.glob("*.parquet")}
    if not files and len(have) < 13 and not inputs:
        fetch_ctu13(cfg, raw)
        files = sorted(raw.rglob("*.binetflow"))
    for f in sorted(files, key=lambda p: int(ctu_scenario(p)) if ctu_scenario(p).isdigit() else 99):
        sc = ctu_scenario(f)
        out = cells / f"{int(sc):02d}.parquet" if sc.isdigit() else cells / f"{sc}.parquet"
        if out.exists():
            log.info("scenario %s already converted", sc)
        else:
            t = time.time()
            build_cells([f], cfg, sc).to_parquet(out, index=False)
            log.info("scenario %s -> %s (%.1f MB, %.0fs)", sc, out.name, out.stat().st_size / 1e6, time.time() - t)
        if not cfg["data"].get("keep_raw", False) and not inputs:
            f.unlink()
        check_budget(cfg)
    if not cfg["data"].get("keep_raw", False) and raw.exists() and not inputs:
        shutil.rmtree(raw, ignore_errors=True)
    return sorted(cells.glob("*.parquet"))


def download(url: str, dest: Path, attempts: int = 6) -> None:
    """Resumable download that aborts a stalled transfer (<100 kB/s for 60 s) and resumes it."""
    for i in range(attempts):
        r = subprocess.run(["curl", "-sfL", "-C", "-", "--speed-limit", "100000", "--speed-time", "60",
                            "-o", str(dest), url])
        if r.returncode == 0:
            return
        log.warning("download interrupted (curl exit %d), resuming (%d/%d)", r.returncode, i + 1, attempts)
        time.sleep(5)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}")


# ----------------------------------------------------------------------------- CIC-IDS2018
def build_cic2018(cfg, inputs) -> list[Path]:
    raw, cells = resolve(cfg["paths"]["raw_dir"]), resolve(cfg["paths"]["cells_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    cells.mkdir(parents=True, exist_ok=True)
    if inputs:
        for f in list_inputs(inputs):
            day = f.stem.replace("_TrafficForML_CICFlowMeter", "")
            if not (cells / f"{day}.parquet").exists():
                build_cells([f], cfg, day).to_parquet(cells / f"{day}.parquet", index=False)
        return sorted(cells.glob("*.parquet"))
    for day in cfg["data"]["days"]:
        out = cells / f"{day}.parquet"
        if out.exists():
            continue
        f = raw / f"{day}.csv"
        url = cfg["data"]["url"] + f"{day}_TrafficForML_CICFlowMeter.csv"
        log.info("downloading %s", day)
        download(url, f)
        build_cells([f], cfg, day).to_parquet(out, index=False)
        f.unlink()
        check_budget(cfg)
    return sorted(cells.glob("*.parquet"))


# ----------------------------------------------------------------------------- synthetic
def make_samples(cfg: dict) -> list[str]:
    """Held-out demo inputs (different seed - never seen in training)."""
    sdir = resolve(cfg["paths"]["sample_dir"])
    sdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    scfg = copy.deepcopy(cfg["synthetic"])
    scfg.update(campaign_prob=1.0, stealth_prob=0.0, decoy_recon_prob=1.0)
    base = pd.Timestamp("2018-03-01 08:00:00").timestamp()
    ep1 = synth.episode(rng, scfg, base)
    scfg.update(campaign_prob=0.0)
    ep2 = synth.episode(rng, scfg, base + 86400)
    synth.to_cic_csv(pd.concat([ep1, ep2], ignore_index=True), sdir / "sample_cic2018.csv")
    synth.to_ctu_csv(ep1, sdir / "sample_ctu13.binetflow")
    synth.make_sample_pcap(sdir / "sample_capture.pcap")
    return [str(p) for p in sorted(sdir.iterdir())]


def build_synthetic(cfg, inputs, regenerate: bool, episodes=None) -> list[Path]:
    raw, cells = resolve(cfg["paths"]["raw_dir"]), resolve(cfg["paths"]["cells_dir"])
    if regenerate or not list_inputs([raw]):
        for f in raw.glob("synthetic_*.csv") if raw.exists() else []:
            f.unlink()
        synth.generate(raw, cfg["synthetic"], seed=cfg["seed"], episodes=episodes)
        shutil.rmtree(cells, ignore_errors=True)
    cells.mkdir(parents=True, exist_ok=True)
    for f in list_inputs(inputs or [raw]):
        out = cells / f"{f.stem}.parquet"
        if not out.exists():
            build_cells([f], cfg, f.stem).to_parquet(out, index=False)
    return sorted(cells.glob("*.parquet"))


# ----------------------------------------------------------------------------- assemble
def assemble(cfg, cell_files) -> pd.DataFrame:
    parts, off = [], 0
    for f in cell_files:
        w = pd.read_parquet(f)
        w["capture"] = w["capture"].astype(str)
        w["segment"] = w["segment"] + off
        off = int(w["segment"].max()) + 1
        parts.append(w)
    w = pd.concat(parts, ignore_index=True)
    states = windows_to_states(w, cfg)
    purge = cfg["forecast"]["context"] + cfg["forecast"]["horizon"]
    states["split_temporal"] = assign_splits(states, cfg["split"], purge, "temporal").to_numpy()
    if cfg["data"].get("family_split"):
        states["split_family"] = assign_splits(states, cfg["split"], purge, "family",
                                               cfg["data"]["family_split"]).to_numpy()
    return states


def feature_coverage(states: pd.DataFrame) -> dict:
    """How much of each packet-level attribute this conversion actually carried.

    Raw PCAP and the synthetic generator carry all of them; Argus/CTU-13 binetflow has
    no TTL, no IP flags and no per-packet payload sizes, and CICFlowMeter exports have
    neither TTL nor IP flags. Recording the real share keeps the shipped stats honest
    rather than implying the model saw packet detail the source never provided.
    """
    probes = (("ttl", "ttl_missing", "ttl_mean"), ("payload_sizes", "payload_missing", "log_payload_mean"),
              ("ip_fragment_flags", "frag_missing", "frag_frac"), ("iat_max", "iat_max_missing", "log_iat_max"),
              ("port_sweep_shape", None, "scan_seq_frac"))
    out = {}
    for name, flag, feat in probes:
        carried = states[feat].notna() if flag is None else states[feat].notna() & states[flag].eq(0)
        out[name] = {"cells_carried": int(carried.sum()), "share": round(float(carried.mean()), 4),
                     "mean_when_carried": round(float(states.loc[carried, feat].mean()), 4) if carried.any() else None}
    return out


def main(argv=None):
    p = base_parser("Build the state-cell dataset")
    p.add_argument("--inputs", nargs="*", help="already-downloaded files or directories")
    p.add_argument("--synthetic", action="store_true", help="regenerate synthetic telemetry (synthetic dataset)")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--no-samples", action="store_true")
    args = p.parse_args(argv)
    cfg = setup(args)
    t0 = time.time()
    ds = cfg["dataset"]
    if ds == "ctu13":
        cell_files = build_ctu13(cfg, args.inputs)
    elif ds == "cic2018":
        cell_files = build_cic2018(cfg, args.inputs)
    else:
        cell_files = build_synthetic(cfg, args.inputs, args.synthetic, args.episodes)
    states = assemble(cfg, cell_files)
    out = resolve(cfg["paths"]["processed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    states.to_parquet(out, index=False)
    summary = {
        "dataset": ds, "captures": sorted(states["capture"].unique().tolist(), key=lambda s: (len(s), s)),
        "cells": int(len(states)), "segments": int(states["segment"].nunique()),
        "hosts": int(states.groupby(["capture", "host"]).ngroups) if "host" in states else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "stage_counts": states["stage"].value_counts().sort_index().to_dict(),
        "phase_counts": states["phase"].value_counts().to_dict(),
        "positive_rate": float(states["y_future"].mean()),
        "splits": {c: states[c].value_counts().to_dict() for c in states if c.startswith("split_")},
        "feature_coverage": feature_coverage(states),
        "disk_gb": round(check_budget(cfg), 3),
    }
    if not args.no_samples and ds == "synthetic":
        summary["samples"] = make_samples(cfg)
    dump_json(summary, resolve(cfg["paths"]["processed"]).parent / "dataset_stats.json")
    log.info("%s: %d cells, %d segments -> %s (%.1fs)", ds, len(states), states["segment"].nunique(), out,
             time.time() - t0)
    for c in (c for c in states if c.startswith("split_")):
        print(c, "\n", states.groupby([c, "phase"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
