"""Benchmark: accuracy summary (from evaluate) + ingestion throughput + inference latency.

  python -m sih_v2 benchmark --dataset ctu13 --protocol temporal
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from ..config import resolve
from ..engine.bundle import load_bundle
from ..engine.evaluation import summary_table, to_markdown
from ..engine.simulate import ForecastEngine
from ..ingest.csv_reader import read_flows
from .common import base_parser, dump_json, setup
from .train import load_split

log = logging.getLogger("benchmark")


def main(argv=None):
    p = base_parser("Benchmark accuracy, throughput and latency")
    p.add_argument("--skip-eval", action="store_true", help="reuse metrics.json")
    args = p.parse_args(argv)
    cfg = setup(args)
    rep = resolve(cfg["paths"]["reports_dir"])
    fwd = [a for a in ("--config", args.config, "--dataset", cfg["dataset"], "--protocol", cfg["protocol"]) if a]
    if not args.skip_eval or not (rep / "metrics.json").exists():
        from .evaluate import main as evaluate_main
        evaluate_main(fwd)
    rows = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))["summary"]
    bundle = load_bundle(resolve(cfg["paths"]["model_bundle"]))
    perf = {"parameters": sum(p.numel() for p in bundle.model.parameters())}
    sample = resolve(cfg["paths"]["sample_dir"])
    for name in ("sample_cic2018.csv", "sample_ctu13.binetflow", "sample_capture.pcap"):
        f = sample / name
        if f.exists():
            t = time.perf_counter()
            n = len(read_flows(f))
            dt = time.perf_counter() - t
            perf[f"ingest_{name}"] = {"flows": n, "sec": round(dt, 3), "flows_per_sec": round(n / dt)}
    test = load_split(cfg)["test"]
    part = test[test["segment"].isin(test["segment"].drop_duplicates().head(200))]
    eng = ForecastEngine(bundle)
    t = time.perf_counter()
    eng.run(part)
    dt = time.perf_counter() - t
    perf["batch_inference"] = {"cells": len(part), "particles": eng.P, "rollout_steps": eng.K,
                               "sec": round(dt, 3), "ms_per_cell": round(1000 * dt / len(part), 3)}
    one = part[part["segment"] == part["segment"].iloc[0]].tail(1)
    t = time.perf_counter()
    for _ in range(20):
        eng.run(one)
    perf["streaming_latency_ms"] = round(1000 * (time.perf_counter() - t) / 20, 2)
    tab = summary_table(rows)[["Method", "F1", "FPR", "PR-AUC", "EW PR-AUC", "Mean lead (min)"]]
    dump_json({"performance": perf, "accuracy": tab.to_dict("records")}, rep / "benchmark.json")
    (rep / "benchmark.md").write_text("\n".join([f"# Benchmark · {cfg['dataset']} · {cfg['protocol']}", "",
                                                 to_markdown(tab), "", "## Performance (CPU)", "", "```json",
                                                 json.dumps(perf, indent=2), "```"]), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string(index=False))
    print(json.dumps(perf, indent=2))


if __name__ == "__main__":
    main()
