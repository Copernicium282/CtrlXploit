"""Evaluate every method on the held-out split of one dataset/protocol.

  python -m sih_v2 evaluate --dataset ctu13 --protocol temporal
Writes reports/<dataset>/<protocol>/{metrics.json, evaluation.md, test_predictions.parquet,
test_arrays.npz, host_triage.csv, stage_horizon.csv}.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from ..config import resolve
from ..engine.bundle import load_bundle
from ..engine.evaluation import OURS, evaluate_protocol, summary_table, to_markdown
from .common import base_parser, dump_json, setup
from .train import load_split

log = logging.getLogger("evaluate")


def render_md(cfg, res, n_test) -> str:
    tab = summary_table(res["summary"])
    st = pd.DataFrame(res["stage_horizon"])
    tri, sur = res["triage"], res["surprise"]
    ds, pr = cfg["dataset"], cfg["protocol"]
    md = [f"# Evaluation · {ds} · {pr} protocol", "",
          f"Test cells: {n_test:,}. Target: exploitation-stage cell within the next {cfg['forecast']['horizon']} "
          f"windows. Thresholds frozen on validation (F1-optimal s.t. FPR ≤ {cfg['engine']['max_fpr']}). "
          "Deep models: mean over seeds (± = std); 95 % CI = bootstrap over host-slots/segments for the primary seed.",
          "", "## Binary forecasting", "", to_markdown(tab), ""]
    if len(st):
        piv = st.pivot(index="method", columns="k", values="macro_f1")
        piv.columns = [f"+{k}" for k in piv.columns]
        tr = st.pivot(index="method", columns="k", values="transition_acc")
        tr.columns = [f"+{k}" for k in tr.columns]
        md += ["## Kill-chain stage forecasting (macro-F1 over stages present, malicious host-slots)", "",
               to_markdown(piv.reset_index()), "", "Accuracy on cells whose stage *changes* by t+k "
               "(persistence scores 0 here by construction):", "", to_markdown(tr.reset_index()), ""]
    md += ["## Host-level triage (two channels)", "",
           f"- infected host-slots caught: **{tri['caught']}/{tri['infected']}** "
           f"(risk channel {tri['caught_by_risk']}, anomaly channel only {tri['caught_only_by_anomaly']})",
           f"- false alarms: {tri['false_alarms']}/{tri['benign']} benign host-slots "
           f"({tri['false_alarm_rate']:.1%}); risk channel alone {tri['risk_only_false_alarms']}",
           "", "## Label-free surprise channel", "",
           f"- direction chosen on validation: sign {sur['sign']:+.0f} (validation raw AUC {sur.get('val_auc_raw', float('nan')):.3f})",
           f"- test window-level AUC (malicious vs benign cells): {sur.get('window_auc', float('nan')):.3f} "
           f"(KL variant {sur.get('window_auc_kl', float('nan')):.3f})",
           f"- test host-level AUC: surprise {sur.get('host_auc', float('nan')):.3f} vs supervised risk "
           f"{sur.get('host_auc_risk', float('nan')):.3f}", "", "## Integrity checks", ""]
    for c in res["checks"]:
        extra = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k not in ("check", "verdict")}
        md.append(f"- **{c['check']}** - {c['verdict']}  `{extra}`")
    return "\n".join(md) + "\n"


def main(argv=None):
    p = base_parser("Evaluate forecasting vs baselines")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    args = p.parse_args(argv)
    cfg = setup(args)
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [cfg["seed"]])
    bundle = load_bundle(resolve(cfg["paths"]["model_bundle"]))
    pack = joblib.load(resolve(cfg["paths"]["baseline_bundle"]))
    sp = load_split(cfg)
    test = sp["test"]
    log.info("evaluating %s/%s on %d cells (%s)", cfg["dataset"], cfg["protocol"], len(test),
             test["phase"].value_counts().to_dict())
    res, primary = evaluate_protocol(cfg, test, sp["train"], bundle, pack, seeds)
    out = resolve(cfg["paths"]["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res["triage_table"].to_csv(out / "host_triage.csv", index=False)
    pd.DataFrame(res["stage_horizon"]).to_csv(out / "stage_horizon.csv", index=False)
    dump_json({k: v for k, v in res.items() if k != "triage_table"} |
              {"dataset": cfg["dataset"], "protocol": cfg["protocol"], "test_cells": len(test), "seeds": seeds,
               "config": {"horizon": cfg["forecast"]["horizon"], "context": cfg["forecast"]["context"],
                          "window_seconds": cfg["features"]["window_seconds"], "max_fpr": cfg["engine"]["max_fpr"]}},
              out / "metrics.json")
    (out / "evaluation.md").write_text(render_md(cfg, res, len(test)), encoding="utf-8")
    primary.frame.to_parquet(out / "test_predictions.parquet", index=False)
    np.savez_compressed(out / "test_arrays.npz", rollout_mean=primary.rollout_mean.astype(np.float16),
                        exploit_q=primary.exploit_q.astype(np.float16), markov_path=primary.markov_path.astype(np.float16),
                        attention=primary.attention.astype(np.float16), stage_now=primary.stage_now.astype(np.float16))
    pd.set_option("display.width", 250)
    t = summary_table(res["summary"])[["Method", "F1", "F1 ±", "FPR", "PR-AUC", "EW PR-AUC", "Mean lead (min)"]]
    print(t.round(3).to_string(index=False))
    print((out / "evaluation.md").read_text(encoding="utf-8").split("## Kill-chain")[1][:2500] if "## Kill-chain" in
          (out / "evaluation.md").read_text(encoding="utf-8") else "")
    log.info("ours = %s; wrote %s", OURS, out)
    return res


if __name__ == "__main__":
    main()
