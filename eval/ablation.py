"""Ablations on a held-out split (no retraining):
  * fusion components (direct / rollout / Markov / combinations) - thresholds re-calibrated on validation
  * imagination branch: number of Monte-Carlo particles x rollout horizon K

    python eval/ablation.py --dataset ctu13 --protocol temporal
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sih_v2.cli.train import load_split  # noqa: E402
from sih_v2.config import load_config, resolve  # noqa: E402
from sih_v2.engine.bundle import load_bundle  # noqa: E402
from sih_v2.engine.calibrate import choose_threshold  # noqa: E402
from sih_v2.engine.evaluation import to_markdown  # noqa: E402
from sih_v2.engine.metrics import evaluate_method  # noqa: E402
from sih_v2.engine.simulate import ForecastEngine  # noqa: E402


def row(name, fr, score, thr, H, W, extra=None):
    r = evaluate_method(name, fr, score, thr, H, W)
    return {"variant": name, **(extra or {}), "F1": r["f1"], "FPR": r["fpr"], "PR-AUC": r["pr_auc"],
            "EW PR-AUC": r["ew_pr_auc"], "Mean lead (min)": r["lead_time"]["mean_lead_min"]}


def focus(df: pd.DataFrame, n_benign: int = 150) -> pd.DataFrame:
    """Every host-slot with malicious traffic + a fixed sample of benign ones (keeps the run minutes-long)."""
    mal = df.groupby("segment")["stage"].transform("max") > 0
    ben = df.loc[~mal, "segment"].drop_duplicates()
    ben = ben.sample(min(n_benign, len(ben)), random_state=0)
    return df[mal | df["segment"].isin(ben)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--protocol", default=None)
    a = ap.parse_args()
    cfg = load_config(dataset=a.dataset, protocol=a.protocol)
    b = load_bundle(resolve(cfg["paths"]["model_bundle"]))
    sp = load_split(cfg)
    va, te = focus(sp["val"]), focus(sp["test"])
    H, W, mf = cfg["forecast"]["horizon"], cfg["features"]["window_seconds"], cfg["engine"]["max_fpr"]
    rv, rt = ForecastEngine(b, particles=8).run(va).frame, ForecastEngine(b, particles=8).run(te).frame
    rows = []
    combos = {"direct": (1, 0, 0), "rollout": (0, 1, 0), "markov": (0, 0, 1), "direct+rollout": (.5, .5, 0),
              "direct+markov": (.7, 0, .3), "all (0.5/0.35/0.15)": (.5, .35, .15)}
    for name, (d, r, m) in combos.items():
        sv = d * rv["risk_direct"] + r * rv["risk_rollout"] + m * rv["risk_markov"]
        st = d * rt["risk_direct"] + r * rt["risk_rollout"] + m * rt["risk_markov"]
        rows.append(row(f"fusion: {name}", rt, st, choose_threshold(sv, va["y_future"], mf), H, W))
    fusion = pd.DataFrame(rows)
    rows = []
    for P in (1, 8, 32):
        for K in (5, 10, 15):
            t = time.perf_counter()
            fv = ForecastEngine(b, particles=P, steps=K).run(va).frame
            ft = ForecastEngine(b, particles=P, steps=K).run(te).frame
            thr = choose_threshold(fv["risk_rollout"], va["y_future"], mf)
            rows.append(row("rollout", ft, ft["risk_rollout"], thr, H, W,
                            {"particles": P, "K": K, "sec": round(time.perf_counter() - t, 1)}))
    roll = pd.DataFrame(rows)
    out = ROOT / "eval/results"
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Ablations · {cfg['dataset']} · {cfg['protocol']}", "",
          f"Evaluated on every malicious host-slot plus 150 sampled benign host-slots ({len(te):,} test cells); "
          "thresholds re-calibrated on the same kind of validation subset.", "",
          "## Fusion components", "", to_markdown(fusion), "",
          "## Imagination branch: particles x rollout horizon", "", to_markdown(roll), ""]
    (out / f"ablation_{cfg['dataset']}_{cfg['protocol']}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
