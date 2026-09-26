"""Cross-dataset comparison + the public SIH-26153 repos' own reported numbers.

    python eval/compare_baselines.py

Reads every reports/<dataset>/<protocol>/metrics.json produced by `python -m sih_v2 evaluate`
and writes eval/results/comparison.{md,csv}.

Two kinds of comparison are kept strictly apart:
  A. MEASURED - every method re-run by us on identical data, features, splits and thresholds.
     Each public repo's *approach* is represented by a re-implementation (see models/baselines.py).
  B. REPORTED - numbers the repos publish in their READMEs, on their own data, cells, splits and
     metrics. They are NOT comparable to each other or to A, and are listed only for context.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sih_v2.engine.evaluation import summary_table, to_markdown  # noqa: E402

APPROACH = {
    "Persistence (oracle current label)": "Rijja-explore (its strongest validated forecaster)",
    "Logistic Regression (single window)": "PS-mandated baseline (all repos)",
    "Logistic Regression (stacked 8)": "HowSuyash/AttackForecast (stacked LR)",
    "Gradient Boosting (stacked 8)": "Rijja-explore (lagged XGBoost)",
    "Random Forest (stacked 8)": "ayushshandilya-dev/netsight (RF forecaster)",
    "LSTM classifier (no world model)": "PratikBorle / csxzor-devcs / ShadowCat (LSTM family)",
    "NetWorldModel - direct head": "ours (component)",
    "NetWorldModel - K-step rollout": "ours (component) · HowSuyash-style imagination",
    "Markov kill-chain prior": "ours (component)",
    "NetWorldModel - fused (ours)": "ours",
}

REPORTED = [
    ("HowSuyash/AttackForecast", "CTU-13, 13 scenarios, per-host 60 s cells, temporal 70/15/15",
     "RSSM world model F1 0.979, AP 0.995; stacked LR F1 0.977; single LR 0.963. Family holdout (5,8,13): "
     "WM F1 0.874 / AP 0.917 vs single LR F1 0.901. Stage forecast beats persistence at 9/10 horizons. "
     "Host triage 28/30 infected caught, 74/1500 false alarms."),
    ("PratikBorle/miniproject", "CSE-CIC-IDS2018 Thursday-01-03 only, 10 s windows, 65/35 chronological",
     "LSTM F1@5%FPR 0.664 vs LR 0.230; PR-AUC 0.683 vs 0.665; ROC-AUC 0.802 vs 0.813."),
    ("csxzor-devcs/26153", "bundled synthetic generator only",
     "Real-data benchmark not run; its results file states synthetic results must not be cited and its test split "
     "had 0 pre-attack sequences."),
    ("Rijja-explore/Network-Attack-Forecasting", "CTU-13-derived flow windows + CTU PCAP family states",
     "XGBoost current-risk F1 0.941 (val); persistence forecasting F1 0.975 beats lagged XGBoost 0.919; "
     "explicitly claims no lead time and no world model."),
    ("ayushshandilya-dev/netsight", "CIC-IDS2017, all 8 days, 500-flow windows, cross-day holdout",
     "RF forecaster ROC-AUC 0.763, F1 0.375 @0.5; walk-forward ROC-AUC 0.48-0.85; LR cross-day ROC-AUC 0.539."),
    ("muthukkumaranb/ShadowCat", "CSE-CIC-IDS2018, 1-min windows, LOEO / episode-grouped",
     "Pre-onset forecasting F1 0.0 in LOEO; hour+day-only features beat traffic features on chronological holdout "
     "(F1 0.689 vs 0.044) - schedule leakage in CIC-IDS2018."),
    ("GMinnu/SIH26153, BuzyU/NetForecast", "-", "not publicly accessible (HTTP 404)"),
]


def main():
    out = ROOT / "eval/results"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Comparison", "", "## A. Measured - every approach re-run on identical data and protocol", ""]
    rows = []
    for mp in sorted((ROOT / "reports").glob("*/*/metrics.json")):
        rep = json.loads(mp.read_text(encoding="utf-8"))
        tab = summary_table(rep["summary"])
        tab.insert(1, "Approach represented", tab["Method"].map(APPROACH).fillna(""))
        keep = ["Method", "Approach represented", "Seeds", "F1", "F1 ±", "F1 95% CI", "FPR", "PR-AUC", "PR-AUC ±",
                "EW PR-AUC", "Pre-attack recall", "Mean lead (min)"]
        md += [f"### {rep['dataset']} · {rep['protocol']} ({rep['test_cells']:,} test cells, seeds {rep['seeds']})", "",
               to_markdown(tab[keep]), ""]
        sh = pd.DataFrame(rep.get("stage_horizon", []))
        if len(sh):
            piv = sh.pivot(index="method", columns="k", values="macro_f1")
            piv.columns = [f"+{k}" for k in piv.columns]
            md += ["Stage forecasting, macro-F1 per horizon (malicious host-slots):", "", to_markdown(piv.reset_index()), ""]
        t = rep.get("triage", {})
        if t:
            md += [f"Host triage: {t['caught']}/{t['infected']} infected host-slots caught "
                   f"({t['caught_only_by_anomaly']} only by the label-free channel); {t['false_alarms']}/{t['benign']} "
                   "benign host-slots flagged.", ""]
        bad = [c for c in rep.get("checks", []) if not c["verdict"].startswith(("OK", "N/A", "reported"))]
        md += [f"Integrity checks: {len(rep.get('checks', [])) - len(bad)} passed"
               + (", flagged: " + "; ".join(f"{c['check']} - {c['verdict']}" for c in bad) if bad else "") + "", ""]
        tab.insert(0, "dataset", rep["dataset"])
        tab.insert(1, "protocol", rep["protocol"])
        rows.append(tab)
    md += ["## B. Reported by the public repos (their own data/protocols - NOT comparable)", "",
           "| Repo | Their setting | Their reported result |", "|---|---|---|"]
    md += [f"| {r} | {s} | {v} |" for r, s, v in REPORTED]
    md += ["", "The closest like-for-like reference is HowSuyash/AttackForecast (same dataset, per-host 60 s cells, the "
           "same held-out family scenarios 5/8/13). Its cell construction, label mapping and positives differ from ours "
           "(e.g. it maps SPAM to Exfiltration, we map it to Impact/T1496), so even there numbers are indicative only."]
    (out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if rows:
        pd.concat(rows).to_csv(out / "comparison.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
