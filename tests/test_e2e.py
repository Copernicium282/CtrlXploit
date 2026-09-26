"""End-to-end: full CLI pipeline in an isolated temp project + pretrained bundles on real / held-out data."""
import copy
import json

import numpy as np
import pytest
import yaml

from sih_v2.config import resolve

pytestmark = pytest.mark.filterwarnings("ignore")


def test_cli_pipeline_end_to_end(tmp_path, cfg):
    from sih_v2.cli import benchmark, build_dataset, evaluate, train

    raw = yaml.safe_load(open(resolve("config.yaml")))
    c = copy.deepcopy(raw)
    c["dataset"] = "synthetic"
    c["seeds"] = [0]
    c["synthetic"].update(episodes=12, windows_per_episode=90, campaign_prob=1.0)
    c["train"].update(epochs=1, batch_size=128, max_seq_per_epoch=2000)
    c["engine"]["particles"] = 4
    c["baselines"]["max_train_rows"] = 3000
    c["paths"] = {k: (str(tmp_path / v) if isinstance(v, str) else v) for k, v in raw["paths"].items()}
    c["paths"]["sample_dir"] = str(resolve("data/sample"))
    cp = tmp_path / "config.yaml"
    cp.write_text(yaml.safe_dump(c))
    build_dataset.main(["--config", str(cp), "--synthetic", "--no-samples"])
    train.main(["--config", str(cp)])
    res = evaluate.main(["--config", str(cp)])
    benchmark.main(["--config", str(cp), "--skip-eval"])
    names = [r["method"] for r in res["summary"]]
    for must in ("Persistence", "Logistic Regression (single window)", "Logistic Regression (stacked 8)",
                 "Gradient Boosting", "Random Forest", "LSTM classifier", "fused (ours)"):
        assert any(must in n for n in names), must
    rep = tmp_path / "reports/synthetic/temporal"
    for f in ("train_summary.json", "metrics.json", "evaluation.md", "test_predictions.parquet", "test_arrays.npz",
              "host_triage.csv", "stage_horizon.csv", "benchmark.json"):
        assert (rep / f).exists(), f
    m = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
    assert {c["check"].split(" (")[0] for c in m["checks"]} >= {"split integrity", "schedule-only probe",
                                                             "no-peeking gradient test"}
    assert next(c for c in m["checks"] if c["check"].startswith("no-peeking"))["verdict"].startswith("OK")


@pytest.mark.skipif(not resolve("models/ctu13/temporal/wm.pt").exists(), reason="CTU-13 bundle not trained yet")
def test_pretrained_ctu13_bundle_on_real_test_split():
    import pandas as pd

    from sih_v2.engine.bundle import load_bundle
    from sih_v2.engine.simulate import ForecastEngine

    b = load_bundle(resolve("models/ctu13/temporal/wm.pt"))
    df = pd.read_parquet(resolve("data/processed/ctu13/cells.parquet"))
    t = df[df["split_temporal"] == "test"]
    mal = t[t.groupby("segment")["stage"].transform("max") > 0]
    part = pd.concat([mal, t[~t["segment"].isin(mal["segment"])].head(3000)])
    f = ForecastEngine(b, particles=8).run(part).frame
    inf = f[f["stage"] > 0]
    assert inf["risk"].mean() > f[f["stage"] == 0]["risk"].mean()
    assert np.isfinite(f["surprise_obs"]).all()


@pytest.mark.skipif(not resolve("models/synthetic/temporal/wm.pt").exists(), reason="synthetic bundle missing")
def test_pretrained_bundle_on_uploaded_samples():
    from sih_v2.engine.bundle import load_bundle
    from sih_v2.engine.pipeline import forecast_files

    b = load_bundle(resolve("models/synthetic/temporal/wm.pt"))
    res, states, stats = forecast_files([resolve("data/sample/sample_cic2018.csv")], b)
    f = res.frame
    assert stats["segments"] == 2
    pre = f[f["phase"] == "pre_attack"]
    assert len(pre) > 0 and pre["alert"].mean() > 0.3
    for name in ("sample_ctu13.binetflow", "sample_capture.pcap"):
        r, _, _ = forecast_files([resolve(f"data/sample/{name}")], b, particles=4)
        assert np.isfinite(r.frame["risk"]).all() and len(r.frame) > 10
