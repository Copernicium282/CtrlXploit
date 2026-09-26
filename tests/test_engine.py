import numpy as np
import pandas as pd
import pytest

from sih_v2.engine.alerts import generate_alerts
from sih_v2.engine.bundle import Bundle, load_bundle, save_bundle
from sih_v2.engine.calibrate import choose_threshold, tune_fusion
from sih_v2.engine.explain import gradient_shap
from sih_v2.engine.metrics import binary_metrics, lead_times
from sih_v2.engine.simulate import ForecastEngine
from sih_v2.features.scaler import RobustScaler
from sih_v2.features.states import state_feature_names
from sih_v2.models.markov import KillChainMarkov
from sih_v2.models.trainer import build_model


@pytest.fixture(scope="module")
def tiny_bundle(small_states, cfg, tmp_path_factory):
    feats = state_feature_names()
    sc = RobustScaler().fit(small_states[feats].to_numpy(np.float32))
    mk = KillChainMarkov().fit(small_states["stage"].to_numpy(), small_states["segment"].to_numpy())
    b = Bundle(build_model(len(feats), cfg).eval(), sc, mk, feats, {"fused": 0.5, "direct": 0.5, "rollout": 0.5,
                                                                    "markov": 0.5}, cfg)
    p = tmp_path_factory.mktemp("b") / "wm.pt"
    save_bundle(p, b)
    return load_bundle(p)


def test_engine_run_outputs(tiny_bundle, small_states):
    eng = ForecastEngine(tiny_bundle, particles=4, steps=5)
    r = eng.run(small_states)
    f = r.frame
    assert len(f) == len(small_states)
    for c in ("risk", "risk_direct", "risk_rollout", "risk_markov"):
        assert f[c].between(0, 1).all()
    assert r.rollout_mean.shape == (len(f), 5, 7) and np.allclose(r.rollout_mean.sum(-1), 1, atol=1e-4)
    assert r.exploit_q.shape == (len(f), 5, 3) and (np.diff(r.exploit_q, axis=-1) >= -1e-6).all()
    assert r.attention.shape == (len(f), tiny_bundle.config["forecast"]["context"])
    r2 = ForecastEngine(tiny_bundle, particles=4, steps=5).run(small_states)
    assert np.allclose(r.frame["risk"], r2.frame["risk"])          # seeded -> reproducible


def test_bundle_roundtrip_identical(tiny_bundle, small_states, tmp_path):
    p = tmp_path / "again.pt"
    save_bundle(p, tiny_bundle)
    b2 = load_bundle(p)
    a = ForecastEngine(tiny_bundle, particles=2, steps=3).run(small_states.head(50)).frame["risk_direct"]
    b = ForecastEngine(b2, particles=2, steps=3).run(small_states.head(50)).frame["risk_direct"]
    assert np.allclose(a, b)


def test_gradient_shap(tiny_bundle, small_states):
    eng = ForecastEngine(tiny_bundle, particles=2, steps=3)
    r = eng.run(small_states.head(60))
    from sih_v2.features.sequences import context_windows
    ctx = context_windows(r.X, small_states.head(60)["segment"].to_numpy(), eng.L)
    e = gradient_shap(tiny_bundle, ctx[-1], ctx[:30], n_samples=16, K=3)
    # one row per instant feature, labelled with the bundle's own names
    assert len(e["features"]) == len(tiny_bundle.feature_names) // 3
    assert set(e["features"]["feature"]) <= set(tiny_bundle.feature_names)
    assert np.isfinite(e["features"]["attribution"]).all()
    assert e["per_time"].shape == (eng.L,) and 0 <= e["risk"] <= 1


def test_gradient_shap_explains_bundles_from_an_earlier_feature_set(small_states):
    """The dashboard explains the shipped bundles, which were trained before the
    packet-level features existed - attribution must follow the bundle's own width."""
    from sih_v2.config import resolve
    from sih_v2.engine.bundle import load_bundle
    from sih_v2.features.sequences import context_windows

    b = load_bundle(resolve("models/synthetic/temporal/wm.pt"))
    x = b.scaler.transform(small_states[b.feature_names].to_numpy(np.float32))
    ctx = context_windows(x, small_states["segment"].to_numpy(), b.config["forecast"]["context"])
    e = gradient_shap(b, ctx[-1], ctx[:30], n_samples=8, K=2)
    nb = len(b.feature_names) // 3
    assert len(e["features"]) == nb                                        # the bundle's width, not today's
    assert set(e["features"]["feature"]) == set(b.feature_names[:nb])     # rows are ranked, so compare as a set
    assert np.isfinite(e["features"]["attribution"]).all()


def test_threshold_respects_fpr_budget():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.2
    s = np.where(y, rng.normal(0.7, 0.15, 2000), rng.normal(0.3, 0.15, 2000))
    t = choose_threshold(s, y, max_fpr=0.02)
    assert binary_metrics(y, s >= t)["fpr"] <= 0.02 + 1e-9


def test_tune_fusion_prefers_informative_component():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.3).astype(int)
    fr = pd.DataFrame({"exploit_now": 0, "y_future": y, "risk_direct": y * 0.6 + rng.random(1000) * 0.4,
                       "risk_rollout": rng.random(1000), "risk_markov": rng.random(1000)})
    w, ap = tune_fusion(fr)
    assert w["direct"] >= 0.8 and ap > 0.9


def test_lead_time_and_alerts():
    fr = pd.DataFrame({"segment": 0, "exploit_now": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]})
    pred = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    lt = lead_times(fr, pred, horizon=4, window_seconds=60)
    assert lt["incidents"] == 2 and [d["lead_windows"] for d in lt["detail"]] == [3, 0]
    assert lt["mean_lead_min"] == 1.5 and lt["warned_before_onset"] == 0.5
    frame = pd.DataFrame({"segment": 0, "risk": [0.1, 0.8, 0.9, 0.2, 0.95], "forecast_stage": "InitialAccess",
                          "eta_windows": 2.0, "time": pd.date_range("2024-01-01", periods=5, freq="min"),
                          "current_stage_pred": "Reconnaissance", "phase": "pre_attack"})
    al = generate_alerts(frame, 0.5, min_consecutive=1)
    assert len(al) == 2 and al.iloc[0]["duration_min"] == 2 and al.iloc[1]["severity"] == "CRITICAL"
    assert "TA0001" in al.iloc[0]["mitre_tactic"]
    assert len(generate_alerts(frame, 0.5, min_consecutive=2)) == 1
