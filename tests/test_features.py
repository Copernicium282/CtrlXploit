import numpy as np
import pandas as pd

from sih_v2.features.scaler import RobustScaler
from sih_v2.features.sequences import context_windows, training_sequences
from sih_v2.features.splits import assign_splits
from sih_v2.features.states import add_targets, state_feature_names
from sih_v2.features.windows import BASE_FEATURES, build_windows_from_files, build_windows_from_frame, compute_window_features
from sih_v2.ingest import read_flows, synth
from sih_v2.ingest.pcap_reader import Packet, write_pcap


def _flows(n, t0, **kw):
    base = dict(src_ip="10.0.0.2", dst_ip="1.1.1.1", src_port=1, dst_port=443, proto=6, duration=1.0, fwd_pkts=5,
                bwd_pkts=5, fwd_bytes=500, bwd_bytes=500, syn=1, ack=9, fin=1, rst=0, psh=2, urg=0, ttl_mean=64.0,
                ttl_std=0.0, init_win=64240, iat_mean=0.1, iat_std=0.01, iat_max=0.2, retrans=0,
                frag_pkts=0.0, df_pkts=0.0, payload_mean=200.0, payload_std=10.0, payload_small_pkts=0.0,
                label="Benign", stage=0)
    base.update(kw)
    d = pd.DataFrame({k: [v] * n for k, v in base.items()})
    d["ts"] = t0 + np.linspace(0, 59, n)
    return d


def test_window_features_basic():
    df = pd.concat([_flows(10, 0.0), _flows(10, 60.0, syn=2, ack=0, fin=0, psh=0, fwd_pkts=1, bwd_pkts=1,
                                                  dst_port=22, stage=1, label="PortScan")])
    w = compute_window_features(df, 60).set_index("wid")
    assert list(w.index) == [0, 1]
    assert np.isclose(w.loc[1, "syn_ratio"], 1.0) and np.isclose(w.loc[0, "syn_ratio"], 0.1)
    assert w.loc[1, "small_flow_frac"] == 1.0 and w.loc[1, "sensitive_port_frac"] == 1.0
    assert w.loc[0, "cnt_0"] == 10 and w.loc[1, "cnt_1"] == 10
    assert not w[BASE_FEATURES].isna().any().any()


def test_beacon_score_detects_periodicity():
    per = _flows(10, 0.0)
    per["ts"] = np.arange(10) * 6.0
    rnd = _flows(10, 60.0)
    rnd["ts"] = 60 + np.sort(np.random.default_rng(1).uniform(0, 59, 10))
    w = compute_window_features(pd.concat([per, rnd]), 60).set_index("wid")
    assert w.loc[0, "beacon_score"] > 0.9 and w.loc[1, "beacon_score"] < w.loc[0, "beacon_score"]


def _sweep_capture(ports, t0, src="10.0.3.7", dst="10.0.4.9"):
    """One source host probing one destination port per packet, in the given order."""
    return [Packet(t0 + i * 0.5, src, dst, 40000 + i, int(p), 6, 64, 40, 0x02, 29200, i + 1, 0)
            for i, p in enumerate(ports)]


def test_scan_shape_separates_sequential_from_randomised_sweep(tmp_path):
    """Crafted capture: window 0 walks ports in order, window 1 jumps across the space.

    Runs the production path end to end - capture -> canonical per-packet flows -> windowed
    features - so the scan-shape terms are proven on the same code the dashboard and the
    CLIs use, not on a hand-built frame.
    """
    rng = np.random.default_rng(0)
    pk = (_sweep_capture(np.arange(1000, 1040), 0.0)                       # 1000, 1001, 1002, ...
          + _sweep_capture(rng.choice(np.arange(1024, 65535), 40, replace=False), 60.0))
    f = tmp_path / "sweep.pcap"
    write_pcap(f, pk)
    w = compute_window_features(read_flows(f), 60).set_index("wid")
    assert w.loc[0, "scan_seq_frac"] > 0.9 and w.loc[1, "scan_seq_frac"] < 0.25
    assert w.loc[0, "scan_step_mean"] < w.loc[1, "scan_step_mean"]
    assert w.loc[0, "n_dst_ports"] > 0 and w.loc[1, "n_dst_ports"] > 0     # same breadth, different shape
    assert w.loc[0, "frag_missing"] == 0 and w.loc[0, "ttl_mean"] > 0     # a capture does carry packet detail


def test_gap_filling_and_segments():
    df = pd.concat([_flows(5, 0.0), _flows(5, 180.0), _flows(5, 3600.0)])
    w = build_windows_from_frame(df, 60, segment_gap=10)
    assert w["segment"].nunique() == 2
    seg0 = w[w["segment"] == w["segment"].iloc[0]]
    assert list(seg0["wid"]) == [0, 1, 2, 3] and seg0["n_flows_raw"].tolist() == [5, 0, 0, 5]


def test_targets_and_phases():
    w = pd.DataFrame({"segment": 0, "stage": [0, 0, 1, 1, 0, 2, 4, 4, 0, 0]})
    t = add_targets(w, horizon=3)
    assert t["y_future"].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    assert t["phase"].tolist()[:6] == ["benign", "benign", "pre_attack", "pre_attack", "pre_attack", "during_attack"]
    assert t["time_to_attack"].tolist()[2:5] == [3, 2, 1]


def test_out_of_order_file_matches_in_memory(tmp_path, cfg):
    rng = np.random.default_rng(3)
    ep = synth.episode(rng, dict(cfg["synthetic"], windows_per_episode=30), 1.5e9)
    ep = ep.sample(frac=1.0, random_state=0)          # completely shuffled rows
    p = tmp_path / "shuffled.csv"
    synth.to_cic_csv(ep, p)
    w_stream, stats = build_windows_from_files([p], 60, chunksize=100)   # many small chunks + buckets
    w_mem = build_windows_from_frame(read_flows(p), 60)
    assert stats["chunks"] > 5
    pd.testing.assert_frame_equal(w_stream[BASE_FEATURES].reset_index(drop=True),
                                  w_mem[BASE_FEATURES].reset_index(drop=True), rtol=1e-4)


def test_states_splits_sequences(small_states, cfg):
    s = small_states
    names = state_feature_names()
    assert len(names) == 3 * len(BASE_FEATURES) and set(names) <= set(s.columns)
    sp = assign_splits(s, cfg["split"], purge=5)
    assert set(sp.unique()) <= {"train", "val", "test", "purged"}
    for seg, g in s.assign(split=sp).groupby("segment"):
        assert g["split"].nunique() >= 1
    sc = RobustScaler().fit(s[names].to_numpy(np.float32))
    X = sc.transform(s[names].to_numpy(np.float32))
    assert np.abs(X).max() <= 10 and np.isfinite(X).all()
    sc2 = RobustScaler.from_dict(sc.to_dict())
    assert np.allclose(sc2.transform(s[names].to_numpy(np.float32)), X)
    seg = s["segment"].to_numpy()
    obs, st, y, m = training_sequences(X, s["stage"].to_numpy(), s["y_future"].to_numpy(), seg, 8, 4, stride=3)
    assert obs.shape[1:] == (12, X.shape[1]) and st.shape == y.shape == m.shape
    assert m[0, :7].sum() == 0 and m[0, 7] == 1       # left padding masked for the first window
    ctx = context_windows(X, seg, 8)
    assert ctx.shape == (len(X), 8, X.shape[1])
    assert np.allclose(ctx[10, -1], X[10])
