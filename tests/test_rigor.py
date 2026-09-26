"""Tests for the upgrade: per-host cells, split protocols, integrity checks, surprise, triage, stage horizons."""
import numpy as np
import pandas as pd
import torch

from sih_v2.engine import leakage
from sih_v2.engine.metrics import host_triage, stage_horizon_table
from sih_v2.engine.simulate import robust_z
from sih_v2.features.splits import assign_splits
from sih_v2.features.windows import build_windows_from_frame
from sih_v2.ingest.schema import normalize_frame
from sih_v2.models.baselines import future_stage, subsample
from sih_v2.models.trainer import build_model


def _ctu_rows():
    rows = []
    t0 = pd.Timestamp("2011-08-10 10:00:00")
    for m in range(30):
        ts = (t0 + pd.Timedelta(minutes=m)).strftime("%Y/%m/%d %H:%M:%S.%f")
        rows.append([ts, 0.5, "tcp", "147.32.84.165", 1025, "->", "8.8.8.8", 25, "S_RA", 4, 400, 200,
                     "flow=From-Botnet-V42-TCP-Attempt-SPAM" if m >= 10 else "flow=From-Botnet-V42-UDP-DNS"])
        rows.append([ts, 0.5, "tcp", "147.32.84.165", 1026, "->", "147.32.84.20", 445, "S_RA", 4, 400, 200,
                     "flow=From-Botnet-V42-TCP-Attempt"])
        rows.append([ts, 1.0, "tcp", "147.32.84.50", 5000, "->", "1.1.1.1", 443, "FSPA_FSPA", 10, 5000, 900,
                     "flow=To-Background-UDP-CVUT-DNS-Server"])
        rows.append([ts, 1.0, "udp", "66.1.1.1", 53, "->", "147.32.84.50", 53, "CON", 2, 200, 100,
                     "flow=Background"])
    return pd.DataFrame(rows, columns=["StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr", "Dport",
                                       "State", "TotPkts", "TotBytes", "SrcBytes", "Label"])


def test_host_mode_cells_and_lateral_rule():
    flows = normalize_frame(_ctu_rows())
    flows = flows[flows["src_ip"].str.startswith("147.32.")]
    w = build_windows_from_frame(flows, 60, by_host=True, internal_prefixes=("147.32.",))
    assert set(w["host"]) == {"147.32.84.165", "147.32.84.50"}
    bot = w[w["host"] == "147.32.84.165"].sort_values("wid")
    assert bot["cnt_3"].min() == 1                     # SMB attempt to an internal host -> Lateral Movement
    assert (bot["stage"].iloc[:10] == 1).all()         # DNS dominates early (dominant rule)
    assert (bot["stage"].iloc[12:] == 6).all() or (bot["stage"].iloc[12:].isin([3, 6])).all()
    assert (w[w["host"] == "147.32.84.50"]["stage"] == 0).all()
    assert w.groupby("segment")["host"].nunique().max() == 1


def test_split_protocols():
    w = pd.DataFrame({"capture": np.repeat(["1", "2", "5"], 100), "host": "h", "wid": np.tile(np.arange(100), 3),
                      "segment": np.repeat([0, 1, 2], 100), "ts": 0.0})
    t = assign_splits(w, {"train": 0.7, "val": 0.15, "test": 0.15}, purge=5, protocol="temporal")
    for cap, g in w.assign(s=t).groupby("capture"):
        assert g.loc[g.s == "train", "wid"].max() < g.loc[g.s == "val", "wid"].min() - 4
        assert g.loc[g.s == "val", "wid"].max() < g.loc[g.s == "test", "wid"].min() - 4
    f = assign_splits(w, {}, 0, "family", {"train": ["1"], "val": ["2"], "test": ["5"]})
    assert set(w.loc[f == "test", "capture"]) == {"5"}
    chk = leakage.split_integrity(w.assign(s=t), "s", "temporal")
    assert chk["verdict"].startswith("OK")
    bad = w.assign(s=np.where(w["wid"] % 2, "train", "test"))
    assert leakage.split_integrity(bad, "s", "temporal")["verdict"].startswith("FAIL")


def test_no_peeking_is_falsifiable(cfg):
    m = build_model(12, cfg)
    r = leakage.no_peeking(m, 12, steps=8, horizon=4)
    assert r["encoder_grad_detached"] == 0.0 and r["encoder_grad_wired"] > 0.0 and r["verdict"].startswith("OK")


def test_permutation_test():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    good = leakage.permutation_test(y + rng.normal(0, 0.3, 2000), y, n=100)
    noise = leakage.permutation_test(rng.random(2000), y, n=100)
    assert good["verdict"].startswith("OK") and good["p_value"] < 0.02
    assert noise["verdict"].startswith("WARNING")


def test_schedule_probe_detects_timing_leak():
    ts = pd.Timestamp("2018-02-14").timestamp() + np.arange(3000) * 60.0
    df = pd.DataFrame({"ts": ts, "capture": "d"})
    df["y_future"] = (pd.to_datetime(df["ts"], unit="s").dt.hour.between(10, 11)).astype(int)  # attacks on a timetable
    r = leakage.schedule_probe(df.iloc[::2], df.iloc[1::2], reference_ap=0.9)
    assert r["schedule_ap"] > 0.9 and r["verdict"].startswith("WARNING") and r["schedule_share_of_lift"] > 0.9


def test_robust_z_and_triage():
    g = pd.Series(["a"] * 50)
    v = np.r_[np.ones(49), 100.0]
    z = robust_z(pd.Series(v), g)
    assert z[-1] > 10 and abs(np.median(z)) < 1e-6
    fr = pd.DataFrame({"segment": np.repeat(np.arange(10), 5), "capture": "a", "host": np.repeat(list("abcdefghij"), 5),
                       "stage": np.r_[np.zeros(45, int), np.ones(5, int)], "risk": np.r_[np.zeros(45), np.full(5, .9)],
                       "surprise_obs": np.r_[np.ones(40) + np.random.default_rng(0).normal(0, .01, 40), np.full(5, 9.0),
                                             np.ones(5)]})
    t = host_triage(fr, 0.5, 3.0)
    assert t["infected"] == 1 and t["caught"] == 1 and t["caught_by_risk"] == 1 and t["false_alarms"] == 1


def test_stage_horizon_and_future_stage():
    stage = np.array([0, 1, 1, 6, 6, 6, 0, 1, 4, 4])
    seg = np.array([0] * 6 + [1] * 4)
    tgt, ok = future_stage(stage, seg, 2)
    assert ok.tolist() == [True] * 4 + [False, False] + [True, True, False, False]
    assert tgt[:4].tolist() == [1, 6, 6, 6]
    fr = pd.DataFrame({"segment": seg, "stage": stage})
    tab = stage_horizon_table(fr, {"oracle": lambda k: np.roll(stage, -k), "persist": lambda k: stage}, [1, 2])
    o = tab[(tab.method == "oracle") & (tab.k == 1)].iloc[0]
    p = tab[(tab.method == "persist") & (tab.k == 1)].iloc[0]
    assert o["accuracy"] == 1.0 and p["transition_acc"] == 0.0


def test_subsample_keeps_positives():
    y = np.r_[np.ones(10), np.zeros(1000)].astype(int)
    r = subsample(y, 100)
    assert y[r].sum() == 10 and len(r) <= 100
