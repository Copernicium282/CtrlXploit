"""NTRO SOC Command Dashboard - AI based Network Attack Forecasting (SIH 26153).

    streamlit run app.py

Fully offline. Loads a trained NetWorldModel bundle (real CTU-13 per-host model by
default) and runs ingestion -> state cells -> world-model filtering -> K-step particle
rollouts -> fusion + label-free surprise channel -> alerts, all locally.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sih_v2.config import load_config, resolve  # noqa: E402
from sih_v2.constants import EXPLOIT_STAGES, MITRE, STAGE_COLORS, STAGES  # noqa: E402
from sih_v2.engine.alerts import generate_alerts  # noqa: E402
from sih_v2.engine.bundle import load_bundle  # noqa: E402
from sih_v2.engine.evaluation import summary_table  # noqa: E402
from sih_v2.engine.explain import gradient_shap  # noqa: E402
from sih_v2.engine.metrics import host_triage  # noqa: E402
from sih_v2.engine.pipeline import forecast_files  # noqa: E402
from sih_v2.engine.simulate import ForecastEngine, ForecastResult  # noqa: E402
from sih_v2.features.sequences import context_index  # noqa: E402
from sih_v2.features.windows import BASE_FEATURES  # noqa: E402

st.set_page_config(page_title="NTRO Attack Forecasting SOC", page_icon="T", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1500px;}
div[data-testid="stMetricValue"] {font-size: 1.45rem; font-family: monospace;}
.sev-CRITICAL {color:#ff5252;font-weight:700} .sev-HIGH {color:#ff9800;font-weight:700}
.sev-MEDIUM {color:#ffd54f;font-weight:700} .sev-OK {color:#66bb6a;font-weight:700}
.small {font-size:0.85rem; opacity:0.8}
/* blocky: square corners, hard borders, no shadows */
section[data-testid="stSidebar"] {border-right: 2px solid #2a3b52 !important;}
div[data-testid="stTabs"] button {border-radius: 0 !important; border: 1px solid #2a3b52 !important;
    text-transform: uppercase; letter-spacing: .06em; font-weight: 700;}
div[data-testid="stTabs"] button[aria-selected="true"] {border-color: #ff4b4b !important;
    border-bottom: 3px solid #ff4b4b !important;}
div[data-testid="metric-container"] {border: 1px solid #2a3b52; border-radius: 0;
    padding: .5rem .6rem; background: #0b1220;}
.stButton button, .stDownloadButton button {border-radius: 0 !important;
    border: 1px solid #3d5a80 !important; text-transform: uppercase;
    letter-spacing: .08em; font-weight: 700;}
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div {border-radius: 0 !important;}
hr {border-color: #2a3b52 !important;}
</style>""", unsafe_allow_html=True)

PHASE_COLORS = {"pre_attack": "rgba(255,193,7,0.18)", "during_attack": "rgba(244,67,54,0.18)"}
MODELS = {
    "CTU-13 · real botnet traffic · per-host · temporal split": ("ctu13", "temporal"),
    "CTU-13 · real · per-host · held-out malware families": ("ctu13", "family"),
    "CSE-CIC-IDS2018 · real · network-wide · temporal split": ("cic2018", "temporal"),
    "Synthetic campaigns · controlled · network-wide": ("synthetic", "temporal"),
}
SAMPLES = {
    "Bundled sample - CSE-CIC-IDS2018-layout CSV (synthetic, held-out)": "data/sample/sample_cic2018.csv",
    "Bundled sample - CTU-13-layout binetflow (synthetic)": "data/sample/sample_ctu13.binetflow",
    "Bundled sample - raw PCAP capture (synthetic)": "data/sample/sample_capture.pcap",
}


# ----------------------------------------------------------------------------- loading
def available_models():
    out = {}
    for label, (ds, pr) in MODELS.items():
        cfg = load_config(dataset=ds, protocol=pr)
        if resolve(cfg["paths"]["model_bundle"]).exists():
            out[label] = (ds, pr)
    return out


@st.cache_resource
def get_bundle(ds: str, pr: str):
    cfg = load_config(dataset=ds, protocol=pr)
    b = load_bundle(resolve(cfg["paths"]["model_bundle"]))
    bp = resolve(cfg["paths"]["baseline_bundle"])
    pack = joblib.load(bp) if bp.exists() else None
    return cfg, b, pack


@st.cache_data(show_spinner="Loading the held-out test split and its pre-computed forward simulation...")
def load_test(ds: str, pr: str, particles: int, steps: int):
    cfg, b, _ = get_bundle(ds, pr)
    df = pd.read_parquet(resolve(cfg["paths"]["processed"]))
    df = df[df[f"split_{pr}"] == "test"].reset_index(drop=True)
    rep = resolve(cfg["paths"]["reports_dir"])
    eng = b.config["engine"]
    t = time.perf_counter()
    if (rep / "test_predictions.parquet").exists() and particles == eng["particles"] and steps == eng["rollout_steps"]:
        fr = pd.read_parquet(rep / "test_predictions.parquet")
        fr = fr[[c for c in fr.columns if not c.startswith("score::")]]
        a = np.load(rep / "test_arrays.npz")
        X = b.scaler.transform(df[b.feature_names].to_numpy(np.float32))
        res = ForecastResult(fr, a["rollout_mean"].astype(np.float32), a["exploit_q"].astype(np.float32),
                             a["markov_path"].astype(np.float32), a["attention"].astype(np.float32),
                             a["stage_now"].astype(np.float32), X)
        src = "held-out test split (pre-computed)"
    else:
        res = ForecastEngine(b, particles=particles, steps=steps).run(df)
        src = "held-out test split"
    return res, df, {"cells": len(df), "segments": int(df["segment"].nunique()), "sec": time.perf_counter() - t,
                     "source": src}


@st.cache_data(show_spinner="Ingesting telemetry and running forward simulation...")
def run_file(ds: str, pr: str, path: str, particles: int, steps: int, _digest: str = ""):
    _, b, _ = get_bundle(ds, pr)
    t = time.perf_counter()
    r, states, stats = forecast_files([path], b, particles=particles, steps=steps)
    return r, states, {**stats, "sec": time.perf_counter() - t, "source": Path(path).name}


def save_upload(up) -> tuple[str, str]:
    data = up.getvalue()
    digest = hashlib.sha1(data).hexdigest()[:16]
    d = Path(tempfile.gettempdir()) / "sih_v2_uploads"
    d.mkdir(exist_ok=True)
    p = d / f"{digest}_{Path(up.name).name}"
    if not p.exists():
        p.write_bytes(data)
    return str(p), digest


# ----------------------------------------------------------------------------- figures
def runs(mask: np.ndarray):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(idx[::2], idx[1::2]))


def timeline_fig(f: pd.DataFrame, thr: float, zthr: float, upto: int, show_components: bool, labelled: bool):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.46, 0.13, 0.2, 0.21],
                        subplot_titles=("Forecast risk: P(exploitation within horizon)", "Kill-chain stage",
                                        "Label-free surprise (robust z within capture)", "Telemetry"))
    x = f["time"]
    if labelled:
        for ph, col in PHASE_COLORS.items():
            for a, b in runs((f["phase"] == ph).to_numpy()):
                fig.add_vrect(x0=x.iloc[a], x1=x.iloc[min(b, len(f) - 1)], fillcolor=col, line_width=0, row=1, col=1)
    live = f.iloc[: upto + 1]
    if show_components:
        for c, n, d in (("risk_direct", "direct head", "dot"), ("risk_rollout", "K-step rollout", "dash"),
                        ("risk_markov", "Markov prior", "dashdot")):
            fig.add_trace(go.Scatter(x=live["time"], y=live[c], name=n, line=dict(width=1, dash=d), opacity=0.8),
                          row=1, col=1)
    fig.add_trace(go.Scatter(x=live["time"], y=live["risk"], name="fused risk", line=dict(color="#ff4b4b", width=2.5),
                             fill="tozeroy", fillcolor="rgba(255,75,75,0.10)"), row=1, col=1)
    fig.add_hline(y=thr, line=dict(color="#ffd54f", dash="dash"), annotation_text=f"threshold {thr:.2f}", row=1, col=1)
    al = live[live["risk"] >= thr]
    fig.add_trace(go.Scatter(x=al["time"], y=al["risk"], mode="markers", name="risk alert",
                             marker=dict(color="#ff1744", size=7, symbol="triangle-up")), row=1, col=1)
    if labelled:
        fig.add_trace(go.Scatter(x=x, y=["truth"] * len(f), mode="markers", showlegend=False,
                                 marker=dict(symbol="square", size=8, color=[STAGE_COLORS[STAGES[s]] for s in f["stage"]]),
                                 text=[STAGES[s] for s in f["stage"]], hovertemplate="%{text}<extra>truth</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=live["time"], y=["model"] * len(live), mode="markers", showlegend=False,
                             marker=dict(symbol="square", size=8, color=[STAGE_COLORS[s] for s in live["current_stage_pred"]]),
                             text=live["current_stage_pred"], hovertemplate="%{text}<extra>model</extra>"), row=2, col=1)
    if "surprise_z" in live:
        fig.add_trace(go.Scatter(x=live["time"], y=live["surprise_z"].clip(-5, 20), name="surprise z",
                                 line=dict(color="#26c6da", width=1.5)), row=3, col=1)
        fig.add_hline(y=zthr, line=dict(color="#26c6da", dash="dot"), row=3, col=1)
    if "n_flows_raw" in f:
        fig.add_trace(go.Bar(x=live["time"], y=live["n_flows_raw"], name="flows / window", marker_color="#4fc3f7",
                             opacity=0.7), row=4, col=1)
    fig.add_vline(x=f["time"].iloc[upto], line=dict(color="white", width=1, dash="dot"))
    fig.update_yaxes(range=[0, 1.02], row=1, col=1)
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.07),
                      template="plotly_dark", hovermode="x unified")
    return fig


def rollout_fig(res, gi: int, window_min: float):
    K = res.rollout_mean.shape[1]
    steps = np.arange(1, K + 1) * window_min
    q = res.exploit_q[gi]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Imagined stage distribution (particle mean)",
                                                        "P(exploitation) per future step - particle fan"))
    for c, s in enumerate(STAGES):
        fig.add_trace(go.Scatter(x=steps, y=res.rollout_mean[gi, :, c], stackgroup="one", name=s,
                                 line=dict(width=0.5, color=STAGE_COLORS[s])), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.r_[steps, steps[::-1]], y=np.r_[q[:, 2], q[::-1, 0]], fill="toself",
                             fillcolor="rgba(255,75,75,0.2)", line=dict(width=0), name="10-90% particles"), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=q[:, 1], name="median particle", line=dict(color="#ff4b4b", width=2.5)),
                  row=1, col=2)
    mk = res.markov_path[gi][:, EXPLOIT_STAGES].sum(-1)
    fig.add_trace(go.Scatter(x=steps, y=mk, name="Markov prior", line=dict(color="#ffd54f", dash="dash")), row=1, col=2)
    fig.update_xaxes(title_text="minutes ahead")
    fig.update_yaxes(range=[0, 1.02])
    fig.update_layout(height=400, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def status_of(risk: float, thr: float):
    """Status of the 10-minute FORECAST: EARLY WARNING once the forecast risk crosses the threshold."""
    if risk < thr:
        return "OK", "NORMAL"
    return ("CRITICAL" if risk >= 0.9 else "HIGH" if risk >= 0.75 else "MEDIUM"), "EARLY WARNING"


# ----------------------------------------------------------------------------- sidebar
models = available_models()
if not models:
    st.error("No trained bundle found. Run `python -m sih_v2 build_dataset --dataset ctu13` then "
             "`python -m sih_v2 train --dataset ctu13` (see docs/README.md).")
    st.stop()
with st.sidebar:
    st.markdown("## [ NTRO · ATTACK FORECASTING ]")
    st.caption("ThreatAhead · SIH 26153 · NetWorldModel v2 · OFFLINE")
    model_label = st.selectbox("Model", list(models), index=0)
    ds, pr = models[model_label]
    cfg, bundle, pack = get_bundle(ds, pr)
    # Offer only sources whose files exist, so a fresh clone never opens on a crash:
    # the held-out split needs `make fetch-data`; the bundled samples ship in data/sample/.
    has_test = resolve(cfg["paths"]["processed"]).exists()
    samples = [k for k, v in SAMPLES.items() if resolve(v).exists()]
    sources = (["Held-out test split (labelled)"] if has_test else []) + samples + ["Upload file"]
    src = st.radio("Telemetry source", sources, index=0)
    if not has_test:
        st.warning("Held-out real-data split not downloaded yet. Run **`make fetch-data`** "
                   "(Windows: **`.\\make.ps1 fetch-data`**) once - about 25 s - then refresh this page.")
    up = None
    if src == "Upload file":
        up = st.file_uploader("CSV (CIC-IDS2017/2018, CTU-13 binetflow, UNSW, NetFlow export), Parquet, PCAP/PCAPNG",
                              type=["csv", "gz", "parquet", "binetflow", "pcap", "pcapng", "cap"])
    st.markdown("---")
    st.markdown("**Forward simulation**")
    particles = st.select_slider("Particles", [4, 8, 16, 32, 64], value=int(bundle.config["engine"]["particles"]))
    steps = st.select_slider("Rollout horizon K (windows)", [5, 10, 15, 20],
                             value=int(bundle.config["engine"]["rollout_steps"]))
    ens = bundle.meta.get("ensemble") if pack is not None else None
    score_opts = ["World model (fused)"] + ([f"Ensemble: world model ⊕ {ens['partner']}"] if ens else [])
    score_mode = st.radio("Risk score", score_opts, index=0,
                          help="The ensemble blend weight, partner model and threshold were fitted on validation only. "
                               + (f"Validation weight on the world model: {ens['w_world_model']:.1f}." if ens else ""))
    use_ens = score_mode != score_opts[0]
    thr0 = ens["threshold"] if use_ens else bundle.threshold
    thr = st.slider("Risk alert threshold", 0.01, 0.99, float(round(thr0, 2)), 0.01, key=f"thr_{score_mode}",
                    help="Calibrated on validation data with FPR <= %.0f%%" % (100 * bundle.config["engine"]["max_fpr"]))
    zthr = st.slider("Surprise threshold (robust z)", 1.0, 10.0, float(bundle.config["engine"].get("surprise_z", 3.0)), 0.5)
    min_consec = st.number_input("Debounce (consecutive windows)", 1, 5, int(bundle.config["engine"]["alert_min_consecutive"]))
    show_comp = st.checkbox("Show fusion components", value=False)
    st.markdown("---")
    fw = bundle.config["engine"]["fusion"]
    W = float(bundle.config["features"]["window_seconds"])
    WMIN = W / 60.0
    H = int(bundle.config["forecast"]["horizon"])
    st.markdown(f"<div class='small'>{ds} · {pr} · {bundle.config['data']['mode']} cells · seed "
                f"{bundle.meta.get('seed', '?')}<br>Window {W:.0f}s · context {bundle.config['forecast']['context']} · "
                f"horizon {H} ({H * WMIN:.0f} min)<br>Fusion d/r/m = {fw['direct']:.2f}/{fw['rollout']:.2f}/"
                f"{fw['markov']:.2f} · surprise sign {bundle.meta.get('surprise_sign', 1):+.0f}<br>"
                f"Trained {bundle.meta.get('trained_at', '?')}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- run
if not has_test:
    st.info(("**Demo mode:** showing the bundled sample data. " if samples else "**No data found yet.** ")
            + "The real CTU-13 / CIC-IDS2018 held-out test views "
            "(and the benchmark numbers behind them) appear after a one-time **`make fetch-data`** "
            "(Windows: **`.\\make.ps1 fetch-data`**), about 25 s. Everything runs offline afterwards.")
if src == "Held-out test split (labelled)":
    res, states, stats = load_test(ds, pr, particles, steps)
elif src == "Upload file":
    if up is None:
        st.info("Upload a telemetry file in the sidebar - or pick a bundled sample. Supported: CSE-CIC-IDS2017/2018 "
                "CICFlowMeter CSV, CTU-13 binetflow, UNSW-NB15, generic NetFlow/IPFIX CSV with a timestamp column, "
                "Parquet, PCAP and PCAPNG. The CTU-13 model scores every private-range / monitored source host "
                "separately; network-wide models score the capture as a whole.")
        st.stop()
    path, digest = save_upload(up)
    try:
        res, states, stats = run_file(ds, pr, path, particles, steps, digest)
    except Exception as exc:  # readable error instead of a stack trace
        st.error(f"Could not process `{up.name}`: {exc}")
        st.stop()
else:
    res, states, stats = run_file(ds, pr, str(resolve(SAMPLES[src])), particles, steps)

frame = res.frame.copy()


@st.cache_data(show_spinner="Scoring the tree-ensemble partner...")
def tree_scores(key: str, _X, seg):
    tree = next(e for e in pack["sklearn"] if e["model"].name == ens["partner"])
    return tree["model"].score(_X, seg)


if use_ens:
    frame["risk_world_model"] = frame["risk"]
    ts = tree_scores(f"{model_label}|{stats.get('source')}|{len(frame)}", res.X, frame["segment"].to_numpy())
    frame["risk"] = ens["w_world_model"] * frame["risk"] + (1 - ens["w_world_model"]) * ts
frame["alert"] = (frame["risk"] >= thr).astype(int)
frame["anomaly"] = (frame["surprise_z"] >= zthr).astype(int)
labelled = "stage" in frame and frame["stage"].gt(0).any()
segs = list(dict.fromkeys(frame["segment"]))
per_host = "host" in frame

hdr1, hdr2 = st.columns([3, 2])
hdr1.markdown("# ThreatAhead · SOC Command Dashboard")
hdr1.caption("Forecasting attacker progression with a latent world model of traffic dynamics · "
             "P(S<sub>t+1</sub> | S<sub>t</sub>) · K-step particle rollouts · label-free surprise channel · "
             "runs fully OFFLINE",
             unsafe_allow_html=True)
hdr2.markdown(f"<div class='small' style='text-align:right'>Model: <b>{model_label}</b><br>Source: <b>{stats.get('source')}</b>"
              f"<br>{len(frame):,} cells · {len(segs)} {'host-slot' if per_host else 'segment'}(s)"
              + (f" · {stats['flows']:,} flows" if "flows" in stats else "")
              + f"<br>{stats.get('sec', 0):.2f}s · {particles} particles × K={steps} · OFFLINE</div>", unsafe_allow_html=True)

# host-level triage (both channels)
tri = host_triage(frame, thr, zthr, 1.0)
tri_tab = tri.pop("table").sort_values(["by_risk", "max_risk", "surprise_z"], ascending=False)


def seg_label(s):
    g = frame[frame["segment"] == s]
    who = f"{g['capture'].iloc[0]} · {g['host'].iloc[0]}" if per_host else f"segment {s}"
    tag = ""
    if labelled:
        tag = " · infected" if g["stage"].gt(0).any() else " · benign"
    r = tri_tab.loc[tri_tab["segment"] == s]
    flag = (" [!]" if (r["by_risk"].any() or r["by_anomaly"].any()) else "") if len(r) else ""
    return f"{who} · {g['time'].iloc[0]:%Y-%m-%d %H:%M}{tag}{flag}"


order = list(tri_tab["segment"])
order += [s for s in segs if s not in order]
default_seg = order[0]
if labelled:
    inf = frame[frame["stage"] > 0].groupby("segment").size()
    pre = frame[frame["phase"] == "pre_attack"].groupby("segment").size()
    default_seg = (pre.idxmax() if len(pre) else inf.idxmax()) if len(inf) else order[0]
c1, c2 = st.columns([2, 3])
seg = c1.selectbox("Host-slot (ranked by triage)" if per_host else "Monitoring segment", order,
                   index=order.index(default_seg), format_func=seg_label)
f = frame[frame["segment"] == seg].reset_index()
gidx = f["index"].to_numpy()
if "pos" not in st.session_state or st.session_state.get("pos_seg") != (model_label, src, seg):
    first_alert = np.flatnonzero(f["risk"].to_numpy() >= thr)
    st.session_state.pos = int(min(first_alert[0] + 3, len(f) - 1)) if len(first_alert) else len(f) - 1
    st.session_state.pos_seg = (model_label, src, seg)
pos = c2.slider("Replay position (live cursor)", 0, max(len(f) - 1, 1), key="pos",
                help="Everything right of the cursor is the future the model has not seen yet")
pos = min(pos, len(f) - 1)
row = f.iloc[pos]
gi = int(row["index"])

tabs = st.tabs(["LIVE MONITOR", "HOST TRIAGE", "FORWARD SIM", "MITRE ATT&CK", "EXPLAIN",
                "BENCHMARK", "ALERTS"])

# ----------------------------------------------------------------------------- live monitor
with tabs[0]:
    sev, state = status_of(row["risk"], thr)
    k = st.columns(6)
    k[0].metric("Fused risk", f"{row['risk']:.2f}", f"{row['risk'] - f['risk'].iloc[max(pos - 1, 0)]:+.2f}")
    k[1].markdown(f"**10-min forecast status**<br><span class='sev-{sev}' style='font-size:1.3rem'>{state}</span>"
                  f"<br><span class='small'>{sev}</span>", unsafe_allow_html=True)
    k[2].metric("Current stage (model estimate)", row["current_stage_pred"],
                help="The model's estimate of what this host is doing now (detection). Compare with the 'truth' strip below.")
    k[3].metric("Predicted next stage (next 10 min)", row["forecast_stage"] if row["risk"] >= thr else "-",
                help="The exploitation stage the imagined futures reach within the horizon (shown when risk >= threshold).")
    eta = row["eta_windows"]
    k[4].metric("ETA to exploitation", f"{eta * WMIN:.0f} min" if eta > 0 and row["risk"] >= thr else "-")
    k[5].metric("Surprise z", f"{row['surprise_z']:.1f}", "novel" if row["surprise_z"] >= zthr else "typical",
                delta_color="inverse" if row["surprise_z"] >= zthr else "off")
    st.plotly_chart(timeline_fig(f, thr, zthr, pos, show_comp, labelled), width="stretch")
    if labelled:
        cc = st.columns(3)
        for col, ph, name in ((cc[0], "pre_attack", "Pre-attack cells alerted (early warning)"),
                              (cc[1], "during_attack", "During-attack cells alerted"),
                              (cc[2], "benign", "Benign cells alerted (false alarm rate)")):
            m = f["phase"] == ph
            col.metric(name, f"{f.loc[m, 'alert'].mean():.0%}" if m.any() else "-", f"{int(m.sum())} cells",
                       delta_color="off")
        st.caption("Shading: amber = pre-attack cells (exploitation begins within the horizon), red = active "
                   "exploitation. They are scored separately so detection of an ongoing attack is never counted as "
                   "forecasting. On CTU-13 every infected host is malicious from its first observed minute (no "
                   "pre-infection baseline exists in the capture), so pre-attack cells there are reconnaissance "
                   "phases that precede C2 / spam - stage *progression* is the forecastable quantity.")
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("REPLAY HOST-SLOT", help="Animate the live cursor from the start"):
        ph = st.empty()
        for p in range(0, len(f), max(1, len(f) // 60)):
            r = f.iloc[p]
            s_, t_ = status_of(r["risk"], thr)
            ph.markdown(f"**t = {r['time']:%H:%M}** · risk **{r['risk']:.2f}** · surprise z {r['surprise_z']:.1f} · "
                        f"<span class='sev-{s_}'>{t_}</span> · current stage `{r['current_stage_pred']}`"
                        + (f" · predicted next `{r['forecast_stage']}`" if r["risk"] >= thr else ""), unsafe_allow_html=True)
            time.sleep(0.08)

    def _jump(first=np.flatnonzero(f["risk"].to_numpy() >= thr)):
        if len(first):
            st.session_state.pos = int(first[0])

    b2.button("JUMP TO FIRST WARNING", on_click=_jump)

# ----------------------------------------------------------------------------- host triage
with tabs[1]:
    st.markdown("### Two-channel host triage")
    st.caption("**Risk** = the supervised forecast crossed its calibrated threshold in any window of the host-slot. "
               "**Anomaly** = the host-slot's mean world-model surprise (prior prediction error, no labels) is more "
               f"than {zthr:g} robust deviations (median/MAD) above the median host-slot of the same capture. The two "
               "fail in different regimes, so they are shown side by side rather than blended.")
    m = st.columns(4)
    if labelled:
        m[0].metric("Infected host-slots caught", f"{tri['caught']}/{tri['infected']}")
        m[1].metric("…only by the anomaly channel", tri["caught_only_by_anomaly"])
        m[2].metric("Benign host-slots flagged", f"{tri['false_alarms']}/{tri['benign']}",
                    f"{tri['false_alarm_rate']:.1%}", delta_color="off")
        m[3].metric("…by the risk channel alone", tri["risk_only_false_alarms"])
    show = tri_tab.assign(channel=np.where(tri_tab["by_risk"] & tri_tab["by_anomaly"], "risk + anomaly",
                                           np.where(tri_tab["by_risk"], "risk",
                                                    np.where(tri_tab["by_anomaly"], "anomaly", "-"))))
    cols = ["capture", "host", "channel", "max_risk", "surprise_z", "cells"] + (["infected"] if labelled else [])
    st.dataframe(show[cols].round(3), hide_index=True, width="stretch", height=420)
    sc = go.Figure()
    sc.add_trace(go.Scatter(x=tri_tab["surprise_z"].clip(-5, 30), y=tri_tab["max_risk"], mode="markers",
                            marker=dict(size=8, color=np.where(tri_tab["infected"], "#ff5252", "#90a4ae") if labelled
                                        else "#90a4ae"),
                            text=tri_tab["host"].astype(str), hovertemplate="%{text}<br>z=%{x:.1f} risk=%{y:.2f}"))
    sc.add_hline(y=thr, line=dict(color="#ffd54f", dash="dash"))
    sc.add_vline(x=zthr, line=dict(color="#26c6da", dash="dot"))
    sc.update_layout(height=420, template="plotly_dark", xaxis_title="host surprise (robust z)",
                     yaxis_title="max forecast risk", title="Every host-slot: supervised risk vs label-free surprise"
                     + (" (red = infected)" if labelled else ""), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(sc, width="stretch")

# ----------------------------------------------------------------------------- forward simulation
with tabs[2]:
    st.markdown(f"### K-step forward simulation at **{row['time']:%Y-%m-%d %H:%M}**")
    st.caption("The posterior belief is filtered over the last L windows; P particles are drawn from it and the learned "
               "latent dynamics p(z<sub>t+1</sub>|h<sub>t+1</sub>) are rolled forward K steps without observations. The "
               "stage head is read at every imagined step.", unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("Direct head", f"{row['risk_direct']:.2f}")
    m[1].metric("Rollout (particle mean of max-step)", f"{row['risk_rollout']:.2f}")
    m[2].metric("Markov kill-chain prior", f"{row['risk_markov']:.2f}")
    m[3].metric("Particles reaching exploitation", f"{row['particle_hit_frac']:.0%}")
    st.plotly_chart(rollout_fig(res, gi, WMIN), width="stretch")
    if labelled:
        fut = f.iloc[pos + 1: pos + 1 + res.rollout_mean.shape[1]]
        if len(fut):
            truth = pd.DataFrame({"minutes ahead": (np.arange(len(fut)) + 1) * WMIN,
                                  "true stage": [STAGES[s] for s in fut["stage"]],
                                  "imagined most-likely stage": [STAGES[i] for i in res.rollout_mean[gi, :len(fut)].argmax(-1)],
                                  "imagined P(exploit)": res.rollout_mean[gi, :len(fut)][:, EXPLOIT_STAGES].sum(-1).round(3)})
            with st.expander("Ground truth vs imagination (labelled data only)", expanded=True):
                st.dataframe(truth, hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- MITRE
with tabs[3]:
    now = res.stage_now[gi]
    fut = res.rollout_mean[gi].max(0)
    cols = st.columns([3, 2])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=STAGES[1:], x=now[1:], name="current (filtered)", orientation="h", marker_color="#4fc3f7"))
    fig.add_trace(go.Bar(y=STAGES[1:], x=fut[1:], name=f"next {res.rollout_mean.shape[1]} windows (max over rollout)",
                         orientation="h", marker_color="#ff4b4b"))
    fig.update_layout(barmode="group", height=380, template="plotly_dark", xaxis=dict(range=[0, 1], title="probability"),
                      yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=30, b=10),
                      title="Kill-chain position: now vs forecast")
    cols[0].plotly_chart(fig, width="stretch")
    tbl = pd.DataFrame([{"Stage": s, "Tactic": MITRE[s]["tactic"], "ID": MITRE[s]["id"],
                         "Techniques": ", ".join(MITRE[s]["techniques"]), "Now": round(float(now[i]), 3),
                         "Forecast": round(float(fut[i]), 3)} for i, s in enumerate(STAGES) if i > 0])
    cols[1].dataframe(tbl, hide_index=True, width="stretch", height=380)
    c3, c4 = st.columns(2)
    P = bundle.markov.P
    hm = go.Figure(go.Heatmap(z=P, x=STAGES, y=STAGES, colorscale="Reds", zmin=0, zmax=1,
                              text=np.round(P, 2), texttemplate="%{text}"))
    hm.update_layout(height=420, template="plotly_dark", title="Learned kill-chain transition prior P(stage_t+1 | stage_t)",
                     xaxis_title="to", yaxis_title="from", margin=dict(l=10, r=10, t=40, b=10))
    c3.plotly_chart(hm, width="stretch")
    strip = res.rollout_mean[gidx][:, :, EXPLOIT_STAGES].max(1)
    sf = go.Figure(go.Heatmap(z=strip.T, x=f["time"], y=[STAGES[i] for i in EXPLOIT_STAGES], colorscale="Inferno",
                              zmin=0, zmax=1))
    sf.update_layout(height=420, template="plotly_dark", title="Forecast tactic heat-strip across the host-slot",
                     margin=dict(l=10, r=10, t=40, b=10))
    c4.plotly_chart(sf, width="stretch")
    st.caption("CTU-13 mapping (documented in constants.py): spam / click-fraud → Impact (T1496 resource hijacking); "
               "DNS lookups, connection attempts, ICMP → Reconnaissance; CC / IRC / P2P / custom-encrypted → C2; "
               "malicious internal SMB/RPC/RDP/SSH → Lateral Movement (behavioural rule).")

# ----------------------------------------------------------------------------- explainability
with tabs[4]:
    st.markdown(f"### Why is the risk **{row['risk']:.2f}** at {row['time']:%H:%M}?")
    L = int(bundle.config["forecast"]["context"])
    ci = context_index(frame["segment"].to_numpy(), L)
    ca, cb = st.columns(2)
    att = res.attention[gi]
    ago = -(np.arange(L)[::-1]) * WMIN
    af = go.Figure(go.Bar(x=ago, y=att, marker_color="#ab47bc"))
    af.update_layout(height=330, template="plotly_dark", title="Temporal attention: which past windows drove the belief",
                     xaxis_title="minutes relative to now", yaxis_title="attention weight", margin=dict(l=10, r=10, t=40, b=10))
    ca.plotly_chart(af, width="stretch")

    @st.cache_data(show_spinner="Computing Expected-Gradients (GradientSHAP) attributions...")
    def explain(key: str, gi: int, _ctx, _bg):
        return gradient_shap(bundle, _ctx, _bg, n_samples=64)

    rng = np.random.default_rng(0)
    quiet = np.flatnonzero(frame["risk"].to_numpy() < thr)
    pick = rng.choice(quiet if len(quiet) > 10 else np.arange(len(frame)), 64)
    ex = explain(f"{model_label}|{stats.get('source')}|{particles}|{steps}", gi, res.X[ci[gi]], res.X[ci[pick]])
    pt = go.Figure(go.Bar(x=ago, y=ex["per_time"], marker_color="#26a69a"))
    pt.update_layout(height=330, template="plotly_dark", title="|attribution| per past window (GradientSHAP)",
                     xaxis_title="minutes relative to now", margin=dict(l=10, r=10, t=40, b=10))
    cb.plotly_chart(pt, width="stretch")
    top = ex["features"].head(15).iloc[::-1]
    sfig = go.Figure(go.Bar(y=top["feature"], x=top["attribution"], orientation="h",
                            marker_color=np.where(top["attribution"] > 0, "#ff5252", "#42a5f5")))
    sfig.update_layout(height=480, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                       title=f"Top drivers of forecast risk (base value {ex['base_value']:.2f} → {ex['risk']:.2f})",
                       xaxis_title="contribution to risk (red raises, blue lowers)")
    c5, c6 = st.columns([3, 2])
    c5.plotly_chart(sfig, width="stretch")
    with c6:
        st.markdown("**Breakdown by temporal scale** (instant · burst vs 3-window EMA · 15-window trend)")
        st.dataframe(ex["features"].head(12).round(4), hide_index=True, width="stretch", height=300)
        if pack is not None and pack.get("sklearn"):
            lr = pack["sklearn"][0]["model"]
            ls = lr.linear_shap(res.X[gi])[0].reshape(3, -1).sum(0)
            # Names come from the bundle, not the current BASE_FEATURES: a bundle
            # trained on an earlier feature set is narrower than today's code.
            _nb = len(bundle.feature_names) // 3
            lsd = pd.DataFrame({"feature": list(bundle.feature_names[:_nb]), "LR SHAP (logit)": ls})
            lsd = lsd.reindex(lsd["LR SHAP (logit)"].abs().sort_values(ascending=False).index).head(8)
            st.markdown("**Baseline comparison - exact linear SHAP of single-window logistic regression**")
            st.dataframe(lsd.round(3), hide_index=True, width="stretch")
    with st.expander("Feature glossary"):
        st.markdown("""
- **syn/ack/fin/rst/psh/urg_ratio** - TCP flag counts per packet (CTU-13: recovered from Argus `State` strings); **flag_bitmask_entropy** - diversity of per-flow flag combinations.
- **ttl_mean / ttl_std** - hop-count / OS fingerprint spread (PCAP input; absent in CICFlowMeter and Argus exports).
- **init_win_mean/std** - initial TCP window; **log_iat_mean/std** - inter-arrival timing; **beacon_score** - periodicity of repeated conversations (C2).
- **dst_port_entropy, n_dst_ports, n_dst_ips** - breadth of scanning / spam fan-out.
- **lateral_frac** - internal-to-internal flows; **sensitive_port_frac** - SSH/SMB/RDP/WinRM/DB ports; **log_out_in_ratio** - upload asymmetry.
- Suffix **__dev_short** = deviation from a 3-window EMA (bursts); **__trend_long** = 15-window EMA (slow ramps).
""")

# ----------------------------------------------------------------------------- benchmark & integrity
with tabs[5]:
    mp = resolve(cfg["paths"]["reports_dir"]) / "metrics.json"
    if not mp.exists():
        st.warning(f"Run `python -m sih_v2 evaluate --dataset {ds} --protocol {pr}` to produce the report.")
    else:
        rep = json.loads(mp.read_text())
        tb = summary_table(rep["summary"])
        st.markdown(f"### {ds} · {pr} protocol · {rep['test_cells']:,} held-out cells · seeds {rep['seeds']}")
        st.caption("Identical features, scaler, labels and threshold rule for every method (F1-optimal s.t. FPR ≤ "
                   f"{rep['config']['max_fpr']} on validation, frozen before test). Deep models: mean ± std over seeds; "
                   "95 % CI by bootstrap over host-slots.")
        show = tb.drop(columns=["F1 95% CI"]).copy()
        show.insert(3, "F1 95% CI", tb["F1 95% CI"].map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]" if isinstance(v, list) else "-"))
        num = [c for c in show.columns if c not in ("Method", "F1 95% CI", "Seeds")]
        st.dataframe(show.style.format({c: "{:.3f}" for c in num}, na_rep="–"), hide_index=True, width="stretch")
        g = st.columns(2)
        colors = ["#9e9e9e", "#42a5f5", "#1e88e5", "#26a69a", "#66bb6a", "#8d6e63", "#ab47bc", "#7e57c2", "#ffd54f",
                  "#ff4b4b"]
        for col, key, err in ((g[0], "PR-AUC", "PR-AUC ±"), (g[1], "F1", "F1 ±")):
            fg = go.Figure(go.Bar(x=tb["Method"], y=tb[key], marker_color=colors[: len(tb)],
                                  error_y=dict(type="data", array=tb[err].fillna(0)), text=tb[key].round(3),
                                  textposition="outside"))
            fg.update_layout(height=380, template="plotly_dark", title=key + " (± std over seeds)",
                             margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(tickangle=-30))
            col.plotly_chart(fg, width="stretch")
        sh = pd.DataFrame(rep.get("stage_horizon", []))
        if len(sh):
            st.markdown("#### Kill-chain stage forecasting per horizon")
            h1, h2 = st.columns(2)
            for col, key, title in ((h1, "macro_f1", "Macro-F1 (stages present, malicious host-slots)"),
                                    (h2, "transition_acc", "Accuracy on cells whose stage changes (persistence = 0)")):
                fg = go.Figure()
                for name, part in sh.groupby("method"):
                    fg.add_trace(go.Scatter(x=part["k"] * WMIN, y=part[key], mode="lines+markers", name=name,
                                            line=dict(width=3 if "ours" in name else 1.5)))
                fg.update_layout(height=380, template="plotly_dark", title=title, xaxis_title="minutes ahead",
                                 margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
                col.plotly_chart(fg, width="stretch")
        st.markdown("#### Evaluation-integrity checks")
        for c in rep.get("checks", []):
            icon = "[OK]" if c["verdict"].startswith(("OK", "N/A", "reported")) else "[!!]"
            extra = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k not in ("check", "verdict")}
            st.markdown(f"{icon} **{c['check']}** - {c['verdict']}  \n<span class='small'>{extra}</span>",
                        unsafe_allow_html=True)
        sur = rep.get("surprise", {})
        if sur:
            st.markdown(f"#### Label-free surprise channel\nwindow-level AUC **{sur.get('window_auc', float('nan')):.3f}** · "
                        f"host-level AUC **{sur.get('host_auc', float('nan')):.3f}** (supervised risk "
                        f"{sur.get('host_auc_risk', float('nan')):.3f}) · direction chosen on validation "
                        f"(sign {sur.get('sign', 1):+.0f})")
        comp = ROOT / "eval/results/comparison.md"
        if comp.exists():
            with st.expander("Cross-dataset comparison and the public repos' reported numbers (eval/results/comparison.md)"):
                st.markdown(comp.read_text())

# ----------------------------------------------------------------------------- alerts
with tabs[6]:
    alerts = generate_alerts(frame, thr, int(min_consec), W)
    if per_host and len(alerts):
        hosts = frame.groupby("segment")[["capture", "host"]].first()
        alerts = alerts.join(hosts, on="segment")
    st.markdown(f"### {len(alerts)} risk alert(s) across {len(segs)} {'host-slot' if per_host else 'segment'}(s)")
    if len(alerts):
        st.dataframe(alerts, hide_index=True, width="stretch")
        st.download_button("Download alerts CSV", alerts.to_csv(index=False).encode(), "alerts.csv", "text/csv")
    st.download_button("Download host triage CSV", tri_tab.to_csv(index=False).encode(), "host_triage.csv", "text/csv")
    exp = frame.drop(columns=[c for c in frame.columns if c.startswith("split")])
    st.download_button("Download per-cell forecasts CSV", exp.to_csv(index=False).encode(), "forecasts.csv", "text/csv")
