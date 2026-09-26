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

st.set_page_config(
    page_title="ThreatAhead | AI Network Attack Forecasting",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root{
    --ta-bg:#050b16;
    --ta-bg-2:#071223;
    --ta-sidebar:#07101e;
    --ta-panel:#0a1729;
    --ta-panel-2:#0d1d32;
    --ta-panel-3:#10243d;
    --ta-border:rgba(91,151,207,.16);
    --ta-border-strong:rgba(82,178,255,.34);
    --ta-text:#f3f7fc;
    --ta-muted:#8192a8;
    --ta-soft:#a9b8c9;
    --ta-blue:#3d9cff;
    --ta-cyan:#28c7ff;
    --ta-red:#ff5364;
    --ta-orange:#f3a64a;
    --ta-yellow:#e5c65a;
    --ta-green:#35d39a;
    --ta-purple:#8d7cff;
}

/* ---------- application shell ---------- */
html,body,[data-testid="stAppViewContainer"]{
    background:
        radial-gradient(circle at 82% 0%, rgba(22,88,150,.12), transparent 30%),
        radial-gradient(circle at 20% 100%, rgba(24,83,150,.07), transparent 28%),
        var(--ta-bg)!important;
    color:var(--ta-text)!important;
}
[data-testid="stHeader"]{background:transparent!important}
#MainMenu, footer{visibility:hidden}
.block-container{
    padding-top:1rem!important;
    padding-bottom:3rem!important;
    max-width:1480px;
}

/* ---------- sidebar / control center ---------- */
section[data-testid="stSidebar"]{
    background:
        linear-gradient(180deg, rgba(10,25,44,.98), rgba(5,13,24,.99))!important;
    border-right:1px solid rgba(82,178,255,.13)!important;
    box-shadow:8px 0 35px rgba(0,0,0,.18);
}
section[data-testid="stSidebar"] .block-container{
    padding:1.05rem .95rem 2rem!important;
}
section[data-testid="stSidebar"] hr{
    border-color:rgba(91,151,207,.13)!important;
    margin:1rem 0!important;
}
section[data-testid="stSidebar"] label{
    color:#aebed0!important;
    font-size:.73rem!important;
    font-weight:650!important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{
    color:#91a2b6;
}
.sidebar-brand{
    position:relative;
    padding:.2rem .15rem 1rem;
    margin-bottom:1rem;
    border-bottom:1px solid rgba(91,151,207,.12);
}
.sidebar-brand:before{
    content:"";
    display:block;
    width:34px;
    height:3px;
    margin-bottom:.7rem;
    border-radius:3px;
    background:linear-gradient(90deg,var(--ta-cyan),var(--ta-blue));
}
.sidebar-brand-title{
    color:#f5f8fc;
    font-size:1.02rem;
    font-weight:800;
    letter-spacing:-.015em;
}
.sidebar-brand-sub{
    margin-top:.28rem;
    color:#64778e;
    font-size:.63rem;
    letter-spacing:.06em;
    text-transform:uppercase;
}
.sidebar-section{
    color:#58708b;
    font-size:.61rem;
    font-weight:800;
    letter-spacing:.14em;
    text-transform:uppercase;
    margin:.15rem 0 .62rem;
}

/* ---------- inputs ---------- */
div[data-baseweb="select"]>div,
div[data-baseweb="input"]>div,
div[data-testid="stFileUploaderDropzone"],
div[data-testid="stTextInput"]>div{
    background:#0c192b!important;
    border:1px solid rgba(91,151,207,.16)!important;
    border-radius:8px!important;
}
div[data-baseweb="select"]>div:hover,
div[data-baseweb="input"]>div:hover{
    border-color:rgba(61,156,255,.38)!important;
}
input,textarea{
    background:#0c192b!important;
    color:#eaf2fb!important;
}
div[data-testid="stFileUploaderDropzone"]{
    padding:.55rem!important;
}
div[data-testid="stFileUploaderDropzone"] small,
div[data-testid="stFileUploaderDropzone"] span{
    color:#8192a8!important;
}
div[data-testid="stSlider"] [role="slider"],
div[data-testid="stSelectSlider"] [role="slider"]{
    background:var(--ta-cyan)!important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] div{
    border-radius:99px;
}

/* ---------- hero ---------- */
.hero-card{
    position:relative;
    padding:1.25rem 1.35rem;
    border-radius:13px;
    border:1px solid rgba(61,156,255,.20);
    background:
        radial-gradient(circle at 92% 18%, rgba(40,199,255,.11), transparent 24%),
        radial-gradient(circle at 70% 120%, rgba(61,156,255,.07), transparent 34%),
        linear-gradient(135deg,#0d1d32 0%,#091321 100%);
    margin:.05rem 0 .65rem;
    min-height:86px;
    box-sizing:border-box;
    overflow:hidden!important;
    box-shadow:0 14px 45px rgba(0,0,0,.22);
}
.hero-card:before{
    content:"";
    position:absolute;
    top:0;
    left:0;
    width:100%;
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(40,199,255,.65),transparent);
}
.hero-card:after{
    content:"";
    position:absolute;
    left:0;
    bottom:0;
    width:24%;
    height:2px;
    background:linear-gradient(90deg,var(--ta-cyan),transparent);
}
.hero-row{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1.5rem;
    width:100%;
    position:relative;
    z-index:1;
}
.hero-title{
    color:#f5f9fd;
    font-size:1.82rem;
    line-height:1.1;
    font-weight:820;
    letter-spacing:-.04em;
    margin:0!important;
    padding:0!important;
}
.hero-subtitle{
    color:#8fa2b7;
    font-size:.77rem;
    line-height:1.4;
    margin:.38rem 0 0!important;
    padding:0!important;
}
.hero-meta{
    color:#7e91a7;
    font-size:.65rem;
    line-height:1.45;
    text-align:right;
    white-space:nowrap;
}
.offline-badge{
    display:inline-flex;
    align-items:center;
    gap:6px;
    padding:5px 9px;
    border-radius:5px;
    background:rgba(53,211,154,.07);
    border:1px solid rgba(53,211,154,.24);
    color:#55dca7;
    font-size:.61rem;
    font-weight:800;
    letter-spacing:.08em;
}
.offline-badge:before{
    content:"";
    width:5px;
    height:5px;
    border-radius:50%;
    background:#35d39a;
    box-shadow:0 0 8px rgba(53,211,154,.65);
}

/* ---------- investigation context ---------- */
.context-strip{
    display:grid;
    grid-template-columns:1.35fr .85fr .7fr .55fr;
    gap:0;
    margin:.35rem 0 1rem;
    border:1px solid rgba(91,151,207,.15);
    border-radius:9px;
    background:rgba(8,20,35,.76);
    overflow:hidden;
    box-shadow:0 8px 25px rgba(0,0,0,.10);
}
.context-item{
    padding:.7rem .85rem;
    border-right:1px solid rgba(91,151,207,.09);
    min-width:0;
}
.context-item:last-child{border-right:0}
.context-label{
    color:#58708b;
    font-size:.64rem;
    font-weight:800;
    letter-spacing:.12em;
    text-transform:uppercase;
}
.context-value{
    margin-top:.2rem;
    color:#dbe7f3;
    font-size:.82rem;
    font-weight:600;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
}

/* ---------- headings ---------- */
.section-kicker{
    color:#4e9bce;
    font-size:.58rem;
    font-weight:800;
    letter-spacing:.15em;
    text-transform:uppercase;
    margin-bottom:.22rem;
}
.section-heading{
    color:#f0f5fa;
    font-size:1.2rem;
    font-weight:760;
    letter-spacing:-.025em;
    margin:0 0 .3rem;
}
.section-description{
    color:#7f91a5;
    font-size:.73rem;
    line-height:1.5;
    margin-bottom:.8rem;
}
.muted-note{
    color:#74879d;
    font-size:.68rem;
}

/* ---------- system status strip ---------- */
.system-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;margin:-.2rem 0 .85rem;border:1px solid rgba(91,151,207,.14);border-radius:9px;background:linear-gradient(145deg,rgba(10,25,43,.86),rgba(7,17,30,.92));overflow:hidden}
.system-item{padding:.65rem .85rem;border-right:1px solid rgba(91,151,207,.09);min-width:0}.system-item:last-child{border-right:0}
.system-top{display:flex;align-items:center;gap:.38rem;color:#70869e;font-size:.63rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}
.system-dot{width:6px;height:6px;border-radius:50%;background:var(--ta-green);box-shadow:0 0 7px rgba(53,211,154,.45);flex:0 0 6px}
.system-value{margin-top:.2rem;color:#e3edf7;font-size:.81rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.system-meta{margin-top:.08rem;color:#61768d;font-size:.67rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* ---------- forecast hero ---------- */
.forecast-hero{display:grid;grid-template-columns:minmax(0,1fr) 235px;gap:.8rem;margin:.3rem 0 1rem}
.forecast-chart-card,.threat-summary-card{border:1px solid rgba(61,156,255,.14);border-radius:11px;background:linear-gradient(145deg,#0b1a2d,#071321);box-shadow:0 12px 34px rgba(0,0,0,.14);overflow:hidden}
.forecast-chart-head{padding:.85rem 1rem .15rem}.forecast-chart-kicker{color:#4f9fd3;font-size:.56rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.forecast-chart-title{margin-top:.18rem;color:#eef5fb;font-size:.98rem;font-weight:760}.forecast-chart-sub{margin-top:.16rem;color:#70869d;font-size:.64rem}
.threat-summary-card{padding:1rem}.summary-kicker{color:#4f9fd3;font-size:.56rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.summary-title{margin-top:.18rem;color:#eef5fb;font-size:.92rem;font-weight:760}.summary-risk{margin-top:.7rem;font-size:1.7rem;line-height:1;font-weight:850;letter-spacing:-.04em}.summary-risk-critical{color:var(--ta-red)}.summary-risk-high{color:var(--ta-orange)}.summary-risk-medium{color:var(--ta-yellow)}.summary-risk-low{color:var(--ta-green)}
.summary-meter{height:5px;margin:.55rem 0 .85rem;border-radius:99px;background:#16263a;overflow:hidden}.summary-meter-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--ta-blue),var(--ta-cyan))}.summary-row{display:flex;justify-content:space-between;gap:.7rem;padding:.48rem 0;border-bottom:1px solid rgba(91,151,207,.08)}.summary-row:last-child{border-bottom:0}.summary-label{color:#687d94;font-size:.62rem}.summary-value{color:#dce7f1;font-size:.64rem;font-weight:700;text-align:right;max-width:62%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* ---------- investigation workspace ---------- */
.investigation-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem;margin:.75rem 0 1rem}.investigation-card{padding:.8rem .9rem;border:1px solid rgba(91,151,207,.13);border-radius:9px;background:linear-gradient(145deg,#0d1d31,#091523)}.investigation-label{color:#6d8299;font-size:.56rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase}.investigation-value{margin-top:.3rem;color:#edf4fa;font-size:1.05rem;font-weight:800;overflow-wrap:anywhere}.investigation-note{margin-top:.2rem;color:#647a91;font-size:.6rem}.investigation-panel{padding:.85rem 1rem;border:1px solid rgba(91,151,207,.11);border-radius:9px;background:rgba(9,21,36,.72);margin:.7rem 0}.investigation-panel-title{color:#dfe9f3;font-size:.72rem;font-weight:750}.investigation-panel-copy{margin-top:.22rem;color:#73889e;font-size:.64rem;line-height:1.5}
/* ---------- alert console ---------- */
.alert-console-head{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;margin:.2rem 0 .8rem}.alert-console-title{color:#eef5fb;font-size:1.35rem;font-weight:800;letter-spacing:-.025em}.alert-console-sub{color:#70869d;font-size:.67rem;margin-top:.18rem}.alert-counts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.65rem;margin:.8rem 0 1rem}.alert-count{padding:.72rem .8rem;border:1px solid rgba(91,151,207,.12);border-radius:8px;background:#0b192b}.alert-count-label{color:#7d93aa;font-size:.62rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase}.alert-count-value{margin-top:.22rem;color:#eef5fb;font-size:1.42rem;font-weight:820}.alert-count.critical{border-left:2px solid var(--ta-red)}.alert-count.high{border-left:2px solid var(--ta-orange)}.alert-count.medium{border-left:2px solid var(--ta-yellow)}.alert-count.total{border-left:2px solid var(--ta-blue)}
.alert-row{display:grid;grid-template-columns:1.15fr 1fr .8fr .8fr .75fr;gap:.7rem;align-items:center;padding:.72rem .85rem;border:1px solid rgba(91,151,207,.09);border-radius:8px;background:rgba(10,24,41,.72);margin:.38rem 0}.alert-row:hover{border-color:rgba(61,156,255,.22);background:rgba(12,29,49,.9)}.alert-primary{color:#e6eff7;font-size:.78rem;font-weight:750;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.alert-secondary{color:#71879d;font-size:.65rem;margin-top:.16rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.alert-severity{font-size:.68rem;font-weight:850;letter-spacing:.08em}.alert-severity.critical{color:var(--ta-red)}.alert-severity.high{color:var(--ta-orange)}.alert-severity.medium{color:var(--ta-yellow)}.alert-severity.low{color:var(--ta-green)}.alert-table-head{display:grid;grid-template-columns:1.15fr 1fr .8fr .8fr .75fr;gap:.7rem;padding:.4rem .85rem;color:#5f7790;font-size:.59rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase}

/* ---------- threat status ---------- */
.threat-banner{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1rem;
    padding:.82rem 1rem;
    border:1px solid rgba(255,83,100,.25);
    border-left:3px solid var(--ta-red);
    border-radius:9px;
    background:
        linear-gradient(90deg,rgba(255,83,100,.09),rgba(255,83,100,.025));
    margin:.2rem 0 1rem;
    box-shadow:inset 0 1px rgba(255,255,255,.015);
}
.threat-banner.normal{
    border-color:rgba(53,211,154,.22);
    border-left-color:var(--ta-green);
    background:linear-gradient(90deg,rgba(53,211,154,.07),rgba(53,211,154,.018));
}
.threat-title{
    color:#ff7480;
    font-size:.6rem;
    font-weight:850;
    letter-spacing:.13em;
    margin-bottom:.18rem;
}
.threat-banner.normal .threat-title{color:#55dca7}
.threat-copy{
    color:#bcc9d7;
    font-size:.76rem;
}
.threat-score{
    color:#f4f7fa;
    font-size:1.15rem;
    font-weight:820;
    white-space:nowrap;
}

/* ---------- KPI cards ---------- */
.metric-grid{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:.75rem;
    margin:.25rem 0 1.15rem;
}
.metric-card{
    position:relative;
    min-height:106px;
    padding:.95rem 1rem .9rem 1.05rem;
    border:1px solid rgba(91,151,207,.15);
    border-radius:10px;
    background:
        radial-gradient(circle at 100% 0%,rgba(61,156,255,.055),transparent 34%),
        linear-gradient(145deg,#0d1c30,#091321);
    overflow:hidden;
    box-sizing:border-box;
    transition:transform .15s ease,border-color .15s ease,background .15s ease;
}
.metric-card:before{
    content:"";
    position:absolute;
    left:0;
    top:12px;
    bottom:12px;
    width:2px;
    border-radius:2px;
    background:var(--metric-accent,var(--ta-blue));
}
.metric-card:hover{
    transform:translateY(-1px);
    border-color:rgba(61,156,255,.28);
    background:linear-gradient(145deg,#10213a,#0b1728);
}
.metric-label{
    color:#71869d;
    font-size:.57rem;
    font-weight:800;
    letter-spacing:.12em;
    text-transform:uppercase;
    margin-bottom:.42rem;
}
.metric-value{
    color:#f2f6fb;
    font-size:1.55rem;
    line-height:1.08;
    font-weight:800;
    letter-spacing:-.03em;
    overflow-wrap:anywhere;
}
.metric-sub{
    margin-top:.42rem;
    font-size:.63rem;
    color:#72869d;
}
.metric-sub.critical{color:var(--ta-red)}
.metric-sub.high{color:var(--ta-orange)}
.metric-sub.normal{color:#4fcfa0}

/* ---------- kill chain ---------- */
.stage-wrap{
    position:relative;
    padding-top:.25rem;
}
.stage-connector{
    position:absolute;
    left:4%;
    right:4%;
    top:1.38rem;
    height:1px;
    background:linear-gradient(90deg,rgba(61,156,255,.05),rgba(61,156,255,.28),rgba(61,156,255,.05));
    z-index:0;
}
.stage-card{
    position:relative;
    z-index:1;
    padding:.78rem .4rem .68rem;
    border-radius:9px;
    border:1px solid rgba(91,151,207,.13);
    text-align:center;
    min-height:78px;
    background:#0b1829;
    box-shadow:0 7px 20px rgba(0,0,0,.12);
}
.stage-index{
    width:20px;
    height:20px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    border-radius:50%;
    background:#13243a;
    border:1px solid rgba(91,151,207,.18);
    color:#71859b;
    font-size:.6rem;
    font-weight:800;
    margin-bottom:.42rem;
}
.stage-name{
    color:#d8e3ee;
    font-size:.67rem;
    font-weight:700;
}
.stage-active{
    box-shadow:0 0 0 1px rgba(255,83,100,.08),0 8px 24px rgba(255,83,100,.07);
}
.stage-active .stage-index{
    background:rgba(255,83,100,.13);
    border-color:rgba(255,83,100,.4);
    color:#ff7480;
}
.stage-forecast{
    box-shadow:0 0 0 1px rgba(243,166,74,.07),0 8px 24px rgba(243,166,74,.06);
}
.stage-forecast .stage-index{
    background:rgba(243,166,74,.12);
    border-color:rgba(243,166,74,.35);
    color:#f3a64a;
}
.stage-label{
    font-size:.55rem;
    letter-spacing:.1em;
    font-weight:800;
    color:#61758b;
}

/* ---------- detail / evidence cards ---------- */
.detail-grid{
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:.75rem;
    margin:.6rem 0 1rem;
}
.detail-card{
    padding:.9rem 1rem;
    border:1px solid rgba(91,151,207,.14);
    border-radius:9px;
    background:linear-gradient(145deg,#0c1a2d,#091423);
}
.detail-label{
    color:#71869d;
    font-size:.58rem;
    font-weight:800;
    letter-spacing:.1em;
    text-transform:uppercase;
}
.detail-value{
    margin-top:.35rem;
    color:#edf3f9;
    font-size:1.03rem;
    font-weight:740;
}
.detail-note{
    margin-top:.22rem;
    color:#6e8298;
    font-size:.63rem;
}
.signal-card{
    display:flex;
    gap:.7rem;
    align-items:flex-start;
    padding:.72rem .8rem;
    border:1px solid rgba(91,151,207,.11);
    border-radius:8px;
    background:rgba(11,27,45,.62);
    margin:.42rem 0;
}
.signal-index{
    flex:0 0 22px;
    width:22px;
    height:22px;
    border-radius:6px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    background:rgba(40,199,255,.08);
    border:1px solid rgba(40,199,255,.12);
    color:#5bcfff;
    font-size:.6rem;
    font-weight:800;
}
.signal-text{
    color:#b8c6d5;
    font-size:.74rem;
    line-height:1.45;
}

/* ---------- tabs ---------- */
div[data-baseweb="tab-list"]{
    gap:.15rem;
    border-bottom:1px solid rgba(91,151,207,.14);
    padding-bottom:0;
}
button[data-baseweb="tab"]{
    font-size:.72rem!important;
    font-weight:650!important;
    color:#71869d!important;
    padding:.56rem .72rem!important;
    border-radius:6px 6px 0 0!important;
    background:transparent!important;
}
button[data-baseweb="tab"]:hover{
    color:#dce8f3!important;
    background:rgba(61,156,255,.045)!important;
}
button[data-baseweb="tab"][aria-selected="true"]{
    color:#f3f7fb!important;
}
div[data-baseweb="tab-highlight"]{
    background:linear-gradient(90deg,var(--ta-cyan),var(--ta-blue))!important;
    height:2px!important;
}

/* ---------- metrics / data ---------- */
div[data-testid="stMetric"]{
    background:linear-gradient(145deg,#0d1d31,#091524)!important;
    border:1px solid rgba(91,151,207,.14)!important;
    border-radius:9px!important;
    padding:.78rem .9rem!important;
}
div[data-testid="stMetricLabel"]{color:#71869d!important}
div[data-testid="stMetricValue"]{font-size:1.3rem;font-weight:780;color:#edf3f9}
div[data-testid="stDataFrame"]{
    border:1px solid rgba(91,151,207,.13);
    border-radius:9px;
    overflow:hidden;
    background:#091423;
}
div[data-testid="stDataFrame"] *{font-size:.73rem}
hr{border-color:rgba(91,151,207,.11)!important}
.small{font-size:.66rem;line-height:1.5;color:#70849b}
.sev-CRITICAL{color:var(--ta-red);font-weight:750}
.sev-HIGH{color:var(--ta-orange);font-weight:750}
.sev-MEDIUM{color:var(--ta-yellow);font-weight:750}
.sev-OK{color:var(--ta-green);font-weight:750}

/* ---------- buttons / downloads / expanders ---------- */
button[kind="primary"],
.stDownloadButton button{
    background:linear-gradient(135deg,#1165b8,#168bd0)!important;
    border:1px solid rgba(86,194,255,.28)!important;
    color:white!important;
    border-radius:7px!important;
    font-weight:650!important;
}
button[kind="primary"]:hover,
.stDownloadButton button:hover{
    background:linear-gradient(135deg,#1674ca,#1b9be0)!important;
}
button[kind="secondary"]{
    background:#0d1c30!important;
    border:1px solid rgba(91,151,207,.16)!important;
    color:#cbd7e3!important;
}
div[data-testid="stExpander"]{
    border:1px solid rgba(91,151,207,.13)!important;
    border-radius:9px!important;
    background:rgba(9,20,35,.58)!important;
}
div[data-testid="stAlert"]{
    border-radius:9px!important;
}

/* ---------- chart containers ---------- */
div[data-testid="stPlotlyChart"]{
    border:1px solid rgba(91,151,207,.09);
    border-radius:10px;
    padding:.15rem;
    background:linear-gradient(145deg,rgba(10,25,43,.48),rgba(5,14,25,.22));
}

/* ---------- responsive ---------- */
@media(max-width:1100px){
    .forecast-hero{grid-template-columns:1fr}
    .system-strip{grid-template-columns:1fr 1fr}
    .investigation-grid,.alert-counts{grid-template-columns:1fr 1fr}
    .alert-row,.alert-table-head{grid-template-columns:1.2fr 1fr .7fr .7fr .7fr}
}
@media(max-width:900px){
    .context-strip,.metric-grid,.detail-grid{grid-template-columns:1fr 1fr}
    .hero-meta{display:none}
}
@media(max-width:600px){
    .context-strip,.metric-grid,.detail-grid{grid-template-columns:1fr}
    .context-item{border-right:0;border-bottom:1px solid rgba(91,151,207,.08)}
    .context-item:last-child{border-bottom:0}
}

/* ---------- modest readability bump ---------- */
html{font-size:106%!important}
body{font-size:1.02rem!important}
[data-testid="stSidebar"] *{font-size:1.015em}
[data-testid="stTabs"] button{font-size:1.04em!important}
[data-testid="stCaptionContainer"]{font-size:1.02rem!important}
.stMarkdown p, .stMarkdown li{font-size:1.02rem}

</style>
""", unsafe_allow_html=True)

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
    processed_path = resolve(cfg["paths"]["processed"])
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Held-out processed data is missing: {processed_path}. "
            "Use a bundled sample/upload, or build this dataset first."
        )
    df = pd.read_parquet(processed_path)
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

def infiltration_timeline(f, thr, upto):
    live = f.iloc[:upto + 1].copy()

    fig = go.Figure()

    # Risk area
    fig.add_trace(
        go.Scatter(
            x=live["time"],
            y=live["risk"],
            mode="lines",
            name="Infiltration probability",
            line=dict(color="#35b9ff", width=3),
            fill="tozeroy",
            fillcolor="rgba(53,185,255,0.10)",
        )
    )

    # Alert threshold
    fig.add_hline(
        y=thr,
        line=dict(dash="dash"),
        annotation_text=f"Alert threshold: {thr:.2f}",
    )

    # Alert points
    alerts = live[live["risk"] >= thr]

    if not alerts.empty:
        fig.add_trace(
            go.Scatter(
                x=alerts["time"],
                y=alerts["risk"],
                mode="markers",
                name="Flagged",
                marker=dict(
                    size=9,
                    symbol="triangle-up",
                ),
            )
        )

    # Current time
    fig.add_vline(
        x=f["time"].iloc[upto],
        line=dict(width=2, dash="dot"),
        annotation_text="NOW",
    )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(6,17,31,.35)",
        title=dict(text="", x=0),
        yaxis=dict(
            title="Probability",
            range=[0, 1.05],
            tickformat=".0%",
        ),
        xaxis_title="Time",
        margin=dict(l=25, r=25, t=20, b=30),
    )

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
    if risk < thr:
        return "OK", "NORMAL"
    return ("CRITICAL" if risk >= 0.9 else "HIGH" if risk >= 0.75 else "MEDIUM"), "EARLY WARNING"

def risk_level(risk: float):
    if risk >= 0.90:
        return "CRITICAL"
    if risk >= 0.75:
        return "HIGH"
    if risk >= 0.50:
        return "MEDIUM"
    return "LOW"


def risk_badge(risk: float):
    level = risk_level(risk)

    classes = {
        "CRITICAL": "sev-CRITICAL",
        "HIGH": "sev-HIGH",
        "MEDIUM": "sev-MEDIUM",
        "LOW": "sev-OK",
    }

    return (
        f'<span class="{classes[level]}" '
        f'style="font-size:1rem">{level}</span>'
    )


def render_header():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-row">
            <div>
                <div class="hero-title">ThreatAhead</div>
                <div class="hero-subtitle">Network Attack Forecasting & Threat Intelligence Console</div>
            </div>
            <div class="hero-meta">
                <div class="offline-badge">OFFLINE · LOCAL INFERENCE</div>
                <div style="margin-top:6px;">NTRO · SIH 26153 · No cloud APIs</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_context_strip(ds, pr, model_label, src, seg, pos, total):
    dataset_name = str(ds).upper()
    protocol_name = str(pr).replace("_", " ").upper()
    source_name = "HELD-OUT TEST" if src.startswith("Held-out") else "TELEMETRY REPLAY"
    st.markdown(f"""
    <div class="context-strip">
        <div class="context-item"><div class="context-label">Dataset</div><div class="context-value" title="{model_label}">{dataset_name} · {protocol_name}</div></div>
        <div class="context-item"><div class="context-label">Source</div><div class="context-value">{source_name}</div></div>
        <div class="context-item"><div class="context-label">Segment</div><div class="context-value">{seg}</div></div>
        <div class="context-item"><div class="context-label">Cursor</div><div class="context-value">{pos + 1} / {total}</div></div>
    </div>
    """, unsafe_allow_html=True)


def render_stage_pipeline(current_stage, forecast_stage):
    st.markdown('<div class="section-kicker">Kill-chain state</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Attack Progression</div>', unsafe_allow_html=True)
    st.markdown('<div class="stage-wrap"><div class="stage-connector"></div>', unsafe_allow_html=True)
    cols = st.columns(len(STAGES))
    for i, stage in enumerate(STAGES):
        if stage == current_stage:
            border, background, status, extra = "rgba(255,91,101,.48)", "rgba(255,91,101,.065)", "CURRENT", "stage-active"
        elif stage == forecast_stage:
            border, background, status, extra = "rgba(240,163,74,.48)", "rgba(240,163,74,.055)", "FORECAST", "stage-forecast"
        else:
            border, background, status, extra = "rgba(148,163,184,.14)", "#101721", "", ""
        cols[i].markdown(f"""
        <div class="stage-card {extra}" style="border-color:{border};background:{background};">
            <div class="stage-index">{i+1}</div><div class="stage-name">{stage}</div>
            <div class="stage-label" style="margin-top:6px;">{status}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_overview_metrics(frame, row, stats):
    max_risk = float(frame["risk"].max())
    host_count = frame["host"].nunique() if "host" in frame.columns else frame["segment"].nunique()
    flow_count = stats.get("flows", 0)
    level = risk_level(max_risk)
    accent = {"CRITICAL":"#ff5b65","HIGH":"#f0a34a","MEDIUM":"#e5c45d","LOW":"#55c98a"}.get(level,"#4da3ff")
    cards = [("Maximum Risk",f"{max_risk:.0%}",level,accent),("Hosts Monitored",f"{host_count:,}","Active host-slots","#4da3ff"),("Flows Analyzed",f"{flow_count:,}" if flow_count else "—","Observed traffic","#55c98a"),("Current Stage",str(row["current_stage_pred"]),"Model estimate","#f0a34a")]
    html=['<div class="metric-grid">']
    for label,value,sub,color in cards:
        cls="critical" if sub=="CRITICAL" else "high" if sub=="HIGH" else "normal"
        html.append(f'<div class="metric-card" style="--metric-accent:{color};"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub {cls}">{sub}</div></div>')
    html.append('</div>')
    st.markdown("".join(html),unsafe_allow_html=True)


def render_investigation(row, WMIN, thr):
    st.markdown('<div class="section-kicker">Current cursor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Investigation Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">Current model state, forecast stage and time-to-exploitation for the selected monitoring segment.</div>', unsafe_allow_html=True)
    risk=float(row["risk"]); forecast=row["forecast_stage"] if risk>=thr else "No immediate escalation"; eta=row.get("eta_windows",0); eta_text=f"{eta*WMIN:.1f} minutes" if eta>0 and risk>=thr else "—"; level=risk_level(risk)
    st.markdown(f"""
    <div class="detail-grid">
      <div class="detail-card"><div class="detail-label">Forecast Risk</div><div class="detail-value">{risk:.0%}</div><div class="detail-note">{level} severity</div></div>
      <div class="detail-card"><div class="detail-label">Current Stage</div><div class="detail-value">{row['current_stage_pred']}</div><div class="detail-note">Observed model state</div></div>
      <div class="detail-card"><div class="detail-label">Forecast Stage</div><div class="detail-value">{forecast}</div><div class="detail-note">Based on current risk threshold</div></div>
    </div>""",unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        st.markdown("**Current state**")
        st.markdown(f'<span class="muted-note">Risk level</span><br>{risk_badge(risk)}',unsafe_allow_html=True)
        st.markdown(f'<span class="muted-note">Observed stage</span><br><b>{row["current_stage_pred"]}</b>',unsafe_allow_html=True)
    with right:
        st.markdown("**Forecast**")
        st.markdown(f'<span class="muted-note">ETA to exploitation</span><br><b>{eta_text}</b>',unsafe_allow_html=True)
        st.markdown(f'<span class="muted-note">Behavioural surprise</span><br><b>{row["surprise_z"]:.2f}</b>',unsafe_allow_html=True)

def render_flagged_traffic(frame, thr, min_consec, W, key_suffix="default"):
    """Render the traffic windows currently flagged by forecast risk/anomaly."""
    st.markdown("### Flagged Traffic")

    work = frame.copy()

    if work.empty:
        st.caption("No traffic windows are available for the current selection.")
        return

    # Build a consecutive-risk flag per monitored segment.
    risk_flag = work["risk"].ge(thr)
    if min_consec > 1:
        grp = work.groupby("segment")["risk"].transform(
            lambda s: s.ge(thr).astype(int).rolling(
                min_consec, min_periods=min_consec
            ).sum().ge(min_consec)
        )
        work["risk_flag"] = grp.fillna(False).astype(bool)
    else:
        work["risk_flag"] = risk_flag

    anomaly_flag = (
        work["surprise_z"].ge(
            work.get("surprise_z", pd.Series(index=work.index, dtype=float)).median()
        )
        if "surprise_z" in work.columns
        else pd.Series(False, index=work.index)
    )

    # Prefer the actual alert/anomaly columns already computed by the app.
    if "alert" in work.columns:
        work["risk_flag"] = work["alert"].astype(bool) | work["risk_flag"]
    if "anomaly" in work.columns:
        work["anomaly_flag"] = work["anomaly"].astype(bool)
    else:
        work["anomaly_flag"] = anomaly_flag

    flagged = work[work["risk_flag"] | work["anomaly_flag"]].copy()

    if flagged.empty:
        st.info("No traffic windows currently exceed the configured risk or anomaly thresholds.")
        return

    # Human-readable channel label. Keep it on the full working frame
    # because the chart below uses a filtered view called ``marked``.
    work["channel"] = np.select(
        [
            work["risk_flag"] & work["anomaly_flag"],
            work["risk_flag"],
            work["anomaly_flag"],
        ],
        ["risk + anomaly", "risk", "anomaly"],
        default="-",
    )

    flagged = work[work["risk_flag"] | work["anomaly_flag"]].copy()

    preferred = [
        "time", "segment", "capture", "host", "channel",
        "risk", "surprise_z", "current_stage_pred", "forecast_stage"
    ]
    cols = [c for c in preferred if c in flagged.columns]

    display = flagged[cols].copy()

    if "time" in display.columns:
        display["time"] = pd.to_datetime(display["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    for c in ("risk", "surprise_z"):
        if c in display.columns:
            display[c] = display[c].round(3)

    st.caption(
        f"{len(flagged):,} flagged window(s) · "
        f"risk threshold {thr:.2f} · debounce {min_consec} window(s)"
    )
    st.dataframe(display, hide_index=True, width="stretch", height=360)

    # Compact risk timeline for the flagged windows.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["time"],
            y=work["risk"],
            mode="lines",
            name="Forecast risk",
            line=dict(width=2),
        )
    )
    fig.add_hline(
        y=thr,
        line=dict(dash="dash"),
        annotation_text=f"threshold {thr:.2f}",
    )

    marked = work[work["risk_flag"] | work["anomaly_flag"]]
    fig.add_trace(
        go.Scatter(
            x=marked["time"],
            y=marked["risk"],
            mode="markers",
            name="Flagged",
            marker=dict(size=7, symbol="triangle-up"),
            customdata=marked["channel"],
            hovertemplate="%{x}<br>risk=%{y:.2f}<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="Time",
        yaxis_title="Risk",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch", key=f"flagged_traffic_timeline_{key_suffix}")


def render_signal_summary(row, zthr):
    st.markdown('<div class="section-kicker">Model evidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Why was this traffic flagged?</div>', unsafe_allow_html=True)
    signals=[]
    if row["risk"]>=0.75: signals.append("Forecast risk crossed the high-risk threshold.")
    if row["surprise_z"]>=zthr: signals.append("Traffic behaviour differs significantly from the learned baseline.")
    if row.get("risk_rollout",0)>=0.75: signals.append("Forward simulation indicates elevated future exploitation probability.")
    if row.get("risk_markov",0)>=0.75: signals.append("The learned attack-stage transition model assigns elevated probability to escalation.")
    if not signals: signals.append("No strong forecast signal is currently above the configured thresholds.")
    st.markdown("".join(f'<div class="signal-card"><div class="signal-index">{i:02d}</div><div class="signal-text">{signal}</div></div>' for i,signal in enumerate(signals,1)),unsafe_allow_html=True)


def render_system_status(ds, pr, stats, bundle, frame, src):
    source_label = "HELD-OUT TEST" if src.startswith("Held-out") else "TELEMETRY READY"
    flow_count = stats.get("flows", 0)
    sec = stats.get("sec", 0)
    st.markdown(f"""
    <div class="system-strip">
        <div class="system-item"><div class="system-top"><span class="system-dot"></span>Model</div><div class="system-value">READY · {str(ds).upper()}</div><div class="system-meta">{str(pr).replace('_',' ').title()} protocol</div></div>
        <div class="system-item"><div class="system-top"><span class="system-dot"></span>Data</div><div class="system-value">{source_label}</div><div class="system-meta">{len(frame):,} forecast windows</div></div>
        <div class="system-item"><div class="system-top"><span class="system-dot"></span>Inference</div><div class="system-value">LOCAL · {sec:.2f}s</div><div class="system-meta">No cloud processing</div></div>
        <div class="system-item"><div class="system-top"><span class="system-dot"></span>Telemetry</div><div class="system-value">{flow_count:,} flows</div><div class="system-meta">Window {float(bundle.config['features']['window_seconds']):.0f}s</div></div>
    </div>
    """, unsafe_allow_html=True)


def render_threat_summary(row, frame, thr, WMIN):
    risk = float(row["risk"])
    level = risk_level(risk)
    forecast = str(row.get("forecast_stage", "—")) if risk >= thr else "No immediate escalation"
    eta = row.get("eta_windows", 0)
    eta_text = f"{float(eta) * WMIN:.1f} min" if eta and risk >= thr else "—"
    hosts = frame["host"].nunique() if "host" in frame.columns else frame["segment"].nunique()
    st.markdown(f"""
    <div class="threat-summary-card">
        <div class="summary-kicker">Threat summary</div><div class="summary-title">Current forecast state</div>
        <div class="summary-risk summary-risk-{level.lower()}">{risk:.0%}</div>
        <div class="summary-meter"><div class="summary-meter-fill" style="width:{max(0,min(100,risk*100)):.1f}%"></div></div>
        <div class="summary-row"><span class="summary-label">Severity</span><span class="summary-value">{level}</span></div>
        <div class="summary-row"><span class="summary-label">Current stage</span><span class="summary-value">{row['current_stage_pred']}</span></div>
        <div class="summary-row"><span class="summary-label">Forecast</span><span class="summary-value">{forecast}</span></div>
        <div class="summary-row"><span class="summary-label">Hosts affected</span><span class="summary-value">{hosts:,}</span></div>
        <div class="summary-row"><span class="summary-label">Threshold</span><span class="summary-value">{thr:.2f}</span></div>
        <div class="summary-row"><span class="summary-label">ETA</span><span class="summary-value">{eta_text}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_alert_console(alerts, segs, per_host):
    total = len(alerts)
    sev = alerts["severity"].astype(str).str.upper() if "severity" in alerts.columns else pd.Series([], dtype=str)
    critical = int((sev == "CRITICAL").sum())
    high = int((sev == "HIGH").sum())
    medium = int((sev == "MEDIUM").sum())
    st.markdown(f"""
    <div class="alert-console-head"><div><div class="section-kicker">Detection queue</div><div class="alert-console-title">Active Alerts</div><div class="alert-console-sub">{total} forecast alert(s) across {len(segs)} monitored {'host-slots' if per_host else 'segments'}.</div></div></div>
    <div class="alert-counts">
        <div class="alert-count total"><div class="alert-count-label">Total alerts</div><div class="alert-count-value">{total}</div></div>
        <div class="alert-count critical"><div class="alert-count-label">Critical</div><div class="alert-count-value">{critical}</div></div>
        <div class="alert-count high"><div class="alert-count-label">High</div><div class="alert-count-value">{high}</div></div>
        <div class="alert-count medium"><div class="alert-count-label">Medium</div><div class="alert-count-value">{medium}</div></div>
    </div>
    """, unsafe_allow_html=True)
    if not total:
        st.markdown('<div class="investigation-panel"><div class="investigation-panel-title">No active alerts</div><div class="investigation-panel-copy">No monitored window currently crosses the configured forecast threshold.</div></div>', unsafe_allow_html=True)
        return
    preferred=["alert_id","segment","start","end","duration","peak_risk","severity","forecast_stage","eta_min"]
    display=alerts[[c for c in preferred if c in alerts.columns]].copy()
    st.markdown('<div class="alert-table-head"><div>Alert</div><div>Segment</div><div>Severity</div><div>Risk</div><div>ETA</div></div>',unsafe_allow_html=True)
    for _,r in display.head(20).iterrows():
        aid=str(r.get("alert_id","Alert")); seg=str(r.get("segment","—")); severity=str(r.get("severity","—")).upper(); sev_cls=severity.lower() if severity.lower() in {"critical","high","medium","low"} else "low"
        rv=r.get("peak_risk",r.get("risk",None)); risk_text=f"{float(rv):.0%}" if pd.notna(rv) and isinstance(rv,(int,float,np.integer,np.floating)) else "—"
        ev=r.get("eta_min",None); eta_text=f"{float(ev):.1f} min" if pd.notna(ev) and isinstance(ev,(int,float,np.integer,np.floating)) else "—"
        stage=str(r.get("forecast_stage","Forecast alert"))
        st.markdown(f"""
        <div class="alert-row"><div><div class="alert-primary">{aid}</div><div class="alert-secondary">{stage}</div></div><div><div class="alert-primary">Segment {seg}</div><div class="alert-secondary">Forecast window</div></div><div class="alert-severity {sev_cls}">{severity}</div><div><div class="alert-primary">{risk_text}</div><div class="alert-secondary">Peak forecast risk</div></div><div><div class="alert-primary">{eta_text}</div><div class="alert-secondary">Time to escalation</div></div></div>
        """,unsafe_allow_html=True)
    with st.expander("View alert data table"):
        st.dataframe(display,hide_index=True,width="stretch")

# ----------------------------------------------------------------------------- sidebar
models = available_models()
if not models:
    st.error("No trained bundle found. Run `python -m sih_v2 build_dataset --dataset ctu13` then "
             "`python -m sih_v2 train --dataset ctu13` (see docs/README.md).")
    st.stop()
with st.sidebar:
    st.markdown("""<div class="sidebar-brand">
<div class="sidebar-brand-title">ThreatAhead</div>
<div class="sidebar-brand-sub">SOC COMMAND CENTER · NTRO · SIH 26153</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>Detection Model</div>", unsafe_allow_html=True)
    model_label = st.selectbox(
        "Model",
        list(models),
        index=list(models).index("Synthetic campaigns · controlled · network-wide")
    )
    ds, pr = models[model_label]
    cfg, bundle, pack = get_bundle(ds, pr)
    st.markdown("<div class='sidebar-section' style='margin-top:.9rem;'>Data Source</div>", unsafe_allow_html=True)

    # A trained model bundle does not imply that its pre-processed held-out
    # dataset is present locally. Keep the model selectable, but only expose
    # the held-out source when the required parquet exists.
    processed_path = resolve(cfg["paths"]["processed"])
    has_heldout = processed_path.exists()

    if has_heldout:
        source_options = ["Held-out test split (labelled)", *SAMPLES, "Upload file"]
        source_index = 0
    else:
        source_options = [*SAMPLES, "Upload file"]
        source_index = 0
        st.warning(
            "Held-out test data is not installed for this model. "
            "Use a bundled sample or upload telemetry."
        )

    src = st.radio("Telemetry source", source_options, index=source_index)
    up = None
    if src == "Upload file":
        up = st.file_uploader("CSV (CIC-IDS2017/2018, CTU-13 binetflow, UNSW, NetFlow export), Parquet, PCAP/PCAPNG",
                              type=["csv", "gz", "parquet", "binetflow", "pcap", "pcapng", "cap"])
    st.markdown("---")
    st.markdown("<div class='sidebar-section'>Forward Simulation</div>", unsafe_allow_html=True)
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
if src == "Held-out test split (labelled)":
    # Defensive check in case the file disappears after the sidebar renders.
    processed_path = resolve(cfg["paths"]["processed"])
    if not processed_path.exists():
        st.error(
            f"Held-out test data is unavailable for `{ds}` / `{pr}`. "
            f"Expected: `{processed_path}`. Select a bundled sample or upload telemetry."
        )
        st.stop()
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
render_header()

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


st.markdown(
    f"<div class='small'>ANALYSIS COMPLETE &nbsp;·&nbsp; "
    f"<b>{stats.get('source', 'network traffic')}</b> &nbsp;·&nbsp; "
    f"{stats.get('sec', 0):.2f}s local processing</div>",
    unsafe_allow_html=True,
)

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
    flag = " · flagged" if (len(r) and (r["by_risk"].any() or r["by_anomaly"].any())) else ""
    return f"{who} · {g['time'].iloc[0]:%Y-%m-%d %H:%M}{tag}{flag}"


order = list(tri_tab["segment"])
order += [s for s in segs if s not in order]
default_seg = order[0]
if labelled:
    inf = frame[frame["stage"] > 0].groupby("segment").size()
    pre = frame[frame["phase"] == "pre_attack"].groupby("segment").size()
    default_seg = (pre.idxmax() if len(pre) else inf.idxmax()) if len(inf) else order[0]
st.markdown("<div class='section-kicker' style='margin-top:.2rem;'>Live investigation</div>", unsafe_allow_html=True)
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
render_context_strip(ds, pr, model_label, src, seg, pos, len(f))
render_system_status(ds, pr, stats, bundle, frame, src)

tabs = st.tabs([
    "Overview",
    "Investigation",
    "Attack Forecast",
    "MITRE ATT&CK",
    "Explainability",
    "Evaluation",
    "Alerts",
])

# ----------------------------------------------------------------------------- live monitor
# ----------------------------------------------------------------------------- overview
with tabs[0]:

    risk = float(row["risk"])

    st.markdown('<div class="section-kicker">Network monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.45rem;">Network Attack Forecast</div>', unsafe_allow_html=True)

    if risk >= thr:
        st.markdown(
            f"""
            <div class="threat-banner">
                <div>
                    <div class="threat-title">ELEVATED THREAT</div>
                    <div class="threat-copy">Attack progression detected · Forecast stage: <b>{row['forecast_stage']}</b></div>
                </div>
                <div class="threat-score">{risk:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="threat-banner normal">
                <div>
                    <div class="threat-title">NO IMMEDIATE THREAT</div>
                    <div class="threat-copy">No high-risk attack progression detected at the current cursor.</div>
                </div>
                <div class="threat-score">{risk:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main metrics
    render_overview_metrics(
        frame,
        row,
        stats,
    )

    st.divider()

    # Forecast hero: the primary decision surface for the dashboard.
    st.markdown('<div class="section-kicker">Primary forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Attack Probability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">Observed risk through the current cursor, calibrated alert threshold, and flagged forecast windows.</div>', unsafe_allow_html=True)
    chart_col, summary_col = st.columns([3.45, 1])
    with chart_col:
        st.markdown("<div class='forecast-chart-card'><div class='forecast-chart-head'><div class='forecast-chart-kicker'>Forecast timeline</div><div class='forecast-chart-title'>Probability of exploitation</div><div class='forecast-chart-sub'>The model's fused risk estimate across the monitored segment.</div></div>", unsafe_allow_html=True)
        st.plotly_chart(infiltration_timeline(f, thr, pos), width="stretch", key="forecast_hero_chart")
        st.markdown('</div>', unsafe_allow_html=True)
    with summary_col:
        render_threat_summary(row, frame, thr, WMIN)

    st.divider()

    # Attack progression
    forecast_stage = (
        row["forecast_stage"]
        if risk >= thr
        else None
    )

    render_stage_pipeline(
        row["current_stage_pred"],
        forecast_stage,
    )

    st.divider()

    # Investigation summary
    render_investigation(
        row,
        WMIN,
        thr,
    )

    st.divider()

    # Why flagged
    render_signal_summary(
        row,
        zthr,
    )

    st.divider()

    # Flagged traffic
    render_flagged_traffic(
        frame,
        thr,
        min_consec,
        W,
        key_suffix="overview",
    )
# ----------------------------------------------------------------------------- host triage
with tabs[1]:
    st.markdown('<div class="section-kicker">Host-level triage</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.35rem;">Investigation Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">Correlate supervised forecast risk with label-free behavioural surprise to identify hosts that require investigation.</div>', unsafe_allow_html=True)
    m = st.columns(4)
    if labelled:
        m[0].metric("Infected host-slots caught", f"{tri['caught']}/{tri['infected']}")
        m[1].metric("Anomaly-only detections", tri["caught_only_by_anomaly"])
        m[2].metric("Benign host-slots flagged", f"{tri['false_alarms']}/{tri['benign']}", f"{tri['false_alarm_rate']:.1%}", delta_color="off")
        m[3].metric("Risk-only false alarms", tri["risk_only_false_alarms"])
    else:
        flagged_risk = int(tri_tab["by_risk"].sum()) if "by_risk" in tri_tab else 0
        flagged_anom = int(tri_tab["by_anomaly"].sum()) if "by_anomaly" in tri_tab else 0
        m[0].metric("Monitored host-slots", len(tri_tab)); m[1].metric("Risk detections", flagged_risk); m[2].metric("Anomaly detections", flagged_anom); m[3].metric("Threshold", f"{thr:.2f}")
    show = tri_tab.assign(channel=np.where(tri_tab["by_risk"] & tri_tab["by_anomaly"], "risk + anomaly", np.where(tri_tab["by_risk"], "risk", np.where(tri_tab["by_anomaly"], "anomaly", "-"))))
    cols = ["capture", "host", "channel", "max_risk", "surprise_z", "cells"] + (["infected"] if labelled else [])
    st.markdown('<div class="investigation-panel"><div class="investigation-panel-title">Host triage</div><div class="investigation-panel-copy">Risk is the calibrated supervised forecast signal. Anomaly is the robust surprise signal relative to the capture baseline. Keep them separate so analysts can see where the channels agree or diverge.</div></div>', unsafe_allow_html=True)
    st.dataframe(show[cols].round(3), hide_index=True, width="stretch", height=390)
    left, right = st.columns([1.55, 1])
    with left:
        sc = go.Figure()
        sc.add_trace(go.Scatter(x=tri_tab["surprise_z"].clip(-5, 30), y=tri_tab["max_risk"], mode="markers", marker=dict(size=9, color=np.where(tri_tab["infected"], "#ff5364", "#3d9cff") if labelled else "#3d9cff", line=dict(width=1, color="#071321")), text=tri_tab["host"].astype(str) if "host" in tri_tab else tri_tab["segment"].astype(str), hovertemplate="%{text}<br>surprise=%{x:.1f}<br>risk=%{y:.2f}<extra></extra>"))
        sc.add_hline(y=thr, line=dict(color="#e5c65a", dash="dash")); sc.add_vline(x=zthr, line=dict(color="#28c7ff", dash="dot"))
        sc.update_layout(height=430, template="plotly_dark", xaxis_title="Behavioural surprise (robust z)", yaxis_title="Maximum forecast risk", title="Risk vs behavioural surprise", margin=dict(l=10,r=10,t=45,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,17,31,.35)")
        st.plotly_chart(sc, width="stretch", key="investigation_risk_scatter")
    with right:
        st.markdown('<div class="investigation-panel"><div class="investigation-panel-title">Investigation guidance</div><div class="investigation-panel-copy">Upper-right hosts are high-priority because forecast risk and behavioural surprise are both elevated. High surprise with lower risk indicates an emerging anomaly that may precede a forecast alert.</div></div>', unsafe_allow_html=True)
        top_risk=float(tri_tab["max_risk"].max()) if len(tri_tab) else 0; top_z=float(tri_tab["surprise_z"].max()) if len(tri_tab) else 0
        st.markdown(f'<div class="investigation-grid" style="grid-template-columns:1fr 1fr;"><div class="investigation-card"><div class="investigation-label">Peak risk</div><div class="investigation-value">{top_risk:.0%}</div><div class="investigation-note">Across monitored host-slots</div></div><div class="investigation-card"><div class="investigation-label">Peak surprise</div><div class="investigation-value">{top_z:.1f}σ</div><div class="investigation-note">Robust deviation</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker" style="margin-top:1rem;">Flagged traffic</div>', unsafe_allow_html=True)
    render_flagged_traffic(frame, thr, min_consec, W, key_suffix="investigation")

# ----------------------------------------------------------------------------- forward simulation
with tabs[2]:

    st.markdown('<div class="section-kicker">Forward simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.35rem;">Attack Forecast</div>', unsafe_allow_html=True)

    st.caption(
        f"Simulating the next "
        f"{res.rollout_mean.shape[1]} windows "
        f"({res.rollout_mean.shape[1] * WMIN:.0f} minutes) "
        "without observing future traffic."
    )
    st.caption("The posterior belief is filtered over the last L windows; P particles are drawn from it and the learned "
               "latent dynamics p(z<sub>t+1</sub>|h<sub>t+1</sub>) are rolled forward K steps without observations. The "
               "stage head is read at every imagined step.", unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("Direct head", f"{row['risk_direct']:.2f}")
    m[1].metric("Rollout (particle mean of max-step)", f"{row['risk_rollout']:.2f}")
    m[2].metric("Markov kill-chain prior", f"{row['risk_markov']:.2f}")
    m[3].metric("Particles reaching exploitation", f"{row['particle_hit_frac']:.0%}")
    st.plotly_chart(rollout_fig(res, gi, WMIN), width="stretch", key="attack_forecast_rollout")
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
    st.markdown('<div class="section-kicker">Technique mapping</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.35rem;">MITRE ATT&CK</div>', unsafe_allow_html=True)
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
    cols[0].plotly_chart(fig, width="stretch", key="mitre_killchain_position")
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
    c3.plotly_chart(hm, width="stretch", key="mitre_transition_heatmap")
    strip = res.rollout_mean[gidx][:, :, EXPLOIT_STAGES].max(1)
    sf = go.Figure(go.Heatmap(z=strip.T, x=f["time"], y=[STAGES[i] for i in EXPLOIT_STAGES], colorscale="Inferno",
                              zmin=0, zmax=1))
    sf.update_layout(height=420, template="plotly_dark", title="Forecast tactic heat-strip across the host-slot",
                     margin=dict(l=10, r=10, t=40, b=10))
    c4.plotly_chart(sf, width="stretch", key="mitre_forecast_heatstrip")
    st.caption("CTU-13 mapping (documented in constants.py): spam / click-fraud → Impact (T1496 resource hijacking); "
               "DNS lookups, connection attempts, ICMP → Reconnaissance; CC / IRC / P2P / custom-encrypted → C2; "
               "malicious internal SMB/RPC/RDP/SSH → Lateral Movement (behavioural rule).")

# ----------------------------------------------------------------------------- explainability
with tabs[4]:
    st.markdown('<div class="section-kicker">Model transparency</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-heading" style="font-size:1.35rem;">Why is the risk {row["risk"]:.2f} at {row["time"]:%H:%M}?</div>', unsafe_allow_html=True)
    L = int(bundle.config["forecast"]["context"])
    ci = context_index(frame["segment"].to_numpy(), L)
    ca, cb = st.columns(2)
    att = res.attention[gi]
    ago = -(np.arange(L)[::-1]) * WMIN
    af = go.Figure(go.Bar(x=ago, y=att, marker_color="#ab47bc"))
    af.update_layout(height=330, template="plotly_dark", title="Temporal attention: which past windows drove the belief",
                     xaxis_title="minutes relative to now", yaxis_title="attention weight", margin=dict(l=10, r=10, t=40, b=10))
    ca.plotly_chart(af, width="stretch", key="explainability_attention")

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
    cb.plotly_chart(pt, width="stretch", key="explainability_temporal_attribution")
    top = ex["features"].head(15).iloc[::-1]
    sfig = go.Figure(go.Bar(y=top["feature"], x=top["attribution"], orientation="h",
                            marker_color=np.where(top["attribution"] > 0, "#ff5252", "#42a5f5")))
    sfig.update_layout(height=480, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                       title=f"Top drivers of forecast risk (base value {ex['base_value']:.2f} → {ex['risk']:.2f})",
                       xaxis_title="contribution to risk (red raises, blue lowers)")
    c5, c6 = st.columns([3, 2])
    c5.plotly_chart(sfig, width="stretch", key="explainability_feature_drivers")
    with c6:
        st.markdown("**Breakdown by temporal scale** (instant · burst vs 3-window EMA · 15-window trend)")
        st.dataframe(ex["features"].head(12).round(4), hide_index=True, width="stretch", height=300)
        if pack is not None and pack.get("sklearn"):
            lr = pack["sklearn"][0]["model"]
            ls = lr.linear_shap(res.X[gi])[0].reshape(3, -1).sum(0)
            # names come from the bundle so a model trained on an earlier feature set still lists its own
            lsd = pd.DataFrame({"feature": bundle.feature_names[:len(ls)], "LR SHAP (logit)": ls})
            lsd = lsd.reindex(lsd["LR SHAP (logit)"].abs().sort_values(ascending=False).index).head(8)
            st.markdown("**Baseline comparison - exact linear SHAP of single-window logistic regression**")
            st.dataframe(lsd.round(3), hide_index=True, width="stretch")
    with st.expander("Feature glossary"):
        st.markdown("""
- **syn/ack/fin/rst/psh/urg_ratio** - TCP flag counts per packet (CTU-13: recovered from Argus `State` strings); **flag_bitmask_entropy** - diversity of per-flow flag combinations.
- **ttl_mean / ttl_std** - hop-count / OS fingerprint spread (PCAP input; absent in CICFlowMeter and Argus exports).
- **init_win_mean/std** - initial TCP window; **log_iat_mean/std/max** - inter-arrival timing; **beacon_score** - periodicity of repeated conversations (C2).
- **log_payload_mean / payload_std / payload_small_frac** - per-packet payload size distribution (PCAP input).
- **frag_frac / df_frac** - share of packets carrying IPv4 fragments (MF) or the don't-fragment bit (PCAP input only).
- **scan_step_mean / scan_seq_frac** - shape of the port sweep: small ordered steps = sequential scan, large jumps = randomised scan.
- Cells built from flow exports (Argus/CTU-13 binetflow, CICFlowMeter) cannot carry the TTL, IP-flag or payload attributes; those read zero and the cell records the absence in its `ttl_missing` / `payload_missing` / `frag_missing` flag instead of inventing a value.
- **dst_port_entropy, n_dst_ports, n_dst_ips** - breadth of scanning / spam fan-out.
- **lateral_frac** - internal-to-internal flows; **sensitive_port_frac** - SSH/SMB/RDP/WinRM/DB ports; **log_out_in_ratio** - upload asymmetry.
- Suffix **__dev_short** = deviation from a 3-window EMA (bursts); **__trend_long** = 15-window EMA (slow ramps).
""")

# ----------------------------------------------------------------------------- benchmark & integrity
with tabs[5]:
    st.markdown('<div class="section-kicker">Model benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.35rem;">Evaluation</div>', unsafe_allow_html=True)
    mp = resolve(cfg["paths"]["reports_dir"]) / "metrics.json"
    if not mp.exists():
        st.warning(f"Run `python -m sih_v2 evaluate --dataset {ds} --protocol {pr}` to produce the report.")
    else:
        rep = json.loads(mp.read_text(encoding="utf-8"))
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
            col.plotly_chart(fg, width="stretch", key=f"evaluation_{key.lower().replace("-", "_")}")
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
                col.plotly_chart(fg, width="stretch", key=f"evaluation_horizon_{key}")
        st.markdown("#### Evaluation-integrity checks")
        for c in rep.get("checks", []):
            icon = "PASS" if c["verdict"].startswith(("OK", "N/A", "reported")) else "CHECK"
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
                st.markdown(comp.read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------- alerts
with tabs[6]:
    alerts = generate_alerts(frame, thr, int(min_consec), W)
    if per_host and len(alerts):
        hosts = frame.groupby("segment")[["capture", "host"]].first()
        alerts = alerts.join(hosts, on="segment")
    render_alert_console(alerts, segs, per_host)
    if len(alerts):
        st.download_button("Download alerts CSV", alerts.to_csv(index=False).encode(), "alerts.csv", "text/csv")
    st.download_button("Download host triage CSV", tri_tab.to_csv(index=False).encode(), "host_triage.csv", "text/csv")
    exp = frame.drop(columns=[c for c in frame.columns if c.startswith("split")])
    st.download_button("Download per-cell forecasts CSV", exp.to_csv(index=False).encode(), "forecasts.csv", "text/csv")
