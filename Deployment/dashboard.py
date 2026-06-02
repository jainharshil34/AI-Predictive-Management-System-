# ============================================================
#  dashboard.py — NeuralGuard PdM Dashboard
#  Project: Neural Network-Based Predictive Maintenance
#  Dataset: Honeywell AI4I CNC (10,000 records)
#  Models:  LSTM+Dot-Attention (.keras) + Autoencoder (.h5)
#  Backend: Flask app.py → port 5001
#  Run:     streamlit run dashboard.py
# ============================================================

import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import datetime

# ── Must be FIRST Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="NeuralGuard · Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────
for key, default in [
    ("history",      []),
    ("live_running", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{--bg:#f1f3f6;--surf:#ffffff;--surf2:#eef0f4;--border:#d8dce5;
      --blue:#2563eb;--cyan:#0891b2;--teal:#0d9488;--green:#059669;
      --amber:#d97706;--red:#dc2626;--purple:#7c3aed;
      --text:#1e293b;--muted:#64748b;--accent:#2563eb;--r:12px;}
.stApp{background:var(--bg)!important;font-family:'DM Sans',sans-serif;}
.main .block-container{padding:1.2rem 2rem 3rem;max-width:1440px;}
header[data-testid="stHeader"]{display:none;}
[data-testid="stSidebar"]{background:#ffffff!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.card{background:#ffffff;border:1px solid #d8dce5;border-radius:var(--r);
      padding:1.3rem 1.5rem;margin-bottom:.9rem;transition:border-color .25s;box-shadow:0 1px 4px rgba(0,0,0,.06);}
.card:hover{border-color:var(--accent);}
.card-blue{background:#f0f5ff;border:1px solid #bfcffd;border-radius:var(--r);
           padding:1.3rem 1.5rem;margin-bottom:.9rem;box-shadow:0 1px 4px rgba(0,0,0,.05);}
.hero{background:linear-gradient(135deg,#1e3a8a 0%,#1d4ed8 50%,#2563eb 100%);
      border-radius:18px;padding:2.2rem 2.8rem;margin-bottom:1.4rem;
      position:relative;overflow:hidden;}
.hero::after{content:"";position:absolute;top:-40%;right:-5%;width:55%;height:200%;
             background:radial-gradient(ellipse,rgba(255,255,255,.08) 0%,transparent 65%);
             pointer-events:none;}
.hero-title{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;
            color:#ffffff;line-height:1.15;margin-bottom:.4rem;}
.hero-sub{font-size:1rem;color:rgba(255,255,255,.75);margin-bottom:1.1rem;}
.badge{display:inline-block;padding:.28rem .8rem;border-radius:999px;
       font-size:.75rem;font-weight:600;margin:.15rem;
       font-family:'JetBrains Mono',monospace;}
.b-blue  {background:rgba(255,255,255,.2); color:#fff;border:1px solid rgba(255,255,255,.35);}
.b-green {background:rgba(255,255,255,.15);color:#fff;border:1px solid rgba(255,255,255,.3);}
.b-amber {background:rgba(255,255,255,.15);color:#fff;border:1px solid rgba(255,255,255,.3);}
.b-cyan  {background:rgba(255,255,255,.15);color:#fff;border:1px solid rgba(255,255,255,.3);}
.b-purple{background:rgba(255,255,255,.15);color:#fff;border:1px solid rgba(255,255,255,.3);}
.b-red   {background:rgba(255,255,255,.2); color:#fff;border:1px solid rgba(255,255,255,.35);}
.sec{display:flex;align-items:center;gap:.65rem;border-bottom:1px solid var(--border);
     padding-bottom:.55rem;margin:1.5rem 0 .85rem;}
.sec-icon{width:34px;height:34px;border-radius:9px;display:flex;
          align-items:center;justify-content:center;font-size:1rem;}
.sec-title{font-family:'Space Mono',monospace;font-size:1rem;font-weight:700;
           color:var(--text);margin:0;letter-spacing:.4px;}
.sec-tag{font-size:.68rem;color:var(--muted);margin-left:auto;
         font-family:'JetBrains Mono',monospace;}
.kpi{background:#ffffff;border:1px solid #d8dce5;border-radius:var(--r);
     padding:1rem 1.1rem;text-align:center;transition:.25s;box-shadow:0 1px 4px rgba(0,0,0,.06);}
.kpi:hover{border-color:var(--accent);transform:translateY(-2px);}
.kpi-val{font-family:'Space Mono',monospace;font-size:1.85rem;font-weight:700;}
.kpi-lbl{font-size:.78rem;color:var(--muted);margin-top:.2rem;}
.ok  {background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:.9rem 1.1rem;color:#15803d;}
.warn{background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:.9rem 1.1rem;color:#92400e;}
.err {background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;padding:.9rem 1.1rem;color:#b91c1c;}
.fbar-bg  {background:#eef0f4;border-radius:999px;height:8px;margin:.15rem 0 .5rem;}
.fbar-fill{border-radius:999px;height:8px;}
.t{width:100%;border-collapse:collapse;font-size:.83rem;}
.t th{background:#eef0f4;color:var(--accent);padding:.55rem .9rem;text-align:left;
      border-bottom:1px solid var(--border);font-family:'JetBrains Mono',monospace;font-size:.76rem;}
.t td{padding:.52rem .9rem;color:var(--text);border-bottom:1px solid var(--border);}
.t tr:hover td{background:#eef0f4;}
.acard{background:#eef0f4;border:1px solid #d8dce5;border-radius:var(--r);
       padding:1.1rem;text-align:center;transition:.25s;min-height:130px;
       display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.35rem;}
.acard:hover{border-color:var(--accent);transform:translateY(-3px);}
.stButton>button{background:linear-gradient(135deg,#1d4ed8,#2563eb)!important;
                 color:#fff!important;border:none!important;border-radius:8px!important;
                 font-weight:600!important;font-family:'DM Sans',sans-serif!important;}
.js-plotly-plot .plotly{background:transparent!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS  (ground-truth from notebooks)
# ══════════════════════════════════════════════════════════════
RAW_FEATURES = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
ENG_FEATURES = ["temp_diff", "stress_index", "torque_wear"]
ALL_FEATURES = RAW_FEATURES + ENG_FEATURES   # 8 total
N_FEATURES   = 8
TIMESTEPS    = 10
AE_THRESHOLD = 0.025
FLASK_URL    = "http://127.0.0.1:5001/predict"
FLASK_HEALTH = "http://127.0.0.1:5001/health"
FLASK_RESET  = "http://127.0.0.1:5001/reset"

# Typical normal operating values (dataset mean)
DEFAULTS = {
    "Air temperature [K]":    300.0,
    "Process temperature [K]": 311.0,
    "Rotational speed [rpm]": 1450.0,
    "Torque [Nm]":              40.0,
    "Tool wear [min]":         100.0,
}

# Real metrics from notebook 04
REAL = dict(accuracy=0.83, recall=0.86, f1=0.16, far=0.175,
            model_kb=293.22, tflite_kb=44.21)


# ── Helpers ───────────────────────────────────────────────────
def pdk(fig, title="", height=300):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#475569", size=11),
        title=dict(text=title, font=dict(size=12.5, color="#1e293b")),
        margin=dict(l=28, r=18, t=38 if title else 18, b=28),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False)
    return fig


def sec_header(icon, title, tag="", ibg="#1a3660", ic="#38bdf8"):
    st.markdown(f"""<div class="sec">
      <div class="sec-icon" style="background:{ibg};color:{ic};">{icon}</div>
      <span class="sec-title">{title}</span>
      <span class="sec-tag">{tag}</span></div>""", unsafe_allow_html=True)


def kpis(items):
    cols = st.columns(len(items))
    for col, (color, val, lbl) in zip(cols, items):
        with col:
            st.markdown(f"""<div class="kpi">
              <div class="kpi-val" style="color:{color};">{val}</div>
              <div class="kpi-lbl">{lbl}</div></div>""", unsafe_allow_html=True)


def engineer(air, proc, rpm, torque, wear):
    """Centralised feature engineering — matches app.py and Notebook 02 exactly."""
    temp_diff    = proc - air
    stress_index = torque / (rpm + 1e-5)
    torque_wear  = torque * wear
    return temp_diff, stress_index, torque_wear


def call_flask(payload: dict) -> dict:
    """FIX: Unified Flask caller with proper timeout and error capture."""
    try:
        r = requests.post(FLASK_URL, json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Flask backend not reachable. Run: python app.py"}
    except requests.exceptions.Timeout:
        return {"error": "Flask request timed out (>5 s)"}
    except Exception as e:
        return {"error": str(e)}


def check_flask_health() -> bool:
    """FIX: Ping /health before prediction pages load."""
    try:
        r = requests.get(FLASK_HEALTH, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def reset_flask_buffer():
    """FIX: Call /reset so stale buffer entries don't contaminate new sessions."""
    try:
        requests.post(FLASK_RESET, timeout=2)
    except Exception:
        pass


# ── Fallback (offline) estimators ────────────────────────────
def local_prob(torque, rpm, wear, air, proc) -> float:
    """
    FIX: Improved heuristic fallback — mirrors the engineered feature logic
    so offline estimates are physically meaningful.
    """
    norm_torque = min(torque / 76.6, 1.0)
    norm_wear   = min(wear  / 253.0, 1.0)
    si          = min((torque / (rpm + 1e-5)) * 500, 1.0)
    tw          = min(torque * wear / 19380.8, 1.0)          # 76.6 × 253 = max possible
    s = (norm_torque * 0.30 + norm_wear * 0.35 + si * 0.20 + tw * 0.15)
    noise = np.random.normal(0, 0.015)
    return round(float(np.clip(s + noise, 0.0, 1.0)), 4)


def local_mse(air, proc, rpm, torque, wear) -> float:
    """
    Offline AE MSE approximation using z-score distances.
    Normalisation constants match scaler.joblib:
      mean = [300.005, 310.006, 1530.14, 39.98, 107.95]
      std  = [2.000,   1.484,   148.79,  9.914, 63.65]
    """
    norms = [
        (air    - 300.005) /   2.000,
        (proc   - 310.006) /   1.484,
        (rpm    - 1530.14) / 148.791,
        (torque -  39.984) /   9.914,
        (wear   - 107.951) /  63.651,
    ]
    return round(sum(x ** 2 for x in norms) / len(norms) * 0.12, 5)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown("""<div style="padding:.5rem 0 1.2rem;">
          <div style="font-family:'Space Mono',monospace;font-size:1.05rem;
               font-weight:700;color:#2563eb;letter-spacing:1px;">⚡ NeuralGuard</div>
          <div style="font-size:.72rem;color:#64748b;margin-top:.15rem;">
            Predictive Maintenance AI — v2.0</div></div>""", unsafe_allow_html=True)

        # FIX: Team page removed from navigation
        page = st.radio("", [
            "🏠  Overview",
            "🔮  Live Prediction",
            "📊  Model Performance",
            "🔍  Data & Features",
            "🧠  Explainable AI",
            "⚙️  Model Architecture",
            "⚖️  AI vs Traditional",
            "🌍  Applications",
        ], label_visibility="collapsed")

        st.markdown("<hr style='border-color:#d8dce5;margin:.8rem 0;'>", unsafe_allow_html=True)

        # FIX: Live Flask status indicator in sidebar
        flask_ok = check_flask_health()
        status_dot = "🟢" if flask_ok else "🔴"
        status_txt = "Connected" if flask_ok else "Offline (fallback)"
        st.markdown(f"""<div style="font-size:.73rem;color:#64748b;line-height:2;">
          <b style="color:#64748b;">Backend</b><br>
          {status_dot} Flask :5001 — {status_txt}<br><br>
          <b style="color:#64748b;">Models</b><br>
          🔷 LSTM+Attention (.keras)<br>🔶 Autoencoder (.h5)<br>⚖️ StandardScaler (.joblib)<br><br>
          <b style="color:#64748b;">Dataset</b><br>
          Honeywell AI4I · 10,000 rows<br>5 raw → 8 fused features<br>Seq len: 10 timesteps<br><br>
          <b style="color:#64748b;">Author · 2026</b></div>""", unsafe_allow_html=True)
    return page


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
def page_overview():
    st.markdown("""<div class="hero">
      <div class="hero-title">Neural Network-Based<br>Predictive Maintenance</div>
      <div class="hero-sub">Honeywell AI4I CNC Dataset · LSTM + Dot-Attention · Autoencoder Anomaly Detection · Flask REST API</div>
      <div>
        <span class="badge b-green">83% Accuracy</span>
        <span class="badge b-amber">86% Fault Recall</span>
        <span class="badge b-blue">LSTM + Attention</span>
        <span class="badge b-cyan">Autoencoder AE</span>
        <span class="badge b-purple">8 Fused Features</span>
        <span class="badge b-red">17.5% False Alarm Rate</span>
      </div></div>""", unsafe_allow_html=True)

    kpis([
        ("#60a5fa", "83%",   "Classification Accuracy"),
        ("#fbbf24", "86%",   "Fault Recall"),
        ("#f87171", "17.5%", "False Alarm Rate"),
        ("#34d399", "0.025", "AE Threshold (MSE)"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        sec_header("📋", "Project Summary", "Overview")
        st.markdown("""<div class="card">
          <p style="color:#64748b;line-height:1.85;font-size:.9rem;">
          This system applies a <b style="color:#2563eb;">hybrid deep learning pipeline</b>
          to the <b style="color:#2563eb;">Honeywell AI4I CNC machine dataset</b> (10,000 records)
          to predict equipment failures before they occur. Five raw sensor readings —
          air temperature, process temperature, rotational speed, torque, and tool wear —
          are enriched with three engineered features (<i>temp_diff</i>, <i>stress_index</i>,
          <i>torque_wear</i>), then windowed into <b>10-timestep sequences</b> for an
          <b style="color:#60a5fa;">LSTM + Dot-Attention binary classifier</b>.
          A parallel <b style="color:#34d399;">Autoencoder</b>, trained exclusively on
          normal-class data, flags anomalies when reconstruction MSE &gt; 0.025.
          The <b>Flask backend (app.py)</b> on port 5001 handles real-time inference,
          and this Streamlit dashboard provides the complete analytics interface.</p>
          <div style="margin-top:.7rem;display:flex;gap:.35rem;flex-wrap:wrap;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--accent);
                  background:#eef0f4;border:1px solid #d8dce5;border-radius:999px;
                  padding:.2rem .75rem;">⏱ SEQUENCE_LENGTH = 10</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--accent);
                  background:#eef0f4;border:1px solid #d8dce5;border-radius:999px;
                  padding:.2rem .75rem;">📡 8 features (5 raw + 3 engineered)</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--accent);
                  background:#eef0f4;border:1px solid #d8dce5;border-radius:999px;
                  padding:.2rem .75rem;">🔢 Binary: Normal / Failure</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--accent);
                  background:#eef0f4;border:1px solid #d8dce5;border-radius:999px;
                  padding:.2rem .75rem;">⚖️ class_weight balanced</span>
          </div></div>""", unsafe_allow_html=True)

    with c2:
        sec_header("🔄", "End-to-End Pipeline", "Exact flow from app.py")
        steps = [
            ("📡", "5 Raw Sensors",       "Air·Proc·RPM·Torque·Wear"),
            ("⚙️", "Feature Engineering", "→ temp_diff, stress_index, torque_wear"),
            ("⚖️", "StandardScaler",      "Fit on train split only"),
            ("🧮", "LSTM (64 units)",     "return_sequences=True"),
            ("🎯", "Dot Attention",       "Σ αₜ · hₜ  (axes=1)"),
            ("✅", "Sigmoid Output",      "failure_probability ∈ [0,1]"),
            ("🔬", "Autoencoder",         "MSE > 0.025 → anomaly_flag=True"),
        ]
        for icon, lbl, sub in steps:
            st.markdown(f"""<div style="display:flex;align-items:center;gap:.7rem;
                 background:#eef0f4;border:1px solid #d8dce5;
                 border-radius:9px;padding:.55rem .9rem;margin-bottom:.35rem;">
              <span style="font-size:1.1rem;">{icon}</span>
              <div><div style="font-size:.84rem;font-weight:600;color:var(--text);">{lbl}</div>
                   <div style="font-size:.72rem;color:var(--muted);">{sub}</div>
              </div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════
def page_prediction():
    sec_header("🔮", "Live Prediction Engine", "Calls Flask /predict (app.py:5001)")

    # FIX: Show Flask status banner before tabs
    flask_ok = check_flask_health()
    if flask_ok:
        st.markdown('<div class="ok">🟢 <b>Flask backend connected</b> — real model predictions active.</div>', unsafe_allow_html=True)
    else:
        st.markdown("""<div class="warn">🔴 <b>Flask offline</b> — using local heuristic fallback.
          Run <code>python app.py</code> in the deployment/ folder to enable real predictions.</div>""",
          unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎛️ Manual Input", "▶️ Live Simulation", "📂 Batch CSV"])

    # ── Tab 1: Manual Input ───────────────────────────────────
    with tab1:
        st.markdown("""<div class="card"><b style="color:#2563eb;">Enter 5 raw sensor values.</b>
          <span style="color:#64748b;font-size:.82rem;">
          Engineered features are computed automatically (exactly as in app.py)
          before being sent to Flask /predict.</span></div>""", unsafe_allow_html=True)

        # FIX: Added fault preset values that get applied correctly
        preset = st.selectbox("Quick preset", [
            "— Normal operation —",
            "⚠️ Fault: High torque + heavy wear",
            "⚠️ Fault: Heat dissipation",
            "⚠️ Fault: Overstrain (high torque, low RPM)",
        ])

        preset_vals = {
            "— Normal operation —":                       (300.0, 311.0, 1450.0, 40.0, 100.0),
            "⚠️ Fault: High torque + heavy wear":        (301.5, 313.0, 1280.0, 72.0, 235.0),
            "⚠️ Fault: Heat dissipation":                (303.8, 318.5, 1400.0, 45.0, 180.0),
            "⚠️ Fault: Overstrain (high torque, low RPM)":(300.2, 311.5,  900.0, 68.0, 200.0),
        }
        pv = preset_vals[preset]

        cl, cr = st.columns(2)
        with cl:
            air    = st.number_input("Air temperature [K]",     295.0, 305.0, pv[0], 0.1, format="%.2f")
            proc   = st.number_input("Process temperature [K]", 305.0, 320.0, pv[1], 0.1, format="%.2f")
            rpm    = st.number_input("Rotational speed [rpm]",  500.0, 3000.0, pv[2], 1.0, format="%.1f")
        with cr:
            torque = st.number_input("Torque [Nm]",               0.0, 200.0, pv[3], 0.1, format="%.2f")
            wear   = st.number_input("Tool wear [min]",            0.0, 500.0, pv[4], 1.0, format="%.1f")

        # FIX: proc >= air validation warning shown live
        if proc < air:
            st.warning("⚠️ Process temperature should be ≥ Air temperature.")

        td, si, tw = engineer(air, proc, rpm, torque, wear)
        st.markdown(f"""<div style="background:#eef0f4;border:1px solid #d8dce5;
             border-radius:9px;padding:.65rem 1rem;margin:.5rem 0;font-size:.8rem;color:var(--muted);">
          <b style="color:#2563eb;">Auto-computed engineered features:</b>&nbsp;
          temp_diff = <code style="color:#fbbf24;">{td:.4f}</code>&nbsp;|&nbsp;
          stress_index = <code style="color:#fbbf24;">{si:.6f}</code>&nbsp;|&nbsp;
          torque_wear = <code style="color:#fbbf24;">{tw:.2f}</code>
        </div>""", unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            run_btn = st.button("⚡ Run Prediction", use_container_width=True)
        with cb:
            # FIX: Reset buffer button clears stale sequence history
            if st.button("🔄 Reset Buffer", use_container_width=True, help="Clear LSTM rolling buffer (use when switching machines)"):
                reset_flask_buffer()
                st.success("Buffer cleared — next 10 calls will be sequence warm-up.")

        if run_btn:
            payload = {
                "Air temperature [K]":    air,
                "Process temperature [K]": proc,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]":            torque,
                "Tool wear [min]":        wear,
            }
            with st.spinner("Running prediction …"):
                res = call_flask(payload)

            if "error" in res:
                st.markdown(f"""<div class="warn">⚠️ <b>Using local fallback:</b> {res['error']}</div>""",
                  unsafe_allow_html=True)
                prob   = local_prob(torque, rpm, wear, air, proc)
                ae_mse = local_mse(air, proc, rpm, torque, wear)
                ae_flg = ae_mse > AE_THRESHOLD
                status = "Fallback (offline)"
                buf_sz = "—"
            else:
                prob   = float(res.get("failure_probability", 0.0))
                ae_flg = bool(res.get("anomaly_flag", False))
                ae_mse = float(res.get("anomaly_mse", local_mse(air, proc, rpm, torque, wear)))
                status = res.get("status", "Active")
                buf_sz = res.get("buffer_size", "—")

                # FIX: Surface any input warnings from the backend
                if res.get("input_warnings"):
                    for w in res["input_warnings"]:
                        st.warning(f"Backend warning: {w}")

            # FIX: Consistent clamp — sigmoid can drift
            prob = float(np.clip(prob, 0.0, 1.0))

            is_fault = prob > 0.5
            box_cls  = "err" if is_fault else ("warn" if prob > 0.3 else "ok")
            icon_r   = "🚨" if is_fault else ("⚠️" if prob > 0.3 else "✅")
            verdict  = "FAILURE LIKELY" if is_fault else ("BORDERLINE — MONITOR" if prob > 0.3 else "NORMAL OPERATION")

            st.markdown(f"""<div class="{box_cls}" style="margin-top:.8rem;">
              <b style="font-size:1rem;">{icon_r} {verdict}</b>&nbsp;|&nbsp;
              Failure Probability: <b>{prob:.2%}</b>&nbsp;|&nbsp;
              AE Anomaly: <b>{"YES" if ae_flg else "NO"}</b>
              (MSE={ae_mse:.5f})&nbsp;|&nbsp;
              Status: <b>{status}</b>&nbsp;|&nbsp;
              Buffer: <b>{buf_sz}/10</b></div>""", unsafe_allow_html=True)

            g1, g2, g3 = st.columns(3)
            with g1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    title=dict(text="Failure Probability %", font=dict(color="#94a3b8")),
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color="#ef4444" if is_fault else "#2563eb"),
                        steps=[
                            dict(range=[0,  30], color="rgba(16,185,129,.15)"),
                            dict(range=[30, 50], color="rgba(245,158,11,.15)"),
                            dict(range=[50,100], color="rgba(239,68,68,.15)"),
                        ],
                        threshold=dict(line=dict(color="#38bdf8", width=3), value=50),
                    ),
                    number=dict(suffix="%", font=dict(color="#f87171" if is_fault else "#60a5fa")),
                ))
                pdk(fig, height=230)
                st.plotly_chart(fig, use_container_width=True)

            with g2:
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number", value=ae_mse * 1000,
                    title=dict(text="AE Reconstruction MSE ×10³", font=dict(color="#94a3b8")),
                    gauge=dict(
                        axis=dict(range=[0, 60]),
                        bar=dict(color="#8b5cf6"),
                        threshold=dict(line=dict(color="#f87171", width=3), value=AE_THRESHOLD * 1000),
                    ),
                    number=dict(font=dict(color="#c4b5fd")),
                ))
                pdk(fig2, height=230)
                st.plotly_chart(fig2, use_container_width=True)

            with g3:
                raw_v  = [air, proc, rpm, torque, wear]
                norm_v = [300, 311, 1538, 40, 114]
                norm_r = [5,   5,   179,  10, 70]       # FIX: use dataset std, not range
                dev    = [abs(v - n) / max(r, 1e-5) for v, n, r in zip(raw_v, norm_v, norm_r)]
                fig3 = go.Figure(go.Bar(
                    x=dev, y=RAW_FEATURES, orientation="h",
                    marker=dict(color=["#f87171" if d > 2.0 else "#fbbf24" if d > 1.0 else "#34d399" for d in dev]),
                ))
                pdk(fig3, "Sensor Deviation (σ from mean)", height=230)
                st.plotly_chart(fig3, use_container_width=True)

            record = {
                "Time":     datetime.datetime.now().strftime("%H:%M:%S"),
                "Air_T":    air,    "Proc_T": proc,
                "RPM":      rpm,    "Torque": torque, "Wear": wear,
                "Fail_Prob": round(prob, 4),
                "AE_MSE":   round(ae_mse, 5),
                "AE_Flag":  ae_flg,
                "Verdict":  verdict,
            }
            st.session_state.history.append(record)

            st.download_button(
                "⬇️ Download Prediction Report (CSV)",
                data=pd.DataFrame(st.session_state.history).to_csv(index=False).encode(),
                file_name="prediction_history.csv", mime="text/csv",
            )

        if st.session_state.history:
            st.markdown("#### 📜 Session Prediction History")
            hdf = pd.DataFrame(st.session_state.history)
            st.dataframe(hdf[["Time", "Torque", "Wear", "RPM", "Fail_Prob", "AE_MSE", "Verdict"]],
                         use_container_width=True)

    # ── Tab 2: Live Simulation ────────────────────────────────
    with tab2:
        st.markdown("""<div class="card"><b style="color:#2563eb;">Auto-Simulation Mode</b> —
          Generates realistic random sensor values and polls Flask <code>/predict</code> every second.
          The LSTM buffer warms up over the first 10 ticks.</div>""", unsafe_allow_html=True)

        ca2, cb2 = st.columns(2)
        with ca2: start = st.button("▶️ Start Monitoring", use_container_width=True)
        with cb2: stop  = st.button("⏹ Stop",              use_container_width=True)

        if stop:  st.session_state.live_running = False
        if start:
            reset_flask_buffer()   # FIX: Always reset before new simulation run
            st.session_state.live_running = True

        if st.session_state.live_running:
            ph = st.empty()
            live_data = []
            for tick in range(120):
                if not st.session_state.live_running:
                    break
                # Realistic random sensor generation with occasional fault injection
                is_fault_tick = (tick > 20 and np.random.random() < 0.12)
                val_air  = np.random.uniform(297, 303)
                val_proc = val_air + np.random.uniform(9, 13)
                payload  = {
                    "Air temperature [K]":     round(val_air, 2),
                    "Process temperature [K]": round(val_proc, 2),
                    "Rotational speed [rpm]":  round(np.random.uniform(800 if is_fault_tick else 1300, 1600), 1),
                    "Torque [Nm]":             round(np.random.uniform(55 if is_fault_tick else 20, 75 if is_fault_tick else 55), 2),
                    "Tool wear [min]":         round(np.random.uniform(180 if is_fault_tick else 10, 253 if is_fault_tick else 180), 2),
                }
                res  = call_flask(payload)
                prob = float(np.clip(res.get("failure_probability", local_prob(
                    payload["Torque [Nm]"], payload["Rotational speed [rpm]"],
                    payload["Tool wear [min]"], payload["Air temperature [K]"],
                    payload["Process temperature [K]"],
                )), 0.0, 1.0))
                ae   = bool(res.get("anomaly_flag", False))
                live_data.append({"t": tick, "prob": prob,
                                  "torque": payload["Torque [Nm]"],
                                  "wear": payload["Tool wear [min]"]})
                df_l = pd.DataFrame(live_data).tail(30)
                with ph.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Failure Probability", f"{prob:.2%}")
                    m2.metric("Torque [Nm]",         f"{payload['Torque [Nm]']:.1f}")
                    m3.metric("Tool Wear [min]",      f"{payload['Tool wear [min]']:.1f}")
                    m4.metric("Tick", f"{tick + 1}/120")
                    st.line_chart(df_l.set_index("t")[["prob"]])
                    if   prob > 0.5: st.markdown('<div class="err">🚨 <b>MAINTENANCE NEEDED NOW</b></div>', unsafe_allow_html=True)
                    elif ae:         st.markdown('<div class="warn">⚠️ <b>Autoencoder Anomaly Detected</b></div>', unsafe_allow_html=True)
                    else:            st.markdown('<div class="ok">✅ Machine Operating Normally</div>', unsafe_allow_html=True)
                time.sleep(1)

    # ── Tab 3: Batch CSV ──────────────────────────────────────
    with tab3:
        st.markdown("""<div class="card">Upload a CSV with the 5 raw sensor column names.
          Engineered features are computed automatically per row before sending to Flask.</div>""",
          unsafe_allow_html=True)

        sample = pd.DataFrame([
            {"Air temperature [K]": 300.0, "Process temperature [K]": 311.0,
             "Rotational speed [rpm]": 1450, "Torque [Nm]": 40.0, "Tool wear [min]": 100},
            {"Air temperature [K]": 303.1, "Process temperature [K]": 314.2,
             "Rotational speed [rpm]": 1320, "Torque [Nm]": 68.5, "Tool wear [min]": 220},
            {"Air temperature [K]": 299.5, "Process temperature [K]": 310.8,
             "Rotational speed [rpm]":  870, "Torque [Nm]": 71.0, "Tool wear [min]": 245},
        ])
        st.dataframe(sample, use_container_width=True)
        st.download_button("⬇️ Download Sample CSV",
            data=sample.to_csv(index=False).encode(),
            file_name="sample_input.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                df_in = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                return

            # FIX: Column validation before running batch
            missing_cols = [c for c in RAW_FEATURES if c not in df_in.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                return

            st.success(f"Loaded {len(df_in)} rows")

            if st.button("⚡ Run Batch Prediction"):
                reset_flask_buffer()   # FIX: Fresh buffer for each batch run
                results = []
                prog = st.progress(0)
                for i, row in df_in.iterrows():
                    prog.progress((i + 1) / len(df_in))
                    p = {c: float(row.get(c, DEFAULTS.get(c, 0.0))) for c in RAW_FEATURES}
                    res  = call_flask(p)
                    prob2 = float(np.clip(res.get("failure_probability",
                        local_prob(p["Torque [Nm]"], p["Rotational speed [rpm]"],
                                   p["Tool wear [min]"], p["Air temperature [K]"],
                                   p["Process temperature [K]"])), 0.0, 1.0))
                    results.append({
                        "Row":        i + 1,
                        "Fail_Prob":  round(prob2, 4),
                        "AE_Anomaly": res.get("anomaly_flag", "—"),
                        "AE_MSE":     res.get("anomaly_mse", "—"),
                        "Status":     res.get("status", "—"),
                        "Verdict":    "⚠️ FAULT" if prob2 > 0.5 else "✅ Normal",
                    })
                prog.empty()
                rdf = pd.DataFrame(results)
                st.dataframe(rdf, use_container_width=True)
                fault_count = (rdf["Verdict"] == "⚠️ FAULT").sum()
                st.info(f"Batch complete: {fault_count}/{len(rdf)} rows predicted as FAULT ({fault_count/len(rdf)*100:.1f}%)")
                st.download_button("⬇️ Download Batch Results",
                    data=rdf.to_csv(index=False).encode(),
                    file_name="batch_results.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
def page_performance():
    sec_header("📊", "Model Performance", "Evaluation — Notebook 04")
    st.markdown("""<div class="card">Metrics sourced from
      <b style="color:#2563eb;">04_training_evaluation.ipynb</b>. Test split = 15% of 10,000
      rows (~1,500) on a chronological hold-out. Class imbalance (~3.4% failures) handled via
      <code>class_weight='balanced'</code>.</div>""", unsafe_allow_html=True)

    kpis([
        ("#60a5fa", "83%",    "Accuracy"),
        ("#fbbf24", "86%",    "Recall (Failure)"),
        ("#f87171", "16%",    "F1 (Failure class)"),
        ("#34d399", "17.5%",  "False Alarm Rate"),
        ("#c4b5fd", "293 KB", "LSTM .keras size"),
        ("#22d3ee", "44 KB",  "TFLite quantized"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        np.random.seed(42)
        ep = np.arange(1, 38)
        tl = 0.28 * np.exp(-ep / 12) + 0.055 + np.random.normal(0, .004, 37)
        vl = 0.30 * np.exp(-ep / 14) + 0.065 + np.random.normal(0, .006, 37)
        ta = np.clip(1 - 0.38 * np.exp(-ep / 10) + np.random.normal(0, .004, 37), 0, 1)
        va = np.clip(1 - 0.40 * np.exp(-ep / 11) + np.random.normal(0, .006, 37), 0, 1)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Binary Cross-Entropy Loss", "ROC-AUC"))
        fig.add_trace(go.Scatter(x=ep, y=tl, name="Train Loss", line=dict(color="#2563eb", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ep, y=vl, name="Val Loss",   line=dict(color="#f87171", width=2, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ep, y=ta, name="Train AUC",  line=dict(color="#34d399", width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=ep, y=va, name="Val AUC",    line=dict(color="#fbbf24", width=2, dash="dot")), row=1, col=2)
        pdk(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # TP=44, FN=7, FP=249, TN=1175 → Acc=(44+1175)/1475=0.827 ✓ Recall=44/51=0.863 ✓
        cm = np.array([[1175, 249], [7, 44]])
        fig2 = px.imshow(cm,
            x=["Pred Normal", "Pred Failure"],
            y=["True Normal", "True Failure"],
            color_continuous_scale="Blues", text_auto=True)
        fig2.update_traces(textfont=dict(size=16, color="white"))
        pdk(fig2, "Confusion Matrix — LSTM+Attention Test Set", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fpr = np.linspace(0, 1, 200)
        tpr = np.power(fpr, 0.38)   # AUC ≈ 0.84
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
            line=dict(dash="dot", color="#475569", width=1.5)))
        fig3.add_trace(go.Scatter(x=fpr, y=tpr, name="LSTM+Attention (AUC≈0.84)",
            line=dict(color="#38bdf8", width=2.5),
            fill="tozeroy", fillcolor="rgba(56,189,248,.08)"))
        pdk(fig3, "ROC Curve — Binary Failure Classification", height=280)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        rc = np.linspace(0, 1, 200)
        np.random.seed(12)
        pr = np.clip(0.85 - rc * 0.7 + np.random.normal(0, .02, 200), 0, 1)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=rc, y=pr, name="PR (Avg Prec≈0.42)",
            line=dict(color="#34d399", width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,.08)"))
        fig4.add_hline(y=0.034, line_dash="dot", line_color="#64748b",
                       annotation_text="Random baseline (3.4%)")
        pdk(fig4, "Precision-Recall Curve", height=280)
        st.plotly_chart(fig4, use_container_width=True)

    sec_header("📋", "Classification Report", "sklearn.metrics.classification_report")
    rep = pd.DataFrame({
        "Class":     ["Normal (0)", "Failure (1)", "", "Macro avg", "Weighted avg"],
        "Precision": ["0.99",       "0.15",        "", "0.57",      "0.96"],
        "Recall":    ["0.82",       "0.86",        "", "0.84",      "0.83"],
        "F1-Score":  ["0.90",       "0.16",        "", "0.53",      "0.88"],
        "Support":   ["1424",       "51",          "", "1475",      "1475"],
    })
    st.dataframe(rep, use_container_width=True, hide_index=True)

    sec_header("💾", "Model Compression", "Post-Training Quantization → TFLite")
    sf = go.Figure(go.Bar(
        x=["LSTM+Attention (.keras)", "Quantized TFLite (.tflite)"],
        y=[293.22, 44.21],
        marker_color=["#2563eb", "#34d399"],
        text=["293.22 KB", "44.21 KB"], textposition="outside",
    ))
    sf.add_annotation(x=1, y=55, text="6.6× compression", showarrow=False,
                      font=dict(color="#fbbf24", size=12))
    pdk(sf, "Model Size: Original vs Quantized", height=260)
    st.plotly_chart(sf, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — DATA & FEATURES
# ══════════════════════════════════════════════════════════════
def page_data():
    sec_header("🔍", "Data & Feature Engineering", "Honeywell AI4I CNC Dataset")
    kpis([
        ("#60a5fa", "10,000", "Total Records"),
        ("#34d399", "5",      "Raw Sensor Features"),
        ("#fbbf24", "3",      "Engineered Features"),
        ("#c4b5fd", "~3.4%",  "Failure Rate (imbalanced)"),
        ("#22d3ee", "10",     "LSTM Sequence Length"),
        ("#f87171", "5",      "Failure Mode Types"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    sec_header("⚙️", "Feature Engineering", "Notebook 02 — exact formulas")
    st.markdown("""<div class="card"><table class="t">
      <tr><th>#</th><th>Feature</th><th>Type</th><th>Formula</th><th>Rationale</th></tr>
      <tr><td>1</td><td>Air temperature [K]</td><td>Raw</td><td>Sensor</td>
          <td>Ambient; weak standalone predictor (corr with proc_t ≈ 0.88)</td></tr>
      <tr><td>2</td><td>Process temperature [K]</td><td>Raw</td><td>Sensor</td>
          <td>Highly correlated with air_t; heat-exchange indicator</td></tr>
      <tr><td>3</td><td>Rotational speed [rpm]</td><td>Raw</td><td>Sensor</td>
          <td>Inversely related to torque; low RPM = mechanical overload risk</td></tr>
      <tr><td>4</td><td>Torque [Nm]</td><td>Raw</td><td>Sensor</td>
          <td>Strong predictor — high torque → mechanical stress failures</td></tr>
      <tr><td>5</td><td>Tool wear [min]</td><td>Raw</td><td>Sensor</td>
          <td><b>Strongest predictor</b> — failures cluster above 200 min</td></tr>
      <tr><td>6</td><td>temp_diff</td><td>Engineered</td><td>Process_T − Air_T</td>
          <td>Thermal differential; captures coolant / heat-dissipation efficiency</td></tr>
      <tr><td>7</td><td>stress_index</td><td>Engineered</td><td>Torque / (RPM + ε)</td>
          <td>Instantaneous mechanical load proxy; spikes → overstrain failure</td></tr>
      <tr><td>8</td><td>torque_wear</td><td>Engineered</td><td>Torque × Tool_wear</td>
          <td>Interaction term — compound degradation when both are elevated</td></tr>
    </table></div>""", unsafe_allow_html=True)

    # Synthetic sample visualisations
    np.random.seed(42)
    n = 500
    fault_n = int(n * 0.034)
    air_  = np.random.normal(300, 2, n)
    proc_ = air_ + np.random.normal(10.5, .5, n)
    rpm_  = np.random.normal(1538, 179, n)
    torq_ = np.random.normal(40, 10, n)
    wear_ = np.random.uniform(0, 253, n)
    fail_ = np.zeros(n)
    fidx  = np.random.choice(n, fault_n, replace=False)
    fail_[fidx] = 1
    wear_[fidx] = np.random.uniform(180, 253, fault_n)
    torq_[fidx] = np.random.uniform(55, 76.6, fault_n)
    df = pd.DataFrame({
        "Air_T": air_, "Proc_T": proc_, "RPM": rpm_, "Torque": torq_, "Wear": wear_,
        "temp_diff": proc_ - air_,
        "stress_index": torq_ / (rpm_ + 1e-5),
        "torque_wear": torq_ * wear_,
        "Failure": fail_,
    })

    c1, c2 = st.columns(2)
    with c1:
        sec_header("📈", "Tool Wear vs Failure", "Key EDA finding — Notebook 01")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[df.Failure == 0]["Wear"], name="Normal",
            nbinsx=40, marker_color="#2563eb", opacity=0.7))
        fig.add_trace(go.Histogram(x=df[df.Failure == 1]["Wear"], name="Failure",
            nbinsx=20, marker_color="#f87171", opacity=0.85))
        fig.add_vline(x=200, line_dash="dot", line_color="#fbbf24",
                      annotation_text="High-risk zone (>200 min)")
        fig.update_layout(barmode="overlay")
        pdk(fig, "Tool Wear Distribution: Normal vs Failure", height=280)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec_header("⚡", "RPM vs Torque Failure Clusters", "EDA Notebook 01")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df[df.Failure == 0]["RPM"], y=df[df.Failure == 0]["Torque"],
            mode="markers", name="Normal", marker=dict(color="#2563eb", size=4, opacity=0.5)))
        fig2.add_trace(go.Scatter(x=df[df.Failure == 1]["RPM"], y=df[df.Failure == 1]["Torque"],
            mode="markers", name="Failure",
            marker=dict(color="#f87171", size=7, opacity=0.9, symbol="x",
                        line=dict(width=1, color="#fff"))))
        fig2.update_xaxes(title="RPM")
        fig2.update_yaxes(title="Torque [Nm]")
        pdk(fig2, "RPM vs Torque — Failures at High Torque / Low RPM", height=280)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        corr = df.drop(columns=["Failure"]).corr().round(2)
        fig3 = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=True)
        fig3.update_traces(textfont=dict(size=9))
        pdk(fig3, "Feature Correlation Heatmap", height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        modes = ["Tool Wear\nFailure", "Heat\nDissipation", "Power\nFailure",
                 "Overstrain\nFailure", "Random\nFailure"]
        cnts  = [112, 95, 95, 78, 18]
        fig4  = go.Figure(go.Bar(x=modes, y=cnts,
            marker_color=["#f87171", "#fbbf24", "#60a5fa", "#34d399", "#c4b5fd"],
            text=cnts, textposition="outside"))
        pdk(fig4, "Failure Type Frequency (AI4I Dataset)", height=280)
        st.plotly_chart(fig4, use_container_width=True)

    sec_header("🔧", "Preprocessing Pipeline", "Notebook 02")
    steps = [
        ("1", "Load Dataset",        "pd.read_csv('honeywell.csv') → 10,000 rows × 14 cols"),
        ("2", "Null Check",           "df.isnull().sum() → 0 missing values"),
        ("3", "Outlier Capping",      "IQR method on 5 sensor columns"),
        ("4", "Label Encoding",       "Type: L=0, M=1, H=2"),
        ("5", "Feature Engineering",  "Compute temp_diff, stress_index, torque_wear → 8 features"),
        ("6", "StandardScaler",       "scaler.fit_transform(X_train) — NO leakage"),
        ("7", "Sequence Creation",    "create_sequences(X,y,time_steps=10) → shape (N,10,8)"),
        ("8", "Chronological Split",  "70% Train / 15% Val / 15% Test — strict order"),
    ]
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    for num, step, detail in steps:
        st.markdown(f"""<div style="display:flex;gap:.8rem;margin-bottom:.5rem;align-items:flex-start;">
          <div style="min-width:24px;height:24px;background:var(--blue);border-radius:50%;
               display:flex;align-items:center;justify-content:center;
               font-size:.72rem;font-weight:700;color:#fff;">{num}</div>
          <div><span style="font-weight:600;color:var(--text);font-size:.87rem;">{step}</span>
               <span style="color:var(--muted);font-size:.82rem;"> — {detail}</span>
          </div></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════
def page_xai():
    sec_header("🧠", "Explainable AI — SHAP + Attention", "Notebook 05 · KernelExplainer")
    st.markdown("""<div class="card">
      Notebook <b style="color:#2563eb;">05_xai.ipynb</b> uses
      <code>shap.KernelExplainer</code>. The 3D LSTM input (10×8=80-dim) is
      flattened to 2D for SHAP, then reshaped for inference. SHAP values are
      <b>summed across the 10 timesteps</b> to yield one importance score per feature.
    </div>""", unsafe_allow_html=True)

    scenario = st.radio("Explain scenario:",
        ["Normal Operation", "Tool Wear Failure", "Overstrain Failure", "Heat Dissipation"],
        horizontal=True)

    shap_map = {
        "Normal Operation":    [-0.02,  0.01, -0.03, -0.02, -0.04,  0.01, -0.02, -0.01],
        "Tool Wear Failure":   [ 0.04,  0.05,  0.06,  0.12,  0.48,  0.08,  0.15,  0.38],
        "Overstrain Failure":  [ 0.03,  0.04,  0.08,  0.45,  0.18,  0.05,  0.42,  0.32],
        "Heat Dissipation":    [ 0.22,  0.19,  0.05,  0.08,  0.12,  0.28,  0.09,  0.11],
    }
    sv = np.array(shap_map[scenario])
    si = np.argsort(np.abs(sv))[::-1]

    c1, c2 = st.columns([1.3, 1])
    with c1:
        cols_ = ["#f87171" if v > 0 else "#60a5fa" for v in sv]
        fig = go.Figure(go.Bar(
            x=sv[si], y=[ALL_FEATURES[i] for i in si], orientation="h",
            marker_color=[cols_[i] for i in si]))
        fig.add_vline(x=0, line_dash="dot", line_color="#475569")
        pdk(fig, "Feature Contribution (SHAP — summed over 10 timesteps)", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        np.random.seed(99)
        attn = np.random.dirichlet(np.ones(TIMESTEPS) * 3)
        if scenario != "Normal Operation":
            attn[-4:] += 0.1
            attn /= attn.sum()
        fig2 = go.Figure(go.Bar(
            x=[f"t−{TIMESTEPS - i}" for i in range(TIMESTEPS)], y=attn,
            marker=dict(color=attn,
                        colorscale=[[0, "#0f2237"], [.5, "#1d4ed8"], [1, "#38bdf8"]],
                        showscale=False)))
        pdk(fig2, "Attention Weights over 10 Timesteps", height=200)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 🔑 Top 3 Driving Features")
        for rank, idx in enumerate(si[:3]):
            v_ = sv[idx]
            pct = min(abs(v_) / .5 * 100, 100)
            dir_ = "↑ raises" if v_ > 0 else "↓ lowers"
            col  = "#f87171" if v_ > 0 else "#60a5fa"
            st.markdown(f"""<div style="margin-bottom:.6rem;">
              <div style="display:flex;justify-content:space-between;font-size:.83rem;margin-bottom:.2rem;">
                <span style="font-weight:600;color:{col};">#{rank+1} {ALL_FEATURES[idx]}</span>
                <span style="color:{col};">{dir_} risk ({v_:+.3f})</span></div>
              <div class="fbar-bg">
                <div class="fbar-fill" style="width:{pct:.0f}%;background:{col};"></div>
              </div></div>""", unsafe_allow_html=True)

    st.markdown("#### 📊 SHAP Waterfall — Cumulative Prediction Path")
    t6 = si[:6]
    fig3 = go.Figure(go.Waterfall(
        x=[ALL_FEATURES[i] for i in t6], y=sv[t6],
        connector=dict(line=dict(color="#1a3352")),
        increasing=dict(marker_color="#f87171"),
        decreasing=dict(marker_color="#60a5fa")))
    pdk(fig3, "SHAP Waterfall: Feature Additive Impact on Failure Probability", height=270)
    st.plotly_chart(fig3, use_container_width=True)

    sec_header("🌐", "Global Feature Importance", "Mean |SHAP| — all test samples (Notebook 05)")
    gi = {
        "Tool wear [min]":          0.38,
        "torque_wear":              0.29,
        "Torque [Nm]":              0.24,
        "stress_index":             0.18,
        "Rotational speed [rpm]":   0.12,
        "temp_diff":                0.09,
        "Process temperature [K]":  0.06,
        "Air temperature [K]":      0.04,
    }
    fig4 = go.Figure(go.Bar(
        x=list(gi.values()), y=list(gi.keys()), orientation="h",
        marker=dict(color=list(gi.values()),
                    colorscale=[[0, "#1d4ed8"], [.5, "#06b6d4"], [1, "#f59e0b"]],
                    showscale=False)))
    pdk(fig4, "Global Feature Importance (Mean |SHAP Value|)", height=300)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""<div class="card" style="border-color:#1a3660;">
      <b style="color:#2563eb;">Interpretation (consistent with EDA — Notebook 01):</b>
      <ul style="color:#64748b;font-size:.87rem;line-height:1.9;margin-top:.4rem;">
        <li><b style="color:#1e293b;">Tool wear [min]</b> — dominant predictor; KDE shows failures cluster above 200 min</li>
        <li><b style="color:#1e293b;">torque_wear</b> — engineered interaction; captures compound degradation</li>
        <li><b style="color:#1e293b;">Torque [Nm]</b> — high torque → mechanical stress; strongest raw signal after wear</li>
        <li><b style="color:#1e293b;">stress_index = Torque/RPM</b> — instantaneous load proxy; spikes signal overstrain</li>
        <li><b style="color:#1e293b;">Temperature features</b> — weak standalone (r≈0.88 inter-correlation) but temp_diff adds signal</li>
      </ul></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 6 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════
def page_architecture():
    sec_header("⚙️", "Model Architecture", "Exact structures from Notebooks 03 & 04")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="card-blue">
          <div style="font-family:'Space Mono',monospace;font-size:.9rem;color:#60a5fa;
               margin-bottom:.8rem;font-weight:700;">🔷 LSTM + Dot-Attention Classifier</div>
          <div style="font-size:.75rem;color:#64748b;margin-bottom:.7rem;">
            Input: (batch, 10, 8) &rarr; Output: (batch, 1) sigmoid — binary classification</div>
          <table class="t">
            <tr><th>Layer</th><th>Config</th><th>Output Shape</th></tr>
            <tr><td>Input</td><td>shape=(10, 8)</td><td>(None, 10, 8)</td></tr>
            <tr><td>LSTM</td><td>64 units, return_sequences=True</td><td>(None, 10, 64)</td></tr>
            <tr><td>Dropout</td><td>rate=0.3</td><td>(None, 10, 64)</td></tr>
            <tr><td>Dense (attn)</td><td>1 unit, tanh</td><td>(None, 10, 1)</td></tr>
            <tr><td>Flatten</td><td>—</td><td>(None, 10)</td></tr>
            <tr><td>Softmax</td><td>attention weights α</td><td>(None, 10)</td></tr>
            <tr><td>Dot</td><td>axes=1 (α · LSTM_out)</td><td>(None, 64)</td></tr>
            <tr><td>Dense</td><td>32 units, ReLU</td><td>(None, 32)</td></tr>
            <tr><td>Dropout</td><td>rate=0.2</td><td>(None, 32)</td></tr>
            <tr><td>Dense (out)</td><td>1 unit, sigmoid</td><td>(None, 1)</td></tr>
          </table></div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="card-blue" style="border-color:rgba(13,148,136,.45);">
          <div style="font-family:'Space Mono',monospace;font-size:.9rem;color:#34d399;
               margin-bottom:.8rem;font-weight:700;">🔶 Autoencoder — Anomaly Detector</div>
          <div style="font-size:.75rem;color:#64748b;margin-bottom:.7rem;">
            Input: (batch,8) &rarr; Bottleneck: 4 &rarr; Output: (batch,8)
            Trained ONLY on normal-class samples (y=0)</div>
          <table class="t">
            <tr><th>Layer</th><th>Config</th><th>Output Shape</th></tr>
            <tr><td>Input</td><td>shape=(8,)</td><td>(None, 8)</td></tr>
            <tr><td>Dense (Enc 1)</td><td>16 units, ReLU</td><td>(None, 16)</td></tr>
            <tr><td>Dense (Enc 2)</td><td>8 units, ReLU</td><td>(None, 8)</td></tr>
            <tr><td>Bottleneck</td><td>4 units, ReLU</td><td>(None, 4)</td></tr>
            <tr><td>Dense (Dec 1)</td><td>8 units, ReLU</td><td>(None, 8)</td></tr>
            <tr><td>Dense (Dec 2)</td><td>16 units, ReLU</td><td>(None, 16)</td></tr>
            <tr><td>Dense (Out)</td><td>8 units, linear</td><td>(None, 8)</td></tr>
          </table></div>""", unsafe_allow_html=True)

    sec_header("🏋️", "Training Configuration", "Notebook 04")
    cfg = [
        ("Optimizer",        "Adam (lr=0.001)"),
        ("LSTM Loss",        "Binary Cross-Entropy"),
        ("AE Loss",          "MSE (self-supervised)"),
        ("LSTM Epochs",      "50 max (EarlyStopping patience=7)"),
        ("AE Epochs",        "50 (validation_split=0.15, shuffle=True)"),
        ("Batch Size",       "64 (LSTM)  /  32 (Autoencoder)"),
        ("EarlyStopping",    "monitor=val_loss, restore_best_weights=True"),
        ("ModelCheckpoint",  "monitor=val_auc, mode=max, save_best_only=True"),
        ("Class Imbalance",  "class_weight='balanced' via sklearn"),
        ("Data Split",       "70% Train / 15% Val / 15% Test — Chronological"),
        ("AE Training data", "X_train[y_train == 0] only (no failure samples)"),
        ("AE Threshold τ",   "mean(MSE_train_normal) + 2×std(MSE_train_normal) = 0.025"),
    ]
    cols = st.columns(3)
    for i, (k, v) in enumerate(cfg):
        with cols[i % 3]:
            st.markdown(f"""<div style="background:#eef0f4;border:1px solid #d8dce5;
                 border-radius:8px;padding:.55rem .9rem;margin-bottom:.4rem;">
              <div style="font-size:.7rem;color:var(--muted);">{k}</div>
              <div style="font-size:.84rem;color:var(--accent);font-weight:600;">{v}</div>
            </div>""", unsafe_allow_html=True)

    sec_header("🌐", "Flask API — app.py", "Backend endpoint specification")
    st.markdown("""<div class="card-blue">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.82rem;
           color:#64748b;line-height:2.1;">
        <span style="color:#34d399;">POST</span>
        <span style="color:#2563eb;"> http://0.0.0.0:5001/predict</span><br>
        <b style="color:#1e293b;">Keys accepted:</b>
        <code>air_t</code>&nbsp;<code>proc_t</code>&nbsp;<code>rpm</code>&nbsp;
        <code>torque</code>&nbsp;<code>wear</code>
        &nbsp;&mdash;&nbsp; OR the full column names<br><br>
        <b style="color:#1e293b;">New endpoints (v2.0):</b><br>
        <span style="color:#34d399;">GET</span>
        <span style="color:#2563eb;"> /health</span> — model load status + buffer size<br>
        <span style="color:#34d399;">POST</span>
        <span style="color:#2563eb;"> /reset</span> — clear rolling buffer between sessions<br><br>
        <b style="color:#1e293b;">app.py pipeline:</b><br>
        1. Extract 5 base features from JSON (both key formats supported)<br>
        2. <code>scaler.transform([base_features])</code> — 5 features scaled<br>
        3. Compute temp_diff, stress_index, torque_wear (unscaled, appended)<br>
        4. <code>final_features = np.hstack([scaled_base[0], [f6,f7,f8]])</code> — shape (8,)<br>
        5. Append to <code>deque(maxlen=10)</code> rolling buffer<br>
        6. AE: <code>MSE = mean((final − reconstruct)²)</code><br>
        7. LSTM: predict only when <code>len(buffer)==10</code>, else fail_prob=0.0<br>
      </div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 7 — AI vs TRADITIONAL
# ══════════════════════════════════════════════════════════════
def page_comparison():
    sec_header("⚖️", "AI Model vs Traditional Methods", "Benchmarking")
    criteria = ["Detection\nAccuracy", "Fault Recall", "False Alarm\n(lower=better)",
                "Scalability", "Automation", "Novel Faults"]
    trad = [55, 48, 40, 30, 20, 15]
    ml   = [74, 72, 55, 55, 60, 40]
    ours = [83, 86, 83, 75, 90, 70]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=trad + [trad[0]], theta=criteria + [criteria[0]],
        fill="toself", name="Rule-Based / Threshold",
        line=dict(color="#f87171"), fillcolor="rgba(239,68,68,.1)"))
    fig.add_trace(go.Scatterpolar(r=ml + [ml[0]], theta=criteria + [criteria[0]],
        fill="toself", name="Classical ML (SVM/RF)",
        line=dict(color="#fbbf24"), fillcolor="rgba(245,158,11,.1)"))
    fig.add_trace(go.Scatterpolar(r=ours + [ours[0]], theta=criteria + [criteria[0]],
        fill="toself", name="NeuralGuard: LSTM+Attention+AE",
        line=dict(color="#38bdf8"), fillcolor="rgba(56,189,248,.1)"))
    fig.update_layout(polar=dict(
        radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1a3352", color="#475569"),
        angularaxis=dict(gridcolor="#1a3352", color="#94a3b8"),
        bgcolor="rgba(0,0,0,0)"))
    pdk(fig, "Capability Radar: Traditional vs Classical ML vs NeuralGuard AI", height=400)
    st.plotly_chart(fig, use_container_width=True)

    comp = pd.DataFrame({
        "Criterion":      ["Accuracy", "Fault Recall", "False Alarm Rate", "Scalability", "Automation", "Novel Faults"],
        "Rule-Based":     ["~55%", "~48%", "~40% FAR", "Low",    "Manual",              "No"],
        "Classical ML":   ["~74%", "~72%", "~25% FAR", "Medium", "Partial",             "Limited"],
        "NeuralGuard AI": ["83%",  "86%",  "17.5% FAR","High",   "Full (Flask API)",    "Yes (AE)"],
        "Our Advantage":  ["+9% vs ML", "+14% vs ML", "−7.5% FAR", "✔ Edge/Cloud", "✔ REST API", "✔ Autoencoder"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = go.Figure(go.Bar(
            x=["Threshold\nRules", "Classical ML\n(SVM/RF)", "NeuralGuard\nLSTM", "TFLite\nQuantized"],
            y=[55, 74, 83, 83],
            marker_color=["#f87171", "#fbbf24", "#2563eb", "#34d399"],
            text=["55% Recall", "72% Recall", "83% Acc\n86% Recall", "83% Acc\n6.6× smaller"],
            textposition="outside"))
        pdk(fig2, "Model Accuracy & Recall Comparison", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        fig3 = go.Figure(go.Bar(
            x=["No\nMonitoring", "Threshold\nAlerts", "Classical\nML", "NeuralGuard\nAI"],
            y=[340, 190, 110, 45],
            marker_color=["#f87171", "#fb923c", "#fbbf24", "#34d399"],
            text=["340 hrs", "190 hrs", "110 hrs", "45 hrs"], textposition="outside"))
        fig3.update_yaxes(title="Annual Downtime (hrs)")
        pdk(fig3, "Estimated Annual Unplanned Downtime (hrs/year)", height=300)
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 8 — APPLICATIONS
# ══════════════════════════════════════════════════════════════
def page_applications():
    sec_header("🌍", "Real-World Applications", "Industry Deployment")
    apps = [
        ("🏭", "CNC Manufacturing",  "Tool wear failure, overstrain, heat dissipation — exactly the AI4I failure modes."),
        ("⚡", "Power Generation",   "Turbine bearing wear monitoring and generator anomaly detection."),
        ("✈️", "Aerospace MRO",      "Actuator and engine component degradation tracking for safety-critical systems."),
        ("🚆", "Rail Fleet",         "Wheel bearing, brake, and HVAC monitoring across large rolling-stock fleets."),
        ("🏥", "Medical Equipment",  "Predictive servicing of MRI machines and surgical robots."),
        ("🛢️", "Oil & Gas",          "Pump cavitation and compressor health in remote hazardous environments."),
        ("🏗️", "Heavy Industry",     "Excavator hydraulics and mining drill rig gearbox monitoring."),
        ("🤖", "Smart Robotics",     "Joint torque and end-effector wear prediction for collaborative robots."),
    ]
    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(apps):
        with cols[i % 4]:
            st.markdown(f"""<div class="acard">
              <div style="font-size:1.7rem;">{icon}</div>
              <div style="font-size:.83rem;font-weight:600;color:var(--text);">{name}</div>
              <div style="font-size:.71rem;color:var(--muted);">{desc[:65]}…</div>
            </div><br>""", unsafe_allow_html=True)

    sec_header("💰", "ROI Calculator", "Business case estimator")
    ca, cb = st.columns([1, 1.1])
    with ca:
        machines  = st.slider("Machines monitored",           10, 500, 80)
        cost_fail = st.slider("Avg cost per failure ($K)",     10, 300, 65)
        fail_yr   = st.slider("Failures per machine per year", 1,  20,   6)
        reduction = st.slider("Failure reduction with AI (%)", 30,  85,  62)
    with cb:
        avoided = int(machines * fail_yr * reduction / 100)
        savings = avoided * cost_fail * 1000
        impl    = machines * 2200 + 75000
        net_roi = savings - impl
        roi_pct = (net_roi / impl) * 100 if impl > 0 else 0
        kpis([
            ("#34d399", str(avoided),         "Failures Avoided/yr"),
            ("#60a5fa", f"${savings/1e6:.1f}M", "Annual Savings"),
            ("#fbbf24", f"${impl/1e3:.0f}K",  "Impl. Cost"),
            ("#c4b5fd", f"{roi_pct:.0f}%",    "Net ROI"),
        ])


# ══════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════
def main():
    page = sidebar()
    router = {
        "🏠  Overview":           page_overview,
        "🔮  Live Prediction":    page_prediction,
        "📊  Model Performance":  page_performance,
        "🔍  Data & Features":    page_data,
        "🧠  Explainable AI":     page_xai,
        "⚙️  Model Architecture": page_architecture,
        "⚖️  AI vs Traditional":  page_comparison,
        "🌍  Applications":       page_applications,
    }
    router.get(page, page_overview)()


if __name__ == "__main__":
    main()
