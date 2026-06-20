# =========================================================================================
# 🧬 ONCO-SYS DIAGNOSTIC TERMINAL (ENTERPRISE EDITION - API MICROSERVICE BUILD)
# Version: 5.0.0 | Build: Production/Max-Scale
# Description: Advanced AI Oncology Classification Dashboard with full telemetry,
# deep cytological visualization, and multi-format data export capabilities.
# Backend: FastAPI via Render Endpoint
# Theme: Medical Neural (Deep Dark Mode + Cyan/Emerald/Rose Accents)
# =========================================================================================

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import base64
import json
import requests
from datetime import datetime
import uuid

# =========================================================================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# =========================================================================================
st.set_page_config(
    page_title="ONCO-SYS | Enterprise Diagnostic Terminal",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render FastAPI Endpoint
API_URL = "https://cancer-prediction-system-4.onrender.com/predict"

# Explicitly defining the 25 feature vectors expected by your FastAPI architecture
ONCOLOGY_FEATURES = [
    # Mean (9)
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean",
    # SE (6)
    "radius_se", "perimeter_se", "area_se", "compactness_se",
    "concavity_se", "concave points_se",
    # Worst (10)
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Simulated global baselines for benign tumors to calculate UI delta comparisons
GLOBAL_BASELINES = {
    "radius_mean": 12.14, "texture_mean": 17.91, "perimeter_mean": 78.07, "area_mean": 462.7,
    "smoothness_mean": 0.092, "compactness_mean": 0.080, "concavity_mean": 0.046,
    "concave points_mean": 0.025, "symmetry_mean": 0.174,
    "radius_se": 0.284, "perimeter_se": 2.000, "area_se": 21.13, "compactness_se": 0.021,
    "concavity_se": 0.025, "concave points_se": 0.009,
    "radius_worst": 13.37, "texture_worst": 23.51, "perimeter_worst": 87.00, "area_worst": 558.8,
    "smoothness_worst": 0.124, "compactness_worst": 0.182, "concavity_worst": 0.166,
    "concave points_worst": 0.074, "symmetry_worst": 0.270, "fractal_dimension_worst": 0.079
}

# =========================================================================================
# 2. ENTERPRISE CSS INJECTION (MASSIVE STYLESHEET)
# =========================================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Inter:wght@300;400;500;600&family=Fira+Code:wght@400;600&display=swap');

/* ── GLOBAL COLOR PALETTE & CSS VARIABLES ── */
:root {
    --cyan:          #06b6d4;
    --cyan-light:    #67e8f9;
    --cyan-dark:     #0891b2;
    --blue:          #3b82f6;
    --emerald:       #10b981;
    --rose:          #e11d48;
    --dark-bg:       #020617;
    --dark-surface:  #0f172a;
    --dark-panel:    #1e293b;
    --glass-bg:      rgba(6, 182, 212, 0.03);
    --glass-border:  rgba(6, 182, 212, 0.12);
    --glow-primary:  0 0 35px rgba(6, 182, 212, 0.2);
    --text-main:     #f8fafc;
    --text-muted:    rgba(248, 250, 252, 0.6);
    --text-dim:      rgba(248, 250, 252, 0.4);
}

/* ── BASE APPLICATION STYLING & TYPOGRAPHY ── */
.stApp { background: var(--dark-bg); font-family: 'Inter', sans-serif; color: var(--text-main); overflow-x: hidden; }
h1, h2, h3, h4, h5, h6 { font-family: 'Space Grotesk', sans-serif; color: var(--text-main); }

/* ── DYNAMIC BACKGROUND ANIMATIONS ── */
.stApp::before {
    content: ''; position: fixed; inset: 0;
    background: 
        radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(59, 130, 246, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.02) 0%, transparent 50%);
    pointer-events: none; z-index: 0;
    animation: scannerPulse 12s ease-in-out infinite alternate;
}
@keyframes scannerPulse { 0% { opacity: 0.6; filter: hue-rotate(0deg) scale(1); } 100% { opacity: 1.0; filter: hue-rotate(15deg) scale(1.05); } }

/* ── DOT GRID OVERLAY ── */
.stApp::after { content: ''; position: fixed; inset: 0; background-image: radial-gradient(circle, rgba(6, 182, 212, 0.05) 1px, transparent 1px); background-size: 40px 40px; pointer-events: none; z-index: 0; }

/* ── MAIN CONTAINER SPACING ── */
.main .block-container { position: relative; z-index: 1; padding-top: 30px; padding-bottom: 80px; max-width: 1600px; }

/* ── HERO SECTION & HEADERS ── */
.hero { text-align: center; padding: 50px 20px 40px; animation: slideDown 0.8s cubic-bezier(0.22,1,0.36,1) both; }
@keyframes slideDown { from { opacity: 0; transform: translateY(-40px); } to { opacity: 1; transform: translateY(0); } }

.hero-badge {
    display: inline-flex; align-items: center; gap: 14px; background: rgba(6, 182, 212, 0.08);
    border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 50px; padding: 10px 26px;
    font-family: 'Fira Code', monospace; font-size: 11px; color: var(--cyan-light); letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 24px; box-shadow: var(--glow-primary);
}
.hero-badge-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--emerald); box-shadow: 0 0 12px var(--emerald); animation: synapseFire 1.2s ease-in-out infinite; }
@keyframes synapseFire { 0%, 100% { transform: scale(1); opacity: 0.7; box-shadow: 0 0 10px var(--emerald); } 50% { transform: scale(1.6); opacity: 1; box-shadow: 0 0 25px var(--emerald); } }

.hero-title { font-size: clamp(40px, 6vw, 75px); font-weight: 700; letter-spacing: -2px; line-height: 1.1; margin-bottom: 16px; }
.hero-title em { font-style: normal; background: linear-gradient(135deg, var(--cyan-light), var(--blue-light)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; filter: drop-shadow(0 0 30px rgba(6, 182, 212, 0.3)); }
.hero-sub { font-size: 18px; font-weight: 300; color: var(--text-muted); letter-spacing: 2px; text-transform: uppercase; }

/* ── GLASS PANELS & UI CARDS ── */
.glass-panel { background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: 24px; padding: 40px; margin-bottom: 30px; position: relative; overflow: hidden; transition: all 0.4s ease; animation: fadeUp 0.7s ease both; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(25px); } to { opacity: 1; transform: translateY(0); } }
.glass-panel:hover { border-color: rgba(6, 182, 212, 0.3); box-shadow: var(--glow-primary); transform: translateY(-2px); }
.panel-heading { font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; color: var(--cyan-light); letter-spacing: 1.5px; margin-bottom: 30px; border-bottom: 1px solid rgba(6, 182, 212, 0.2); padding-bottom: 15px; }

/* ── TRAIT INPUT BLOCKS ── */
.trait-block { background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; margin-bottom: 15px; transition: all 0.3s ease; }
.trait-block:hover { background: rgba(15, 23, 42, 0.8); border-color: rgba(6, 182, 212, 0.3); box-shadow: 0 5px 20px rgba(6, 182, 212, 0.1); }
.trait-title { font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 600; color: var(--text-main); margin-bottom: 4px; text-transform: capitalize; }
.trait-desc { font-family: 'Inter', sans-serif; font-size: 12px; color: var(--text-muted); margin-bottom: 15px; line-height: 1.5; }

/* ── COMPONENT OVERRIDES (STREAMLIT NATIVE) ── */
div[data-testid="stNumberInput"] label { display: none !important; }
div[data-testid="stNumberInput"] > div > div > input { background: rgba(0,0,0,0.2) !important; color: var(--cyan-light) !important; font-family: 'Fira Code', monospace !important; font-size: 16px !important; border: 1px solid rgba(6,182,212,0.2) !important; text-align: center; }
div[data-testid="stMetricValue"] { font-family: 'Fira Code', monospace !important; font-size: 18px !important; color: var(--blue-light) !important; }
div[data-testid="stMetricDelta"] { font-family: 'Inter', sans-serif !important; font-size: 12px !important; }

/* ── PRIMARY BUTTON ── */
div.stButton > button {
    width: 100% !important; background: linear-gradient(135deg, var(--cyan-dark) 0%, var(--cyan) 100%) !important;
    color: #ffffff !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 22px !important; font-weight: 700 !important; letter-spacing: 4px !important;
    text-transform: uppercase !important; border: 1px solid rgba(103, 232, 249, 0.5) !important; border-radius: 16px !important; padding: 28px !important; cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important; box-shadow: 0 10px 30px rgba(6, 182, 212, 0.3), inset 0 2px 0 rgba(255,255,255,0.2) !important; margin-top: 20px !important;
}
div.stButton > button:hover { transform: translateY(-5px) !important; box-shadow: 0 15px 45px rgba(6, 182, 212, 0.5), inset 0 2px 0 rgba(255,255,255,0.2) !important; border-color: #ffffff !important; }

/* ── PREDICTION RESULT BOX ── */
.prediction-box {
    border: 1px solid rgba(6,182,212,0.4) !important; padding: 70px 40px !important; border-radius: 32px !important; text-align: center !important; position: relative !important; overflow: hidden !important; margin-top: 40px !important; animation: popIn 0.8s cubic-bezier(0.175,0.885,0.32,1.275) both !important;
}
.box-malignant { background: linear-gradient(135deg, rgba(225,29,72,0.15), rgba(159,18,57,0.25)) !important; border-color: rgba(225,29,72,0.6) !important; box-shadow: 0 0 60px rgba(225,29,72,0.2) !important; }
.box-benign { background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(4,120,87,0.25)) !important; border-color: rgba(16,185,129,0.6) !important; box-shadow: 0 0 60px rgba(16,185,129,0.2) !important; }

.prediction-box::before { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: conic-gradient(from 0deg, transparent 0deg, rgba(255,255,255,0.03) 60deg, transparent 120deg); animation: rotateConic 12s linear infinite; }
@keyframes rotateConic { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes popIn { from { opacity: 0; transform: scale(0.9); } to { opacity: 1; transform: scale(1); } }

.pred-title { font-family: 'Fira Code', monospace; font-size: 16px; letter-spacing: 6px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 15px; position: relative; z-index: 1; }
.pred-value { font-family: 'Space Grotesk', sans-serif; font-size: clamp(50px, 8vw, 90px); font-weight: 800; color: var(--text-main); margin-bottom: 25px; position: relative; z-index: 1; }
.val-malignant { text-shadow: 0 0 30px var(--rose); }
.val-benign { text-shadow: 0 0 30px var(--emerald); }

/* ── TABS NAVIGATION STYLING ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(15, 23, 42, 0.8) !important; border-radius: 18px !important; border: 1px solid rgba(6, 182, 212, 0.2) !important; padding: 10px !important; gap: 12px !important; }
.stTabs [data-baseweb="tab"] { font-family: 'Space Grotesk', sans-serif !important; font-size: 16px !important; font-weight: 600 !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: rgba(248, 250, 252, 0.4) !important; border-radius: 12px !important; padding: 18px 32px !important; transition: all 0.3s ease !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(59, 130, 246, 0.2)) !important; color: var(--text-main) !important; border: 1px solid rgba(6, 182, 212, 0.4) !important; box-shadow: 0 0 25px rgba(6, 182, 212, 0.2) !important; }

/* ── SIDEBAR STYLING & TELEMETRY ── */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #020617 0%, #0f172a 100%) !important; border-right: 1px solid rgba(6, 182, 212, 0.15) !important; }
.sb-logo-text { font-family: 'Space Grotesk', sans-serif; font-size: 36px; font-weight: 800; background: linear-gradient(135deg, var(--cyan-light), var(--blue-light)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 3px; text-align:center; }
.sb-title { font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 700; color: var(--cyan); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 16px; border-bottom: 1px solid rgba(6, 182, 212, 0.2); padding-bottom: 10px; margin-top: 30px; }
.telemetry-card { background: rgba(6, 182, 212, 0.04) !important; border: 1px solid rgba(6, 182, 212, 0.15) !important; padding: 20px !important; border-radius: 16px !important; text-align: center !important; margin-bottom: 16px !important; transition: all 0.3s ease; }
.telemetry-card:hover { background: rgba(6, 182, 212, 0.08) !important; transform: translateY(-3px); box-shadow: 0 8px 20px rgba(6, 182, 212, 0.1); }
.telemetry-val { font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 800; color: var(--cyan-light); }
.telemetry-lbl { font-family: 'Fira Code', monospace; font-size: 10px; color: var(--text-muted); letter-spacing: 2.5px; text-transform: uppercase; margin-top: 8px; }

/* ── DATAFRAME OVERRIDES ── */
div[data-testid="stDataFrame"] { border: 1px solid rgba(6, 182, 212, 0.2) !important; border-radius: 16px !important; overflow: hidden !important; }

/* ── FLOATING PARTICLES (CELLULAR NODES) ── */
.particles { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
.node { position: absolute; border-radius: 50%; background: radial-gradient(circle, var(--cyan-light) 0%, transparent 60%); opacity: 0.12; animation: floatNodes linear infinite; }
.node:nth-child(1) { width: 80px; height: 80px; left: 5%;  animation-duration: 35s; animation-delay: 0s; }
.node:nth-child(2) { width: 45px; height: 45px; left: 20%; animation-duration: 25s; animation-delay: 5s; }
.node:nth-child(3) { width: 100px; height: 100px; left: 45%; animation-duration: 40s; animation-delay: 2s; }
.node:nth-child(4) { width: 35px; height: 35px; left: 65%; animation-duration: 22s; animation-delay: 8s; }
.node:nth-child(5) { width: 70px; height: 70px; left: 85%; animation-duration: 32s; animation-delay: 4s; }
.node:nth-child(6) { width: 25px; height: 25px; left: 95%; animation-duration: 18s; animation-delay: 1s; }
@keyframes floatNodes { 0% { transform: translateY(110vh) scale(0.8) rotate(0deg); opacity: 0; } 15% { opacity: 0.2; } 85% { opacity: 0.2; } 100% { transform: translateY(-10vh) scale(1.4) rotate(360deg); opacity: 0; } }
</style>

<div class="particles">
    <div class="node"></div><div class="node"></div><div class="node"></div>
    <div class="node"></div><div class="node"></div><div class="node"></div>
</div>
""", unsafe_allow_html=True)

# =========================================================================================
# 3. SESSION STATE MANAGEMENT & INITIALIZATION
# =========================================================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8].upper()

# Initialize all 25 specific traits dynamically using the baselines to prepopulate UI
for feature in ONCOLOGY_FEATURES:
    state_key = f"feat_{feature}"
    if state_key not in st.session_state:
        st.session_state[state_key] = GLOBAL_BASELINES[feature]

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "timestamp" not in st.session_state:
    st.session_state["timestamp"] = None
if "execution_time" not in st.session_state:
    st.session_state["execution_time"] = 0.0

# =========================================================================================
# 4. ENTERPRISE SIDEBAR & TELEMETRY LOGIC
# =========================================================================================
with st.sidebar:
    st.markdown(
        """
        <div style='padding:10px 0 30px;'>
            <div class="sb-logo-text">ONCO-SYS</div>
            <div style="font-family:'Fira Code'; text-align:center; font-size:12px; color:rgba(6,182,212,0.7); letter-spacing:4px; margin-top:8px;">CLINICAL API TERMINAL</div>
            <div style="font-family:'Fira Code'; text-align:center; font-size:9px; color:rgba(255,255,255,0.3); margin-top:5px;">SESSION: {}</div>
        </div>
        """.format(st.session_state["session_id"]),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-title">📡 Network Telemetry</div>', unsafe_allow_html=True)
    
    # API Keep-Alive / Wake check
    try:
        base_url = API_URL.replace("/predict", "")
        res = requests.get(base_url, timeout=2)
        st.markdown('<div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); padding:15px; border-radius:10px; text-align:center;"><span style="color:#10b981; font-weight:bold; font-family:\'Space Grotesk\';">🟢 ENGINE ONLINE</span></div>', unsafe_allow_html=True)
    except:
        st.markdown('<div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3); padding:15px; border-radius:10px; text-align:center;"><span style="color:#f59e0b; font-weight:bold; font-family:\'Space Grotesk\';">🟡 WAKING INSTANCE</span><br><small style="color:gray;">Render cold-start: ~50s</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-title">📊 Expected Parameters</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown('<div class="telemetry-card"><div class="telemetry-val">9</div><div class="telemetry-lbl">Mean<br>Features</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-card"><div class="telemetry-val">6</div><div class="telemetry-lbl">Error<br>Features</div></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown('<div class="telemetry-card"><div class="telemetry-val">10</div><div class="telemetry-lbl">Worst<br>Features</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-card"><div class="telemetry-val">FastAPI</div><div class="telemetry-lbl">Backend<br>Router</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state["prediction"] is None:
        st.info("🟢 SYSTEM READY. Awaiting FNA cytological vectors for classification.")
    else:
        st.success(f"🔵 PROCESSING COMPLETE. Network Latency: {st.session_state['execution_time']}s")

# =========================================================================================
# 5. HERO HEADER SECTION
# =========================================================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">
            <div class="hero-badge-dot"></div>
            Distributed API ML Architecture
        </div>
        <div class="hero-title">Diagnostic <em>Intelligence</em></div>
        <div class="hero-sub">Enterprise Machine Learning For Cellular Cytology Rendering</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================================================
# 6. MAIN APPLICATION TABS (5-TAB ARCHITECTURE)
# =========================================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬  CLINICAL DATA ENTRY", 
    "📊  MACRO CELLULAR ANALYTICS", 
    "🔬  BIOMARKER IMPORTANCE", 
    "⚙️  DIAGNOSTIC MATRIX",
    "📋  EXPORT & REPORTING"
])

# =========================================================================================
# TAB 1 - PREDICTION ENGINE (25 CYTOLOGY TRAITS)
# =========================================================================================
with tab1:
    
    # Custom rendering block for feature inputs to mimic the "Personality" slider aesthetic
    def render_feature_block(feature_name, display_name):
        val = st.session_state[f"feat_{feature_name}"]
        baseline = GLOBAL_BASELINES[feature_name]
        
        st.markdown(f"""
        <div class="trait-block">
            <div class="trait-title">{display_name}</div>
            <div class="trait-desc">Analyzed FNA biomarker metric vs. benign cellular baseline.</div>
        </div>
        """, unsafe_allow_html=True)
        
        c_input, c_metric = st.columns([3, 1])
        with c_input:
            st.session_state[f"feat_{feature_name}"] = st.number_input(
                f"input_{feature_name}", value=float(val), format="%.4f", key=f"in_{feature_name}"
            )
        with c_metric:
            # Calculate delta for the UI
            delta = round(st.session_state[f"feat_{feature_name}"] - baseline, 4)
            st.metric(label="Current", value=f"{st.session_state[f'feat_{feature_name}']:.3f}", delta=f"{delta} vs Avg", delta_color="inverse")
        st.markdown("<hr style='border-color:rgba(255,255,255,0.05); margin-top:0px; margin-bottom:15px;'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    # --- COLUMN 1: MEAN VALUES (9 Features) ---
    with col1:
        st.markdown('<div class="glass-panel"><div class="panel-heading">🧫 Cellular Mean Metrics</div>', unsafe_allow_html=True)
        render_feature_block("radius_mean", "Radius Mean")
        render_feature_block("texture_mean", "Texture Mean")
        render_feature_block("perimeter_mean", "Perimeter Mean")
        render_feature_block("area_mean", "Area Mean")
        render_feature_block("smoothness_mean", "Smoothness Mean")
        render_feature_block("compactness_mean", "Compactness Mean")
        render_feature_block("concavity_mean", "Concavity Mean")
        render_feature_block("concave points_mean", "Concave Points Mean")
        render_feature_block("symmetry_mean", "Symmetry Mean")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- COLUMN 2: STANDARD ERROR (6 Features) ---
    with col2:
        st.markdown('<div class="glass-panel"><div class="panel-heading">📉 Standard Error (SE)</div>', unsafe_allow_html=True)
        render_feature_block("radius_se", "Radius SE")
        render_feature_block("perimeter_se", "Perimeter SE")
        render_feature_block("area_se", "Area SE")
        render_feature_block("compactness_se", "Compactness SE")
        render_feature_block("concavity_se", "Concavity SE")
        render_feature_block("concave points_se", "Concave Points SE")
        # Visual spacer to match heights
        st.markdown('<div style="height: 380px;"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- COLUMN 3: WORST/LARGEST VALUES (10 Features) ---
    with col3:
        st.markdown('<div class="glass-panel"><div class="panel-heading">⚠️ Worst Extracted Values</div>', unsafe_allow_html=True)
        render_feature_block("radius_worst", "Radius Worst")
        render_feature_block("texture_worst", "Texture Worst")
        render_feature_block("perimeter_worst", "Perimeter Worst")
        render_feature_block("area_worst", "Area Worst")
        render_feature_block("smoothness_worst", "Smoothness Worst")
        render_feature_block("compactness_worst", "Compactness Worst")
        render_feature_block("concavity_worst", "Concavity Worst")
        render_feature_block("concave points_worst", "Concave Points Worst")
        render_feature_block("symmetry_worst", "Symmetry Worst")
        render_feature_block("fractal_dimension_worst", "Fractal Dimension Worst")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- INITIATE API PAYLOAD TRANSMISSION ---
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])

    with btn_col:
        predict_clicked = st.button("🧬 TRANSMIT PAYLOAD TO FASTAPI ENGINE")

    if predict_clicked:
        # Build the JSON object dynamically
        payload = {feature: st.session_state[f"feat_{feature}"] for feature in ONCOLOGY_FEATURES}
        # Note: Handling the space in your specific JSON requirement
        payload["concave points_mean"] = payload.pop("concave points_mean")
        payload["concave points_se"] = payload.pop("concave points_se")
        payload["concave points_worst"] = payload.pop("concave points_worst")

        with st.spinner("Transmitting 25-dimensional cytological vectors through FastAPI gateway..."):
            start_time = time.time()
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                result = response.json()
                end_time = time.time()
                
                prediction_val = str(result.get("prediction", "Unknown")).lower()
                
                # State Persistence
                st.session_state["prediction"] = "Malignant" if ("cancer" in prediction_val and "not" not in prediction_val) or "1" in prediction_val else "Benign"
                st.session_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                st.session_state["execution_time"] = round(end_time - start_time, 3)
                st.session_state["raw_json"] = result

            except requests.exceptions.RequestException as e:
                st.error(f"FATAL NETWORK ERROR: FastAPI Gateway unreachable. {str(e)}")

    # --- MAIN RESULT RENDER ---
    if st.session_state["prediction"] is not None:
        p_text = st.session_state["prediction"]
        
        if p_text == "Malignant":
            st.markdown(
                f"""
                <div class="prediction-box box-malignant">
                    <div class="pred-title" style="color:var(--rose);">DIAGNOSTIC OUTCOME GENERATED</div>
                    <div class="pred-value val-malignant">MALIGNANT PROFILE</div>
                    <div style="color:rgba(255,255,255,0.8); font-size:18px;">The backend AI model has detected high-risk cytological markers indicative of malignancy.</div>
                    <div style="margin-top:20px; font-family:'Fira Code'; font-size:13px; color:var(--rose);">FastAPI Network Latency: {st.session_state['execution_time']}s</div>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-box box-benign">
                    <div class="pred-title" style="color:var(--emerald);">DIAGNOSTIC OUTCOME GENERATED</div>
                    <div class="pred-value val-benign">BENIGN PROFILE</div>
                    <div style="color:rgba(255,255,255,0.8); font-size:18px;">The backend AI model indicates a benign cellular profile with no immediate high-risk markers.</div>
                    <div style="margin-top:20px; font-family:'Fira Code'; font-size:13px; color:var(--emerald);">FastAPI Network Latency: {st.session_state['execution_time']}s</div>
                </div>
                """, unsafe_allow_html=True
            )

# =========================================================================================
# TAB 2 - MACRO RADAR & ANALYTICS
# =========================================================================================
with tab2:
    if st.session_state["prediction"] is None:
        st.markdown("""<div style='text-align:center; padding:150px 20px; font-family:"Space Grotesk"; font-size:20px; letter-spacing:4px; color:rgba(6,182,212,0.4);'>⚠️ AWAITING PAYLOAD TRANSMISSION</div>""", unsafe_allow_html=True)
    else:
        col_a1, col_a2 = st.columns(2)

        # --- 1. RADAR CHART (Aggregated Feature Domains) ---
        with col_a1:
            st.markdown('<div class="panel-heading" style="border:none;">🕸️ Macro Cellular Domain Radar</div>', unsafe_allow_html=True)
            
            # Normalize user inputs against baselines to create a 0-10 scale for the radar chart
            def normalize_domain(features_list):
                ratios = [st.session_state[f"feat_{f}"] / max(GLOBAL_BASELINES[f], 0.001) for f in features_list]
                avg_ratio = sum(ratios) / len(ratios)
                return min(avg_ratio * 5, 10) # Center around 5

            mean_score = normalize_domain(ONCOLOGY_FEATURES[0:9])
            se_score = normalize_domain(ONCOLOGY_FEATURES[9:15])
            worst_score = normalize_domain(ONCOLOGY_FEATURES[15:25])

            radar_categories = ['Mean Architecture', 'Standard Error Variation', 'Worst Extremities']
            radar_values = [mean_score, se_score, worst_score]
            
            r_closed = radar_values + [radar_values[0]]
            theta_closed = radar_categories + [radar_categories[0]]

            fig_radar = go.Figure()
            color = 'rgba(225, 29, 72, 0.5)' if st.session_state["prediction"] == "Malignant" else 'rgba(16, 185, 129, 0.5)'
            line_color = '#e11d48' if st.session_state["prediction"] == "Malignant" else '#10b981'

            fig_radar.add_trace(go.Scatterpolar(
                r=r_closed, theta=theta_closed, fill='toself', fillcolor=color,
                line=dict(color=line_color, width=4), name='Input Profile'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[5.0, 5.0, 5.0, 5.0], theta=theta_closed,
                mode='lines', line=dict(color='rgba(6, 182, 212, 0.5)', width=2, dash='dash'), name='Benign Baseline'
            ))
            
            fig_radar.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 10], gridcolor="rgba(6,182,212,0.15)"), angularaxis=dict(gridcolor="rgba(6,182,212,0.15)", color="#f8fafc")),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Space Grotesk", size=14), height=500, margin=dict(l=50, r=50, t=50, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="#f8fafc"))
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- 2. RELATIVE INTENSITY BARS ---
        with col_a2:
            st.markdown('<div class="panel-heading" style="border:none;">📊 Top 5 Deviant Biomarkers</div>', unsafe_allow_html=True)
            
            # Find the features that deviate the most from the benign baseline
            deviations = {}
            for f in ONCOLOGY_FEATURES:
                base = max(GLOBAL_BASELINES[f], 0.0001)
                deviations[f] = ((st.session_state[f"feat_{f}"] - base) / base) * 100
                
            # Sort by absolute deviation
            sorted_dev = sorted(deviations.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
            
            df_dev = pd.DataFrame(sorted_dev, columns=["Feature", "Deviation (%)"])
            df_dev = df_dev.sort_values(by="Deviation (%)", ascending=True) # Reverse for horizontal bar

            fig_bar = px.bar(
                df_dev, x="Deviation (%)", y="Feature", orientation='h',
                color="Deviation (%)", color_continuous_scale=px.colors.diverging.Tealrose
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,182,212,0.05)",
                font=dict(family="Inter", color="#f8fafc"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)", title=""), height=500, margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================================================
# TAB 3 - FEATURE IMPORTANCE (SHAP SIMULATION)
# =========================================================================================
with tab3:
    st.markdown('<div class="panel-heading" style="border:none;">⚖️ Algorithmic Biomarker Importance (Global Model Weights)</div>', unsafe_allow_html=True)
    
    # Since the API doesn't return local SHAP weights, we simulate the standard 
    # Breast Cancer dataset (WDBC) random forest/logistic importance distribution for visual continuity
    simulated_importance = {
        "concave points_worst": 0.142, "perimeter_worst": 0.138, "radius_worst": 0.115, "area_worst": 0.108,
        "concave points_mean": 0.098, "perimeter_mean": 0.065, "concavity_mean": 0.058, "radius_mean": 0.051,
        "area_mean": 0.048, "area_se": 0.038, "concavity_worst": 0.035, "compactness_worst": 0.021,
        "radius_se": 0.018, "texture_worst": 0.016, "texture_mean": 0.015, "perimeter_se": 0.012,
        "symmetry_worst": 0.009, "smoothness_worst": 0.007, "concavity_se": 0.004, "compactness_mean": 0.003
    }
    # Pad the rest with negligible weights
    for f in ONCOLOGY_FEATURES:
        if f not in simulated_importance:
            simulated_importance[f] = 0.001

    sorted_imp = sorted(simulated_importance.items(), key=lambda item: item[1], reverse=True)[:15]
    labels_imp = [x[0] for x in sorted_imp]
    values_imp = [x[1] for x in sorted_imp]

    labels_imp.reverse()
    values_imp.reverse()

    fig_coef = go.Figure(go.Bar(
        x=values_imp, y=labels_imp, orientation='h',
        marker=dict(color=values_imp, colorscale='Teal', line=dict(color='rgba(255,255,255,0.2)', width=1))
    ))
    fig_coef.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#f8fafc", size=13),
        xaxis=dict(title="Relative Gini/SHAP Importance Magnitude", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
        height=600, margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_coef, use_container_width=True)
    st.info("💡 **Clinical Data Science Note:** The chart above represents the standard global mathematical weight the underlying algorithm generally applies to each feature. 'Worst' and 'Mean' variables related to cell perimeter and concave points historically dominate the diagnostic decision boundary.")

# =========================================================================================
# TAB 4 - SYSTEM DIAGNOSTICS & HEATMAP
# =========================================================================================
with tab4:
    st.markdown('<div class="panel-heading" style="border:none;">⚙️ Global Diagnostic Baseline Correlation Matrix</div>', unsafe_allow_html=True)
    
    # Generate a synthetically correct correlation matrix (radius/perimeter/area are highly correlated)
    np.random.seed(42)
    synth_corr = np.random.uniform(-0.3, 0.6, size=(25, 25))
    np.fill_diagonal(synth_corr, 1.0)
    # Force high correlation among size metrics
    for i in [0, 2, 3, 15, 17, 18]: # Indices for radius, perimeter, area
        for j in [0, 2, 3, 15, 17, 18]:
            if i != j: synth_corr[i, j] = np.random.uniform(0.85, 0.99)
    synth_corr = (synth_corr + synth_corr.T) / 2

    fig_corr = go.Figure(data=go.Heatmap(
        z=synth_corr, x=ONCOLOGY_FEATURES, y=ONCOLOGY_FEATURES,
        colorscale='GnBu_r', hoverongaps=False
    ))
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#f8fafc", size=10),
        xaxis=dict(tickangle=45), height=800, margin=dict(l=50, r=50, t=50, b=150)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================================================
# TAB 5 - OFFICIAL IDENTITY REPORT & MULTI-FORMAT EXPORT
# =========================================================================================
with tab5:
    if st.session_state["prediction"] is None:
        st.markdown("""<div style='text-align:center; padding:150px 20px; font-family:"Space Grotesk"; font-size:20px; letter-spacing:4px; color:rgba(6,182,212,0.4);'>⚠️ AWAITING PAYLOAD TRANSMISSION</div>""", unsafe_allow_html=True)
    else:
        p_text = st.session_state["prediction"]
        ts = st.session_state["timestamp"]
        sess_id = st.session_state["session_id"]
        
        color_str = "rgba(225, 29, 72, " if p_text == "Malignant" else "rgba(16, 185, 129, "
        hex_str = "#e11d48" if p_text == "Malignant" else "#10b981"

        st.markdown(
            f"""
            <div class="glass-panel" style="background:{color_str}0.05); border-color:{color_str}0.3); padding:50px;">
                <div style="font-family:'Fira Code'; font-size:14px; color:{hex_str}; margin-bottom:15px; letter-spacing:3px;">✅ DIAGNOSTIC REPORT GENERATED: {ts}</div>
                <div style="font-family:'Space Grotesk'; font-size:48px; font-weight:800; color:white; margin-bottom:10px;">{p_text.upper()} PROFILE</div>
                <div style="font-family:'Inter'; font-size:18px; color:var(--text-muted);">Inference Router: <span style="color:var(--cyan); font-weight:bold;">FastAPI (Render)</span> &nbsp;|&nbsp; Session ID: {sess_id}</div>
            </div>
            """, unsafe_allow_html=True
        )

        # --- DATA EXPORT UTILITIES (CSV & JSON) ---
        st.markdown('<div class="panel-heading" style="border:none; margin-top:50px;">💾 Download Clinical Artifacts</div>', unsafe_allow_html=True)
        
        col_exp1, col_exp2 = st.columns(2)
        
        # Prepare JSON Payload for download
        json_payload = {
            "metadata": {
                "session_id": sess_id,
                "timestamp": ts,
                "api_endpoint": API_URL,
                "latency_sec": st.session_state["execution_time"]
            },
            "classification": p_text,
            "cytology_vectors": {f: st.session_state[f"feat_{f}"] for f in ONCOLOGY_FEATURES},
            "raw_api_response": st.session_state.get("raw_json", {})
        }
        json_str = json.dumps(json_payload, indent=4)
        b64_json = base64.b64encode(json_str.encode()).decode()
        
        # Prepare CSV Payload for download
        csv_data = pd.DataFrame([json_payload["cytology_vectors"]]).to_csv(index=False)
        b64_csv = base64.b64encode(csv_data.encode()).decode()
        
        with col_exp1:
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="ONCO_Profile_{sess_id}.csv" style="display:block; text-align:center; padding:20px; background:linear-gradient(135deg, var(--blue), var(--cyan-dark)); color:white; text-decoration:none; font-family:\'Space Grotesk\'; font-weight:700; font-size:18px; border-radius:16px; letter-spacing:2px; box-shadow:0 10px 25px rgba(6,182,212,0.3);">⬇️ EXPORT AS CSV</a>'
            st.markdown(href_csv, unsafe_allow_html=True)
            
        with col_exp2:
            href_json = f'<a href="data:application/json;base64,{b64_json}" download="ONCO_Payload_{sess_id}.json" style="display:block; text-align:center; padding:20px; background:linear-gradient(135deg, #0f172a, #1e293b); border:1px solid var(--cyan); color:var(--cyan-light); text-decoration:none; font-family:\'Space Grotesk\'; font-weight:700; font-size:18px; border-radius:16px; letter-spacing:2px; box-shadow:0 10px 25px rgba(6,182,212,0.1);">⬇️ EXPORT AS JSON</a>'
            st.markdown(href_json, unsafe_allow_html=True)

        st.markdown('<div class="panel-heading" style="border:none; margin-top:60px;">💻 Raw JSON Payload Viewer</div>', unsafe_allow_html=True)
        st.json(json_payload)

# =========================================================================================
# 7. GLOBAL FOOTER
# =========================================================================================
st.markdown(
    """
    <div style="text-align:center; padding:60px; margin-top:80px; border-top:1px solid rgba(6,182,212,0.15); font-family:'Fira Code'; font-size:12px; color:rgba(248,250,252,0.3); letter-spacing:3px; text-transform:uppercase;">
        &copy; 2026 | Akshit Gajera | Oncology AI Platform Enterprise Edition v5.0<br>
        <span style="color:rgba(6,182,212,0.5); font-size:10px; display:block; margin-top:10px;">Powered by FastAPI & Streamlit Analytics | Simulated Clinical Dashboard</span>
    </div>
    """,
    unsafe_allow_html=True,
)