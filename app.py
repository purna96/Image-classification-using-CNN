"""
COVID-19 Chest X-ray Detection — Streamlit App
================================================
Premium UI: dark theme, Plotly gauge, risk badges, animated cards.
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 X-Ray Detector | AI Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Injection ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:   #0d1117;
    --bg-card:      #161b22;
    --bg-card2:     #1c2128;
    --accent-blue:  #58a6ff;
    --accent-green: #3fb950;
    --accent-red:   #f85149;
    --accent-amber: #d29922;
    --accent-purple:#bc8cff;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --border:       #30363d;
    --glow-blue:    0 0 20px rgba(88,166,255,0.15);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Main container padding ── */
.main .block-container { padding: 2rem 3rem; max-width: 1400px; }

/* ── Hide default Streamlit elements ── */
footer, header { visibility: hidden; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #1a2332 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(88,166,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(63,185,80,0.15);
    border: 1px solid rgba(63,185,80,0.4);
    color: #3fb950;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Stat cards ── */
.stats-row { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 160px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover { transform: translateY(-2px); box-shadow: var(--glow-blue); }
.stat-value { font-size: 1.8rem; font-weight: 700; color: var(--accent-blue); }
.stat-label { font-size: 0.78rem; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Upload zone ── */
.upload-zone {
    background: var(--bg-card);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.3s;
    margin-bottom: 1.5rem;
}
.upload-zone:hover { border-color: var(--accent-blue); }

/* ── Result card ── */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-covid {
    background: linear-gradient(135deg, rgba(248,81,73,0.1), rgba(248,81,73,0.03));
    border: 1px solid rgba(248,81,73,0.4);
}
.result-normal {
    background: linear-gradient(135deg, rgba(63,185,80,0.1), rgba(63,185,80,0.03));
    border: 1px solid rgba(63,185,80,0.4);
}
.result-pneumonia {
    background: linear-gradient(135deg, rgba(210,153,34,0.1), rgba(210,153,34,0.03));
    border: 1px solid rgba(210,153,34,0.4);
}
.result-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.result-class {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.1;
}
.result-confidence {
    font-size: 1rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1rem;
}
.risk-high   { background: rgba(248,81,73,0.2);  color: #f85149; border: 1px solid rgba(248,81,73,0.5); }
.risk-low    { background: rgba(63,185,80,0.2);  color: #3fb950; border: 1px solid rgba(63,185,80,0.5); }
.risk-medium { background: rgba(210,153,34,0.2); color: #d29922; border: 1px solid rgba(210,153,34,0.5); }

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    border-left: 3px solid var(--accent-blue);
    padding-left: 0.75rem;
    margin-bottom: 1rem;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(210,153,34,0.08);
    border: 1px solid rgba(210,153,34,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #d29922;
    margin-top: 2rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: var(--text-muted) !important; font-size: 0.88rem; }

/* ── Metric style override ── */
[data-testid="metric-container"] {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #58a6ff, #bc8cff) !important; border-radius: 4px !important; }

/* ── Plotly chart background ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE   = (128, 128)
MODEL_PATH   = 'covid_detection_model.keras'
CLASS_NAMES  = ['Covid', 'Normal', 'Viral Pneumonia']

CLASS_CONFIG = {
    'Covid': {
        'icon': '🦠',
        'css': 'result-covid',
        'risk_css': 'risk-high',
        'risk': '⚠️ HIGH RISK',
        'color': '#f85149',
        'advice': 'Immediate medical attention is strongly recommended. Isolate and contact healthcare professionals.',
    },
    'Normal': {
        'icon': '✅',
        'css': 'result-normal',
        'risk_css': 'risk-low',
        'risk': '✅ LOW RISK',
        'color': '#3fb950',
        'advice': 'No abnormalities detected. Maintain regular health monitoring and hygiene practices.',
    },
    'Viral Pneumonia': {
        'icon': '🫁',
        'css': 'result-pneumonia',
        'risk_css': 'risk-medium',
        'risk': '⚡ MEDIUM RISK',
        'color': '#d29922',
        'advice': 'Lung abnormality detected. Consult a pulmonologist for further evaluation.',
    },
}

# ─── Model Loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)

# ─── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ─── Plotly Probability Bar Chart ───────────────────────────────────────────────
def probability_chart(probs: np.ndarray) -> go.Figure:
    colors = [CLASS_CONFIG[c]['color'] for c in CLASS_NAMES]
    fig = go.Figure(go.Bar(
        x=[f"{p*100:.1f}%" for p in probs],
        y=CLASS_NAMES,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{p*100:.2f}%" for p in probs],
        textposition='outside',
        textfont=dict(color='#e6edf3', size=13, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Confidence: %{text}<extra></extra>',
        width=0.55,
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=220,
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, 110],
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color='#8b949e', size=12, family='Inter'),
        ),
        font=dict(family='Inter', color='#e6edf3'),
    )
    return fig

# ─── Plotly Gauge ───────────────────────────────────────────────────────────────
def confidence_gauge(confidence: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number=dict(suffix="%", font=dict(size=28, color='#e6edf3', family='Inter')),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#30363d', tickfont=dict(color='#8b949e', size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor='rgba(22,27,34,0.8)',
            bordercolor='#30363d',
            borderwidth=1,
            steps=[
                dict(range=[0, 33],  color='rgba(248,81,73,0.08)'),
                dict(range=[33, 66], color='rgba(210,153,34,0.08)'),
                dict(range=[66, 100],color='rgba(63,185,80,0.08)'),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.8,
                value=confidence,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=210,
        margin=dict(l=20, r=20, t=20, b=10),
        font=dict(family='Inter', color='#e6edf3'),
    )
    return fig

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 AI Diagnostics")
    st.markdown("---")

    st.markdown("### 🧠 Model Architecture")
    st.markdown("""
- **Base Model:** VGG16 (Transfer Learning)
- **Fine-tuning:** Last 4 layers unfrozen
- **Augmentation:** ImageDataGenerator
- **Input Size:** 128 × 128 × 3
- **Classes:** 3 (COVID, Normal, Pneumonia)
- **Optimizer:** Adam
- **Early Stopping:** ✅ Enabled
    """)

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")

    col1s, col2s = st.columns(2)
    with col1s:
        st.metric("Train", "~251", delta=None)
        st.metric("COVID", "111", delta=None)
    with col2s:
        st.metric("Test",  "~66",  delta=None)
        st.metric("Classes", "3", delta=None)

    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
1. Upload a chest X-ray image (JPG/PNG)
2. Wait for the AI to analyze
3. Review prediction & confidence
4. Consult a doctor for diagnosis
    """)

    st.markdown("---")
    st.markdown("### ⚡ Supported Formats")
    st.markdown("`JPG` · `JPEG` · `PNG`")

    st.markdown("---")
    st.caption("Built with TensorFlow · VGG16 · Streamlit")

# ─── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🔬 AI-Powered Medical Imaging</div>
    <h1 class="hero-title">COVID-19 X-Ray<br>Detection System</h1>
    <p class="hero-subtitle">Deep learning–powered chest X-ray analysis using VGG16 Transfer Learning.<br>
    Instantly classify images as <strong>COVID-19</strong>, <strong>Viral Pneumonia</strong>, or <strong>Normal</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ─── Stats Row ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">VGG16</div>
        <div class="stat-label">Base Architecture</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">3</div>
        <div class="stat-label">Disease Classes</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">128²</div>
        <div class="stat-label">Input Resolution</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">TF 2.x</div>
        <div class="stat-label">Framework</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading AI model..."):
    model = load_model()

if model is None:
    st.error(f"⚠️ Model file `{MODEL_PATH}` not found. Please place the `.keras` file in the same directory as this app.")
    st.stop()

# ─── Main Layout ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

# ── Left: Upload & Preview ────────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-header">📤 Upload Chest X-Ray</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Drop your X-ray image here",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Best results with frontal chest X-rays.",
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption=f"📁 {uploaded_file.name}  |  {image.size[0]}×{image.size[1]} px",
            use_column_width=True,
        )
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                    padding:0.7rem 1rem;margin-top:0.5rem;font-size:0.82rem;color:#8b949e;">
            <b style="color:#e6edf3;">File:</b> {uploaded_file.name} &nbsp;|&nbsp;
            <b style="color:#e6edf3;">Size:</b> {uploaded_file.size/1024:.1f} KB &nbsp;|&nbsp;
            <b style="color:#e6edf3;">Mode:</b> {image.mode}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:3rem;margin-bottom:0.8rem;">🩻</div>
            <div style="color:#8b949e;font-size:0.9rem;">
                Drag & drop or click <b style="color:#58a6ff;">Browse files</b><br>
                JPG · JPEG · PNG supported
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Right: Results ────────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="section-header">🔬 Analysis Results</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("🧠 Analyzing X-ray..."):
            time.sleep(0.5)                       # brief pause for UX
            arr = preprocess(image)
            predictions = model.predict(arr, verbose=0)[0]

        idx        = int(np.argmax(predictions))
        pred_class = CLASS_NAMES[idx]
        confidence = float(predictions[idx]) * 100
        cfg        = CLASS_CONFIG[pred_class]

        # ── Result card ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card {cfg['css']}">
            <div class="result-label" style="color:{cfg['color']};">AI Prediction</div>
            <p class="result-class">{cfg['icon']} {pred_class}</p>
            <p class="result-confidence">Model confidence: <b style="color:{cfg['color']};">{confidence:.2f}%</b></p>
            <span class="risk-badge {cfg['risk_css']}">{cfg['risk']}</span>
            <p style="color:#8b949e;font-size:0.85rem;margin-top:1rem;">{cfg['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauge ─────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header" style="margin-top:1.5rem;">📈 Confidence Gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(
            confidence_gauge(confidence, cfg['color']),
            use_container_width=True,
            config={'displayModeBar': False},
        )

        # ── Probability Bars ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Class Probabilities</div>', unsafe_allow_html=True)
        st.plotly_chart(
            probability_chart(predictions),
            use_container_width=True,
            config={'displayModeBar': False},
        )

        # ── Probability breakdown table ──────────────────────────────────────
        st.markdown('<div class="section-header">🗂️ Detailed Scores</div>', unsafe_allow_html=True)
        for i, cls in enumerate(CLASS_NAMES):
            c = CLASS_CONFIG[cls]
            pct = float(predictions[i]) * 100
            marker = " ← **Predicted**" if i == idx else ""
            st.progress(pct / 100)
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;margin:-0.6rem 0 0.6rem 0;"
                f"font-size:0.85rem;'>"
                f"<span style='color:{c['color']};font-weight:600;'>{c['icon']} {cls}{marker}</span>"
                f"<span style='color:#8b949e;font-family:JetBrains Mono,monospace;'>{pct:.4f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    else:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:16px;
                    padding:3rem 2rem;text-align:center;color:#8b949e;">
            <div style="font-size:3.5rem;margin-bottom:1rem;">🩺</div>
            <div style="font-size:1rem;font-weight:600;color:#e6edf3;margin-bottom:0.5rem;">
                Awaiting X-Ray Image
            </div>
            <div style="font-size:0.85rem;">
                Upload a chest X-ray on the left to start AI analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚕️ <strong>Medical Disclaimer:</strong> This tool is for <em>educational and research purposes only</em>.
    Predictions generated by this AI model are <strong>not a substitute for professional medical advice,
    diagnosis, or treatment</strong>. Always consult a qualified healthcare provider for medical decisions.
    This application was developed as part of an academic Data Science project.
</div>
""", unsafe_allow_html=True)
