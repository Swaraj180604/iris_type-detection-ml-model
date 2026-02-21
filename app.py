import os
import sys

# ── Dependency guard ──────────────────────────────────────────────────────────
_missing = []
for _pkg in ["joblib", "numpy", "pandas", "sklearn", "plotly"]:
    try:
        __import__(_pkg)
    except ModuleNotFoundError:
        _missing.append(_pkg)

import streamlit as st

if _missing:
    st.error(f"❌ Missing packages: {', '.join(_missing)}. Check requirements.txt and redeploy.")
    st.stop()

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IRIS · Neural Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iris_model.joblib")
DATA_PATH  = os.path.join(BASE_DIR, "iris_dataset.csv")

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

SPECIES_INFO = {
    0: {"name": "Setosa",     "emoji": "🌸", "color": "#FF6B9D", "bg": "rgba(255,107,157,0.12)", "desc": "Found in Arctic & sub-Arctic regions. Characterized by its small, compact blooms and distinctly separated petals.",                       "habitat": "Arctic Tundra",  "rarity": "Common"},
    1: {"name": "Versicolor", "emoji": "🌺", "color": "#A855F7", "bg": "rgba(168,85,247,0.12)",  "desc": "Native to eastern North America. A striking mid-size iris with beautifully veined petals and rich purple hues.",                         "habitat": "Wetlands",       "rarity": "Moderate"},
    2: {"name": "Virginica",  "emoji": "🌼", "color": "#06B6D4", "bg": "rgba(6,182,212,0.12)",   "desc": "The largest of the three species, native to eastern North America. Distinguished by its grand, showy flowers.",                          "habitat": "Coastal Plains", "rarity": "Rare"},
}

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_or_train_model():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    X = df[FEATURES]
    y = df["species_id"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            model.predict(X_train.iloc[:1])   # sanity check
        except Exception:
            model = None

    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        try:
            joblib.dump(model, MODEL_PATH)
        except Exception:
            pass  # read-only filesystem on some cloud envs — fine, model is in memory

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, df, acc

model, df, model_acc = load_or_train_model()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background: #050A14; color: #E8EDF5; }
.stApp { background: #050A14; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }

.stApp::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0; opacity: 0.4;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
}
.stApp::after {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(255,255,255,0.012) 2px,rgba(255,255,255,0.012) 4px);
}

.hero-wrap { position: relative; padding: 4rem 3rem 2.5rem; border-bottom: 1px solid rgba(255,255,255,0.06); overflow: hidden; }
.hero-glow { position: absolute; top:-100px; left:50%; transform:translateX(-50%); width:700px; height:400px; background:radial-gradient(ellipse,rgba(168,85,247,0.18) 0%,transparent 70%); pointer-events:none; }
.hero-label { font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:0.25em; color:#A855F7; text-transform:uppercase; margin-bottom:1rem; }
.hero-title { font-size:clamp(2.8rem,6vw,5.5rem); font-weight:800; line-height:0.95; letter-spacing:-0.03em; color:#F0F4FF; margin-bottom:1.2rem; }
.hero-title span { background:linear-gradient(135deg,#FF6B9D 0%,#A855F7 50%,#06B6D4 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero-sub { font-family:'Space Mono',monospace; font-size:0.8rem; color:rgba(255,255,255,0.35); letter-spacing:0.05em; max-width:480px; }
.hero-stats { display:flex; gap:2.5rem; margin-top:2rem; flex-wrap:wrap; }
.stat-item { display:flex; flex-direction:column; gap:0.2rem; }
.stat-val { font-family:'Space Mono',monospace; font-size:1.5rem; font-weight:700; color:#F0F4FF; }
.stat-lbl { font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.15em; color:rgba(255,255,255,0.3); text-transform:uppercase; }

.panel-label { font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.25em; color:rgba(255,255,255,0.25); text-transform:uppercase; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:0.75rem; margin-bottom:1.2rem; }

.result-card { border-radius:16px; padding:1.8rem 2rem; position:relative; overflow:hidden; transition:all 0.4s cubic-bezier(0.23,1,0.32,1); }
.result-species { font-size:3rem; font-weight:800; letter-spacing:-0.04em; line-height:1; margin-bottom:0.4rem; }
.result-latin { font-family:'Space Mono',monospace; font-size:0.7rem; font-style:italic; color:rgba(255,255,255,0.35); letter-spacing:0.05em; margin-bottom:1rem; }
.result-desc { font-size:0.82rem; color:rgba(255,255,255,0.5); line-height:1.6; margin-bottom:1.2rem; }
.result-tags { display:flex; gap:0.6rem; flex-wrap:wrap; }
.tag { font-family:'Space Mono',monospace; font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase; padding:0.3rem 0.7rem; border-radius:20px; border:1px solid rgba(255,255,255,0.12); color:rgba(255,255,255,0.45); }

.prob-row { display:flex; flex-direction:column; gap:0.4rem; margin-bottom:0.8rem; }
.prob-header { display:flex; justify-content:space-between; align-items:center; }
.prob-name { font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:0.1em; color:rgba(255,255,255,0.5); text-transform:uppercase; }
.prob-pct { font-family:'Space Mono',monospace; font-size:0.75rem; font-weight:700; }
.prob-track { height:4px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden; }
.prob-fill { height:100%; border-radius:2px; }

.mini-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:1.2rem; transition:border-color 0.3s; }
.mini-card:hover { border-color:rgba(168,85,247,0.3); }
.mini-card-label { font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:rgba(255,255,255,0.25); margin-bottom:0.5rem; }
.mini-card-val { font-size:1.4rem; font-weight:700; color:#F0F4FF; letter-spacing:-0.02em; }

.stTabs [data-baseweb="tab-list"] { gap:0; background:rgba(255,255,255,0.03); border-radius:10px; padding:4px; border:1px solid rgba(255,255,255,0.06); }
.stTabs [data-baseweb="tab"] { font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:rgba(255,255,255,0.35) !important; background:transparent !important; border-radius:7px; padding:0.5rem 1.2rem; transition:all 0.2s; }
.stTabs [aria-selected="true"] { background:rgba(168,85,247,0.2) !important; color:#A855F7 !important; border:1px solid rgba(168,85,247,0.3) !important; }
.stTabs [data-baseweb="tab-border"] { display:none; }

.stSlider > div > div > div > div { background:linear-gradient(90deg,#A855F7,#06B6D4) !important; height:3px !important; }
.stSlider > div > div > div > div > div { background:#fff !important; border:2px solid #A855F7 !important; width:16px !important; height:16px !important; box-shadow:0 0 12px rgba(168,85,247,0.6) !important; }

.tooltip-card { background:rgba(10,14,26,0.95); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:1rem 1.2rem; font-family:'Space Mono',monospace; font-size:0.72rem; color:rgba(255,255,255,0.6); line-height:1.7; margin-top:0.8rem; }

::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(168,85,247,0.3); border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-glow"></div>
  <div class="hero-label">// botanical intelligence system v2.1</div>
  <h1 class="hero-title">IRIS<br><span>Classifier</span></h1>
  <p class="hero-sub">Random Forest · 100 estimators · scikit-learn + joblib — Real-time species identification from morphological data.</p>
  <div class="hero-stats">
    <div class="stat-item"><span class="stat-val">{model_acc*100:.0f}%</span><span class="stat-lbl">Accuracy</span></div>
    <div class="stat-item"><span class="stat-val">150</span><span class="stat-lbl">Samples</span></div>
    <div class="stat-item"><span class="stat-val">4</span><span class="stat-lbl">Features</span></div>
    <div class="stat-item"><span class="stat-val">3</span><span class="stat-lbl">Species</span></div>
    <div class="stat-item"><span class="stat-val">100</span><span class="stat-lbl">Trees</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Inputs + Prediction ───────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.markdown('<div class="panel-label">// morphological input</div>', unsafe_allow_html=True)
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.8, 0.1)
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

    features    = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_id     = int(model.predict(features)[0])
    class_order = list(model.classes_)
    raw_proba   = model.predict_proba(features)[0]
    proba       = {int(cls): float(raw_proba[i]) for i, cls in enumerate(class_order)}
    info        = SPECIES_INFO[pred_id]
    conf        = proba[pred_id] * 100
    color       = info["color"]

    st.markdown(f"""
    <div class="result-card" style="background:{info['bg']};margin-top:1.5rem;border:1px solid {color}30;">
      <div class="result-species" style="color:{color};">{info['emoji']} {info['name']}</div>
      <div class="result-latin">Iris {info['name'].lower()} · {conf:.1f}% confidence</div>
      <div class="result-desc">{info['desc']}</div>
      <div class="result-tags">
        <span class="tag">🗺 {info['habitat']}</span>
        <span class="tag">◉ {info['rarity']}</span>
        <span class="tag">🌿 Species #{pred_id}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br><div class="panel-label">// classification confidence</div>', unsafe_allow_html=True)
    for sid, sinfo in SPECIES_INFO.items():
        pct  = proba.get(sid, 0.0) * 100
        bold = "font-weight:700;" if sid == pred_id else ""
        st.markdown(f"""
        <div class="prob-row">
          <div class="prob-header">
            <span class="prob-name">{sinfo['emoji']} {sinfo['name']}</span>
            <span class="prob-pct" style="color:{sinfo['color']};{bold}">{pct:.1f}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width:{pct}%;background:linear-gradient(90deg,{sinfo['color']}88,{sinfo['color']});"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="panel-label">// analytical dashboard</div>', unsafe_allow_html=True)

    PLOT_BG  = "rgba(0,0,0,0)"
    GRID_COL = "rgba(255,255,255,0.05)"
    TICK_COL = "rgba(255,255,255,0.25)"
    MONO     = "Space Mono"
    base     = dict(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=dict(family=MONO, color=TICK_COL, size=10), margin=dict(l=10,r=10,t=30,b=10))

    tab1, tab2, tab3, tab4 = st.tabs(["RADAR", "SCATTER", "DISTRIBUTION", "FEATURE IMPORTANCE"])

    with tab1:
        fl = ["Sepal L","Sepal W","Petal L","Petal W"]
        fig = go.Figure()
        for sid, sinfo in SPECIES_INFO.items():
            sp_df = df[df["species_id"]==sid]
            means = [sp_df[f].mean() for f in FEATURES]+[sp_df[FEATURES[0]].mean()]
            fig.add_trace(go.Scatterpolar(r=means,theta=fl+[fl[0]],fill='toself',fillcolor=sinfo["color"]+"18",line=dict(color=sinfo["color"],width=1.5,dash="dot"),name=sinfo["name"],opacity=0.7))
        user_r = [sepal_length,sepal_width,petal_length,petal_width,sepal_length]
        fig.add_trace(go.Scatterpolar(r=user_r,theta=fl+[fl[0]],fill='toself',fillcolor=color+"30",line=dict(color=color,width=3),name="Your Input",mode='lines+markers',marker=dict(size=6,color=color)))
        fig.update_layout(**base,height=360,polar=dict(bgcolor="rgba(255,255,255,0.02)",radialaxis=dict(visible=True,gridcolor=GRID_COL,linecolor=GRID_COL,tickfont=dict(size=8)),angularaxis=dict(gridcolor=GRID_COL,linecolor=GRID_COL,tickfont=dict(size=9,color="rgba(255,255,255,0.5)"))),showlegend=True,legend=dict(font=dict(family=MONO,size=9,color="rgba(255,255,255,0.4)"),bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.1,x=0.5,xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="tooltip-card">◈ Your input (solid) vs. average morphology per species.<br>◈ Current match → <b style="color:{color}">Iris {info["name"]}</b> at {conf:.1f}% confidence.</div>', unsafe_allow_html=True)

    with tab2:
        c1,c2 = st.columns(2)
        with c1: x_feat = st.selectbox("X Axis",FEATURES,index=2,key="sx")
        with c2: y_feat = st.selectbox("Y Axis",FEATURES,index=3,key="sy")
        fig2 = go.Figure()
        for sid,sinfo in SPECIES_INFO.items():
            sp_df = df[df["species_id"]==sid]
            fig2.add_trace(go.Scatter(x=sp_df[x_feat],y=sp_df[y_feat],mode='markers',name=sinfo["name"],marker=dict(color=sinfo["color"],size=7,opacity=0.7,line=dict(width=0))))
        fig2.add_trace(go.Scatter(x=[features[0][FEATURES.index(x_feat)]],y=[features[0][FEATURES.index(y_feat)]],mode='markers',name="Your Input",marker=dict(color=color,size=18,symbol="star",line=dict(color="#fff",width=1.5))))
        fig2.update_layout(**base,height=320,xaxis=dict(title=x_feat,gridcolor=GRID_COL,linecolor=GRID_COL,tickcolor=TICK_COL,showgrid=True,zeroline=False),yaxis=dict(title=y_feat,gridcolor=GRID_COL,linecolor=GRID_COL,tickcolor=TICK_COL,showgrid=True,zeroline=False),showlegend=True,legend=dict(font=dict(family=MONO,size=9,color="rgba(255,255,255,0.4)"),bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.18,x=0.5,xanchor="center"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        dist_feat = st.selectbox("Feature",FEATURES,index=2,key="df")
        fig3 = go.Figure()
        for sid,sinfo in SPECIES_INFO.items():
            sp_df = df[df["species_id"]==sid]
            fig3.add_trace(go.Violin(x=sp_df[dist_feat],name=sinfo["name"],fillcolor=sinfo["color"]+"40",line_color=sinfo["color"],meanline_visible=True,meanline=dict(color=sinfo["color"],width=2),opacity=0.8,orientation="h",side="positive",width=1.5,points="all",pointpos=-1.5,marker=dict(color=sinfo["color"],size=4,opacity=0.4)))
        fig3.add_vline(x=features[0][FEATURES.index(dist_feat)],line=dict(color="#fff",width=2,dash="dash"),annotation_text=f"↑ {features[0][FEATURES.index(dist_feat)]:.1f}cm",annotation_font=dict(family=MONO,size=9,color="#fff"))
        fig3.update_layout(**base,height=340,xaxis=dict(title=dist_feat,gridcolor=GRID_COL,linecolor=GRID_COL,tickcolor=TICK_COL,showgrid=True,zeroline=False),yaxis=dict(showticklabels=True,gridcolor=GRID_COL,linecolor=GRID_COL),showlegend=True,legend=dict(font=dict(family=MONO,size=9,color="rgba(255,255,255,0.4)"),bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.15,x=0.5,xanchor="center"))
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        imp = model.feature_importances_
        fn  = ["Sepal Length","Sepal Width","Petal Length","Petal Width"]
        si  = np.argsort(imp)
        bc  = [SPECIES_INFO[2]["color"] if imp[i]>0.3 else SPECIES_INFO[1]["color"] if imp[i]>0.1 else SPECIES_INFO[0]["color"] for i in si]
        fig4 = go.Figure(go.Bar(x=[imp[i] for i in si],y=[fn[i] for i in si],orientation='h',marker=dict(color=bc,line=dict(width=0)),text=[f"{imp[i]*100:.1f}%" for i in si],textposition='outside',textfont=dict(family=MONO,size=10,color="rgba(255,255,255,0.6)")))
        fig4.update_layout(**base,height=280,showlegend=False,xaxis=dict(title="Importance Score",gridcolor=GRID_COL,linecolor=GRID_COL,tickcolor=TICK_COL,showgrid=True,zeroline=False,range=[0,max(imp)*1.25]),yaxis=dict(gridcolor=GRID_COL,linecolor=GRID_COL,tickfont=dict(size=10,color="rgba(255,255,255,0.5)")),bargap=0.35)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('<div class="tooltip-card">◈ Petal dimensions dominate — petal length contributes most.<br>◈ Sepal width has the least predictive power across all three species.</div>', unsafe_allow_html=True)

# ── Bottom Stats ──────────────────────────────────────────────────────────────
st.markdown("---")
b1,b2,b3,b4 = st.columns(4)
with b1: st.markdown(f'<div class="mini-card"><div class="mini-card-label">// predicted species</div><div class="mini-card-val" style="color:{color};">{info["emoji"]} {info["name"]}</div></div>', unsafe_allow_html=True)
with b2: st.markdown(f'<div class="mini-card"><div class="mini-card-label">// confidence score</div><div class="mini-card-val">{conf:.1f}%</div></div>', unsafe_allow_html=True)
with b3:
    nn    = df[FEATURES].sub(features[0]).pow(2).sum(axis=1).pow(0.5)
    nname = SPECIES_INFO[int(df.loc[nn.idxmin(),"species_id"])]["name"]
    st.markdown(f'<div class="mini-card"><div class="mini-card-label">// nearest neighbor</div><div class="mini-card-val">{nname} <span style="font-size:0.8rem;color:rgba(255,255,255,0.3)">d={nn.min():.2f}</span></div></div>', unsafe_allow_html=True)
with b4:
    ratio = round(petal_length/petal_width,2) if petal_width>0 else 0
    st.markdown(f'<div class="mini-card"><div class="mini-card-label">// petal aspect ratio</div><div class="mini-card-val">{ratio}×</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
