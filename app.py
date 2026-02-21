import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iris_model.joblib")
DATA_PATH  = os.path.join(BASE_DIR, "iris_dataset.csv")

# ── Train & cache model ───────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        # Train from the CSV bundled in the repo
        df = pd.read_csv(DATA_PATH)
        FEATURES = ["sepal length (cm)", "sepal width (cm)",
                    "petal length (cm)", "petal width (cm)"]
        X = df[FEATURES]
        y = df["species_id"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        joblib.dump(model, MODEL_PATH)
        st.toast(f"✅ Model trained on the fly! Accuracy: {acc*100:.1f}%")
    return model

model = load_or_train_model()

SPECIES = {0: "🌸 Setosa", 1: "🌺 Versicolor", 2: "🌼 Virginica"}
COLORS  = {0: "#ff6b9d",   1: "#845ec2",        2: "#0081cf"}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .hero { text-align: center; padding: 2rem 1rem 1rem; }
    .hero h1 {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(90deg, #ff6b9d, #845ec2, #0081cf);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p { color: #555; font-size: 1rem; font-weight: 300; }

    .card {
        background: white; border-radius: 20px; padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08); margin: 1.5rem 0;
    }

    .result-box {
        border-radius: 16px; padding: 1.8rem; text-align: center;
        color: white; font-size: 1.8rem; font-weight: 700; margin-top: 1rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        animation: fadeIn 0.5s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #333;
        margin-bottom: 1rem; border-left: 4px solid #845ec2;
        padding-left: 0.6rem;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌸 Iris Classifier</h1>
    <p>Predict the species of an Iris flower using a Random Forest model — trained with scikit-learn & saved with joblib.</p>
</div>
""", unsafe_allow_html=True)

# ── Input sliders ─────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📐 Enter Flower Measurements (cm)</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8, 0.1)
    petal_length = st.slider("Petal Length", 1.0, 7.0, 3.8, 0.1)
with col2:
    sepal_width  = st.slider("Sepal Width",  2.0, 4.5, 3.0, 0.1)
    petal_width  = st.slider("Petal Width",  0.1, 2.5, 1.2, 0.1)

st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
features   = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred_id    = int(model.predict(features)[0])
proba      = model.predict_proba(features)[0]
species    = SPECIES[pred_id]
color      = COLORS[pred_id]
confidence = proba[pred_id] * 100

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔍 Prediction Result</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="result-box" style="background: linear-gradient(135deg, {color}cc, {color});">
    {species}
    <div style="font-size:1rem; font-weight:400; margin-top:0.4rem; opacity:0.9;">
        Confidence: {confidence:.1f}%
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Class Probabilities**")
for sid, sname in SPECIES.items():
    st.markdown(f"**{sname}**")
    st.progress(float(proba[sid]))
    st.caption(f"{proba[sid]*100:.1f}%")

st.markdown('</div>', unsafe_allow_html=True)

# ── About ─────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this app"):
    st.markdown("""
    - **Dataset**: Classic Iris dataset (150 samples, 3 classes)
    - **Model**: Random Forest Classifier (100 estimators)
    - **Accuracy**: 100% on test set
    - **Saved with**: `joblib`
    - **Framework**: Streamlit
    - **Note**: If `iris_model.joblib` is missing, the model is auto-trained from `iris_dataset.csv` at startup.
    """)
