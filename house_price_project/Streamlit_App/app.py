"""
House Price Prediction System — Streamlit App
AIML Summer Internship 2026, IIHMF, MNNIT Allahabad
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #F2EAD3; }
.block-container { padding: 2rem 2.5rem; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem; font-weight: 700;
    color: #1B2A3E; line-height: 1.1; margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1rem; color: #5A6A7A; margin-bottom: 2rem;
}
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em;
    color: #2EC4B6; text-transform: uppercase; margin-bottom: 0.5rem;
}
.prediction-card {
    background: #1B2A3E;
    border-radius: 16px;
    padding: 2rem;
    color: white;
    position: sticky; top: 2rem;
}
.pred-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem; letter-spacing: 0.12em; color: #2EC4B6;
    text-transform: uppercase; margin-bottom: 0.5rem;
}
.pred-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem; font-weight: 600; color: #F2EAD3;
    line-height: 1; margin-bottom: 0.2rem;
}
.pred-sub { font-size: 0.85rem; color: #8899AA; margin-bottom: 1.5rem; }

.metric-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.8rem; margin-top: 1rem;
}
.metric-box {
    background: rgba(255,255,255,0.07);
    border-radius: 10px; padding: 0.8rem;
}
.metric-box-label { font-size: 0.7rem; color: #8899AA; text-transform: uppercase; letter-spacing: 0.1em; }
.metric-box-value { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: #F2EAD3; font-weight: 600; }

.gauge-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 6px; height: 8px; margin: 0.5rem 0 1.5rem;
}
.gauge-bar-fill {
    background: linear-gradient(90deg, #2EC4B6, #48E0D4);
    border-radius: 6px; height: 8px; transition: width 0.5s ease;
}
.input-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #E8E0D0;
}
.model-badge {
    display: inline-block;
    background: #2EC4B6; color: white;
    border-radius: 20px; padding: 0.25rem 0.8rem;
    font-size: 0.75rem; font-weight: 600; margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, '..', 'Model')
    model   = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
    scaler  = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    le      = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    with open(os.path.join(model_dir, 'metrics.json')) as f:
        meta = json.load(f)
    return model, scaler, le, meta

model, scaler, le, meta = load_artifacts()
locations = meta['locations']
best_name = meta['best_model']
best_r2   = meta['best_r2']

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🏠 House Price<br>Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Enter property details to get an instant AI-powered valuation</p>', unsafe_allow_html=True)

col_inputs, col_pred = st.columns([1.4, 1], gap="large")

with col_inputs:
    # ── Property Basics ────────────────────────────────────────────────────────
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Property Basics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Area (sq ft)", min_value=300, max_value=10000, value=1500, step=50)
        bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5,6], index=2)
    with c2:
        location = st.selectbox("Location", locations)
        bathrooms = st.selectbox("Bathrooms", [1,2,3,4], index=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Structure ──────────────────────────────────────────────────────────────
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Structure</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        age = st.slider("Age of Property (years)", 0, 50, 10)
    with c4:
        floors = st.selectbox("Number of Floors", [1,2,3], index=0)
    garage = st.selectbox("Garage Spaces", [0,1,2], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Amenities ──────────────────────────────────────────────────────────────
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Amenities</p>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        has_pool   = st.checkbox("Swimming Pool 🏊", value=False)
    with c6:
        has_garden = st.checkbox("Garden / Yard 🌿", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
loc_enc       = le.transform([location])[0]
price_per_sqft_est = 200  # seed estimate
total_rooms   = bedrooms + bathrooms
amenity_score = int(has_pool) + int(has_garden) + garage

X_input = pd.DataFrame([{
    'area_sqft': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
    'age_years': age, 'floors': floors, 'garage_spaces': garage,
    'has_pool': int(has_pool), 'has_garden': int(has_garden),
    'location_enc': loc_enc,
    'price_per_sqft': price_per_sqft_est,
    'total_rooms': total_rooms, 'amenity_score': amenity_score
}])

pred_price = model.predict(X_input)[0]
pred_price = max(pred_price, 50000)
pred_psf   = pred_price / area
low_est    = pred_price * 0.93
high_est   = pred_price * 1.07

# gauge: 0–2M scale
gauge_pct  = min(pred_price / 2_000_000 * 100, 100)

with col_pred:
    st.markdown(f"""
    <div class="prediction-card">
        <p class="pred-label">Estimated Value</p>
        <p class="pred-value">${pred_price:,.0f}</p>
        <p class="pred-sub">${low_est:,.0f} – ${high_est:,.0f} range</p>

        <div class="gauge-bar-bg">
            <div class="gauge-bar-fill" style="width:{gauge_pct:.1f}%"></div>
        </div>
        <p style="font-size:0.72rem;color:#8899AA;margin-bottom:1.2rem;">
            ▲ {gauge_pct:.0f}th percentile of $0–$2M scale
        </p>

        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-box-label">Price / sqft</div>
                <div class="metric-box-value">${pred_psf:,.0f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Location</div>
                <div class="metric-box-value">{location}</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Model R²</div>
                <div class="metric-box-value">{best_r2:.4f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Amenity Score</div>
                <div class="metric-box-value">{amenity_score} / 4</div>
            </div>
        </div>

        <div style="margin-top:1.2rem;">
            <span class="model-badge">✦ {best_name}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Model Metrics Table ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Model Performance Comparison")
rows = []
for name, m in meta['results'].items():
    rows.append({'Model': name, 'MAE ($)': f"{m['MAE']:,.0f}", 'RMSE ($)': f"{m['RMSE']:,.0f}", 'R² Score': m['R2']})
df_metrics = pd.DataFrame(rows)
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

st.caption("AIML Summer Internship 2026 · IIHMF, MNNIT Allahabad · House Price Prediction System")
