import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Draft Prospect Evaluator", layout="wide")

st.title("🏈 NFL Combine-to-Pro Performance Evaluator")
st.write("Predicting regular-season success probability using Combine kinematics & Next Gen Stats.")

MODEL_PATH = "artifacts/nfl_prospect_model.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Directly load the model object (LGBMClassifier)
model = load_model()

if model is None:
    st.warning("⚠️ No model artifact found in `artifacts/nfl_prospect_model.joblib`. Please run `python -m src.train` first.")
else:
    col1, col2, col3 = st.columns(3)

    with col1:
        forty = st.slider("40-Yard Dash (s)", 4.20, 5.50, 4.50)
        ten_yd = st.slider("10-Yard Split (s)", 1.40, 1.80, 1.55)
        vertical = st.slider("Vertical Jump (in)", 25.0, 45.0, 35.0)

    with col2:
        weight = st.number_input("Weight (lbs)", 160, 380, 225)
        height = st.number_input("Height (in)", 65.0, 82.0, 74.0)
        bench = st.slider("Bench Reps (225 lbs)", 0, 40, 18)

    with col3:
        ngs_ath = st.slider("NGS Athleticism Score", 50, 99, 78)
        ngs_prod = st.slider("NGS Production Score", 50.0, 99.0, 72.0)
        combine_max_s = st.slider("Combine Max Speed (yd/s)", 5.0, 12.0, 9.5)

    input_data = pd.DataFrame([{
        'combine_height': height,
        'combine_weight': weight,
        'hand_size': 9.5,
        'arm_length': 32.0,
        'ten_yd_split': ten_yd,
        'forty': forty,
        'vertical': vertical,
        'broad_jump': 120.0,
        'three_cone': 7.10,
        'short_shuttle': 4.25,
        'bench_reps': bench,
        'ngs_athleticism_score': ngs_ath,
        'ngs_college_production_score': ngs_prod,
        'combine_max_speed': combine_max_s,
        'combine_max_accel': 4.5,
        'avg_jerk_ratio': 0.85
    }])

    if st.button("Evaluate Prospect"):
        prob = model.predict_proba(input_data)[0][1]
        st.metric(label="High-Impact Starter Probability", value=f"{prob * 100:.1f}%")
        if prob >= 0.65:
            st.success("Target Classification: Day 1 Immediate Starter / Star")
        else:
            st.info("Target Classification: Developmental / System Player")