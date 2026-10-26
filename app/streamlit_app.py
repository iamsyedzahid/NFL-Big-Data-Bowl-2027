import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Draft Prospect Evaluator", layout="wide")

st.title("🏈 NFL Combine-to-Pro Performance Evaluator")
st.title("BY Syed Zahid Hussain - 24K0015- Lab 09 - Machine Learning")
st.write("Predicting regular-season success probability using Combine kinematics & Next Gen Stats.")

MODEL_PATH = "artifacts/nfl_prospect_model.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

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

    # Dynamic kinematic calculations scaled from slider inputs
    dynamic_accel = max(2.0, (12.0 - forty) * 1.5)
    
    input_data = pd.DataFrame([{
        'combine_height': float(height),
        'combine_weight': float(weight),
        'hand_size': 9.5,
        'arm_length': 32.0,
        'ten_yd_split': float(ten_yd),
        'forty': float(forty),
        'vertical': float(vertical),
        'broad_jump': float(vertical * 3.2),
        'three_cone': float(7.10),
        'short_shuttle': float(4.25),
        'bench_reps': float(bench),
        'ngs_athleticism_score': float(ngs_ath),
        'ngs_college_production_score': float(ngs_prod),
        'combine_max_speed': float(combine_max_s),
        'combine_max_accel': float(dynamic_accel),
        'avg_jerk_ratio': 0.85,
        'game_max_speed': float(combine_max_s * 1.05),
        'game_avg_speed': float(combine_max_s * 0.60),
        'game_max_accel': float(dynamic_accel * 1.1)
    }])

    # Match exact feature names and order expected by trained model
    if hasattr(model, "feature_name_"):
        input_data = input_data.reindex(columns=model.feature_name_, fill_value=0.0)

    if st.button("Evaluate Prospect"):
        prob = model.predict_proba(input_data)[0][1]
        
        st.divider()
        st.metric(label="High-Impact Starter Probability", value=f"{prob * 100:.4f}%")
        st.caption(f"Raw Output Probability: {prob:.8f}")
        
        if prob >= 0.65:
            st.success("Target Classification: Day 1 Immediate Starter /
