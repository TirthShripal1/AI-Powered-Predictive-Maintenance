from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/best_model.joblib")
META_PATH = Path("models/model_metadata.json")

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="centered")
st.title("⚙️ AI-Powered Predictive Maintenance")
st.write(
    "Estimate machine-failure risk from operating-condition measurements. "
    "This educational prototype uses the UCI AI4I 2020 dataset."
)

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.warning("No trained model found. From the project folder, run `python -m src.train` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

st.caption(f"Selected model: {metadata['best_model']} · Decision threshold: {metadata['threshold']:.2f}")

with st.form("prediction_form"):
    machine_type = st.selectbox("Machine product type", ["L", "M", "H"])
    air_temp = st.number_input("Air temperature (K)", min_value=295.0, max_value=305.0, value=300.0, step=0.1)
    process_temp = st.number_input("Process temperature (K)", min_value=305.0, max_value=315.0, value=310.0, step=0.1)
    rotational_speed = st.number_input("Rotational speed (rpm)", min_value=1000, max_value=3000, value=1500, step=10)
    torque = st.number_input("Torque (Nm)", min_value=3.0, max_value=80.0, value=40.0, step=0.5)
    tool_wear = st.number_input("Tool wear (minutes)", min_value=0, max_value=300, value=100, step=1)
    submitted = st.form_submit_button("Estimate failure risk")

if submitted:
    sample = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }])
    probability = float(model.predict_proba(sample)[0, 1])
    threshold = float(metadata.get("threshold", 0.50))
    st.subheader("Prediction")
    st.metric("Estimated failure probability", f"{probability:.1%}")
    if probability >= threshold:
        st.error("The model flags this input as higher risk. Consider inspection by a qualified maintenance professional.")
    else:
        st.success("The model does not flag this input above the configured threshold.")
    st.caption("This is a model estimate, not a safety guarantee or a substitute for equipment inspection.")
