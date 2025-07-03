import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/hvac_failure_model.pkl")

st.title("HVAC Predictive Maintenance & Optimization")

st.markdown("Enter HVAC parameters to predict failure risk in the next 7 days.")

# User input
ext_temp = st.number_input("External Temperature (°C)", 0.0, 50.0, 35.0)
target_temp = st.number_input("Target Temperature (°C)", 16.0, 30.0, 22.0)
runtime = st.number_input("Runtime Hours", 0.0, 24.0, 8.0)
comp_load = st.slider("Compressor Load (%)", 0, 100, 70)
fan_rpm = st.slider("Fan RPM", 900, 1500, 1100)
valve_stuck = st.radio("Valve Status", [0, 1], format_func=lambda x: "OK" if x == 0 else "Stuck")

if st.button("Predict Failure Risk"):
    data = pd.DataFrame([{
        "External_Temp": ext_temp,
        "Target_Temp": target_temp,
        "Runtime_Hours": runtime,
        "Compressor_Load": comp_load,
        "Fan_RPM": fan_rpm,
        "Valve_Stuck": valve_stuck
    }])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    st.markdown(f"### Prediction: {'⚠️ Likely Failure' if prediction else '✅ Normal'}")
    st.markdown(f"**Failure Probability:** {probability:.2%}")
