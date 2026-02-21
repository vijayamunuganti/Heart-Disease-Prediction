import streamlit as st
import pandas as pd
import joblib
import pickle

# Load trained model
model = joblib.load("heart_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.markdown("Enter patient details below to predict heart disease.")

# -------- INPUT FIELDS --------
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
chest_pain = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
bp = st.number_input("Blood Pressure (BP)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0,1])
ekg = st.selectbox("EKG Results (0-2)", [0,1,2])
max_hr = st.number_input("Maximum Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0,1])
st_depression = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of ST (0-2)", [0,1,2])
vessels = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thallium = st.selectbox("Thallium (1-3)", [1,2,3])

# -------- PREDICTION --------
# -------- PREDICTION --------
if st.button("Predict"):

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Chest pain type': [chest_pain],
        'BP': [bp],
        'Cholesterol': [cholesterol],
        'FBS over 120': [fbs],
        'EKG results': [ekg],
        'Max HR': [max_hr],
        'Exercise angina': [exercise_angina],
        'ST depression': [st_depression],
        'Slope of ST': [slope],
        'Number of vessels fluro': [vessels],
        'Thallium': [thallium]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")

        st.subheader("Recommended Diet Plan:")
        st.write("- Eat more fruits and vegetables")
        st.write("- Whole grains")
        st.write("- Lean proteins")
        st.write("- Reduce salt intake")

    else:
        st.success("✅ No Heart Disease Detected")

        st.subheader("Preventive Diet Plan:")
        st.write("- Maintain balanced diet")
        st.write("- Eat fresh fruits daily")
        st.write("- Limit junk food")
        st.write("- Stay hydrated")
        st.write("- Regular exercise")