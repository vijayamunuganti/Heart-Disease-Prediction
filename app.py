import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("heart_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# ---------------- STYLE ----------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #D4EFDF, #A9DFBF);
    }
    h1 {
        text-align: center;
        color: #145A32;
    }
    label {
        color: #1E8449 !important;
        font-weight: 600;
    }
    div.stButton > button {
        background-color: #196F3D !important;
        color: white !important;
        border-radius: 12px;
        height: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("❤️ Heart Disease Prediction System")
st.markdown("### 🩺 Patient Information")
st.markdown("---")

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 0, 120, 30)
    sex = st.selectbox("Gender", [0, 1])
    chest_pain = st.selectbox("Chest Pain Type", [0,1,2,3])
    bp = st.number_input("Blood Pressure", 70, 200, 120)
    cholesterol = st.number_input("Cholesterol", 70, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])

with col2:
    ekg = st.selectbox("EKG Results", [0,1,2])
    max_hr = st.number_input("Max Heart Rate", 0, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", [0,1])
    st_depression = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0,1,2])
    vessels = st.selectbox("Vessels", [0,1,2,3])
    thallium = st.selectbox("Thallium", [1,2,3])

# ---------------- BUTTON ----------------
predict_button = st.button("🔍 Predict Heart Disease")

# ---------------- DIET FUNCTION ----------------
def get_diet_plan(prediction, disease_prob, no_disease_prob):

    if prediction == 1:
        if disease_prob >= 0.75:
            return """
🔴 High Risk (Heart Disease Present)

- Strictly avoid fried & oily foods
- Avoid red meat and processed foods
- Reduce salt intake
- Eat oats and brown rice
- Include leafy vegetables daily
- Fruits: Apple, Pomegranate, Berries
- Use olive oil in small quantity
- Walk 30–40 minutes daily
"""
        else:
            return """
🟠 Moderate Risk

- Limit oil & salt
- Avoid junk food
- Prefer grilled or steamed food
- Include fish twice a week
- Eat nuts like almonds & walnuts
- Include fresh fruits daily
- Light exercise daily
"""

    else:
        if no_disease_prob >= 0.75:
            return """
🟢 Very Low Risk

- Maintain balanced diet
- Eat fruits & vegetables daily
- Whole grains and legumes
- Stay physically active
- Drink enough water
- Annual health check-up
"""
        else:
            return """
🟡 Slight Risk

- Reduce junk food and oily food
- Control sugar and salt
- Add more fiber-rich foods
- Include nuts and seeds
- At least 30 minutes exercise daily
- Monitor blood pressure regularly
"""

# ---------------- PREDICTION ----------------
if predict_button:

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

    # Model prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    # ✅ UNIVERSAL LOGIC (works for any dataset)
    max_index = probabilities.argmax()
    predicted_class = classes[max_index]

    disease_prob = probabilities[max_index]
    no_disease_prob = 1 - disease_prob

    st.markdown("---")
    st.subheader("📋 Prediction Result")

    if predicted_class != 0:
        st.error("⚠️ Heart Disease Detected")
        final_prediction = 1
    else:
        st.success("✅ No Heart Disease Detected")
        final_prediction = 0

    # ---------------- PROBABILITY ----------------
    st.subheader("📊 Prediction Probability")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("No Heart Disease", f"{no_disease_prob * 100:.2f}%")
    with col2:
        st.metric("Heart Disease", f"{disease_prob * 100:.2f}%")

    # ---------------- DIET OUTPUT ----------------
    st.subheader("🥗 Personalized Diet Recommendation")

    diet_plan = get_diet_plan(final_prediction, disease_prob, no_disease_prob)
    st.markdown(diet_plan)

# ---------------- FOOTER ----------------
st.markdown("""
    <div style='text-align:center; margin-top:50px;'>
        ✦ Developed by Vijaya Munuganti ✦
    </div>
""", unsafe_allow_html=True)