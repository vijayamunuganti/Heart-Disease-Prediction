import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("heart_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# ---------------- FINAL CLEAN CUSTOM STYLE ----------------
st.markdown("""
    <style>

    /* Gradient Emerald Background */
    .stApp {
        background: linear-gradient(to bottom right, #D4EFDF, #A9DFBF);
    }

    /* Main Title */
    h1 {
        text-align: center;
        color: #145A32;
        font-weight: 700;
    }

    /* Labels */
    h2, h3, h4, h5, h6, p, span, label {
        color: #1E8449 !important;
        font-weight: 600;
    }

    /* Input Fields */
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 12px;
        border: 1px solid #82E0AA;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    }

    /* Predict Button - Clean & Bold */
    div.stButton > button {
    background-color: #196F3D !important;
    border-radius: 14px;
    height: 52px;
    border: none;
    box-shadow: 0px 6px 16px rgba(0,0,0,0.2);
}

div.stButton > button p {
    color: #FFFFFF !important;
    font-weight: 900 !important;
    font-size: 18px !important;
}

    

    div.stButton > button:hover {
        background-color: #145A32 !important;
        color: #FFFFFF !important;
    }

    /* Premium Footer */
    .premium-footer {
        margin-top: 70px;
        padding: 30px;
        text-align: center;
        font-size: 20px;
        font-weight: 700;
        color: #145A32;
        border-top: 2px solid #82E0AA;
        letter-spacing: 1.5px;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("❤️ Heart Disease Prediction System")
st.markdown("### 🩺 Patient Information")
st.markdown("---")

# ---------------- INPUT FIELDS ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(" Age", min_value=0, max_value=120, step=1)
    sex = st.selectbox(" Gender (0 = Female, 1 = Male)", [0, 1])
    chest_pain = st.selectbox(" Chest Pain Type (0-3)", [0,1,2,3])
    bp = st.number_input(" Blood Pressure", 0, 200, 120)
    cholesterol = st.number_input(" Cholesterol", 0, 600, 200)
    fbs = st.selectbox(" Fasting Blood Sugar > 120", [0,1])

with col2:
    ekg = st.selectbox(" EKG Results (0-2)", [0,1,2])
    max_hr = st.number_input(" Maximum Heart Rate", 0, 220, 150)
    exercise_angina = st.selectbox(" Exercise Induced Angina", [0,1])
    st_depression = st.number_input(" ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox(" Slope of ST (0-2)", [0,1,2])
    vessels = st.selectbox(" Number of Major Vessels", [0,1,2,3])
    thallium = st.selectbox("Thallium (1-3)", [1,2,3])

st.markdown("")

# ---------------- CENTERED PREDICT BUTTON ----------------
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    predict_button = st.button("🔍 Predict Heart Disease")

# ---------------- DIET PLAN FUNCTION ----------------
def get_diet_plan(prediction, disease_prob, no_disease_prob):

    if prediction == 1:
        if disease_prob >= 0.75:
            return """
🔴 High Risk (Heart Disease Present)

- Strictly Avoid fried & oily foods
- Avoid red meat and processed foods
- Reduce salt intake completely
- Eat oats, brown rice
- Include leafy vegetables daily
- Fruits: Apple, Pomegranate, Berries
- Use olive oil in small Quantity
- Walk 30–40 minutes daily
"""
        else:
            return """
🟠 Moderate Risk

- Limit oil & salt
- Avoid junk food
- Prefer grilled or steamed food
- Include fish twice a week
- Eat nuts like Almonds & walnuts
- Include fresh fruits daily
- Light exercise daily
"""
    else:
        if no_disease_prob >= 0.75:
            return """
🟢 Very Low Risk

- Maintain balanced diet
- Eat fruits & vegetables Daily
- Whole grains and legumes
- Stay physically active
- Drink Enough Water
- Annual Health check-up
"""
        else:
            return """
🟡 Slight Risk

- Reduce junk food and oily food
- Control sugar and salt
- Add more fiber-rich foods
- Include nuts and Seeds
- At least 30 minutes Exercise daily
- Monitor Blood Pressure Regularly
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

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    no_disease_prob = probabilities[0]
    disease_prob = probabilities[1]

    st.markdown("---")
    st.subheader("📋 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    st.subheader("📊 Prediction Probability")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("No Heart Disease", f"{no_disease_prob * 100:.2f}%")
    with col2:
        st.metric("Heart Disease", f"{disease_prob * 100:.2f}%")

    st.subheader("🥗 Personalized Diet Recommendation")
    diet_plan = get_diet_plan(prediction, disease_prob, no_disease_prob)
    st.info(diet_plan)

# ---------------- FOOTER ----------------
st.markdown("""
    <div class="premium-footer">
        HEART DISEASE PREDICTION <br>
        <span style="font-size:16px; font-weight:500;">
         ✦ Developed by Vijaya Munuganti ✦
        </span>
    </div>
""", unsafe_allow_html=True)