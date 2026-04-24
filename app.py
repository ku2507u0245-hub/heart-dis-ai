import streamlit as st
import joblib
import pandas as pd
import numpy as np
from fpdf import FPDF

# 1. Load the model
model = joblib.load('model.pkl')

st.set_page_config(page_title="CardioCheck Pro AI", page_icon="❤️", layout="wide")

# --- SIDEBAR INPUTS ---
st.sidebar.header("📋 Patient Data Entry")

def get_user_inputs():
    age = st.sidebar.slider("Age", 1, 100, 50)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    restecg = st.sidebar.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Mapping text back to numbers for the AI
    sex_bin = 1 if sex == "Male" else 0
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal": 2, "Asymptomatic": 3}
    fbs_bin = 1 if fbs == "True" else 0
    rest_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    ex_bin = 1 if exang == "Yes" else 0
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    features = np.array([[age, sex_bin, cp_map[cp], trestbps, chol, fbs_bin, 
                          rest_map[restecg], thalach, ex_bin, oldpeak, 
                          slope_map[slope], ca, thal_map.get(thal, 0)]])
    return features, locals()

input_features, raw_data = get_user_inputs()

# --- MAIN PAGE DISPLAY ---
st.title("🏥 CardioCheck Pro: Clinical Dashboard")
st.write("Diagnostic AI assistant for cardiovascular risk assessment.")

if st.button("🚀 Run Diagnostic Analysis"):
    # Probability Calculation
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)[0][1] * 100

    st.markdown("---")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.subheader("Analysis Result")
        if prediction[0] == 1:
            st.error(f"### HIGH RISK DETECTED")
            st.metric("Confidence Score", f"{probability:.1f}%")
        else:
            st.success(f"### LOW RISK DETECTED")
            st.metric("Confidence Score", f"{100 - probability:.1f}%")

    with col_res2:
        st.subheader("Risk Interpretation")
        st.write(f"Based on the input parameters, the AI model predicts a **{probability:.1f}%** probability of heart disease presence.")
        st.progress(probability / 100)

    # --- FEATURE 3: PDF GENERATOR ---
    def create_pdf(prob, pred):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="CardioCheck AI Diagnostic Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Patient Age: {raw_data['age']}", ln=True)
        pdf.cell(200, 10, txt=f"Risk Prediction: {'High Risk' if pred == 1 else 'Low Risk'}", ln=True)
        pdf.cell(200, 10, txt=f"Probability Score: {prob:.2f}%", ln=True)
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt="Disclaimer: This is an AI-generated assessment and should only be used by medical professionals as a decision-support tool.")
        return pdf.output(dest='S').encode('latin-1')

    pdf_data = create_pdf(probability, prediction[0])
    st.download_button(label="📥 Download Clinical Report", data=pdf_data, file_name="Heart_Report.pdf", mime="application/pdf")