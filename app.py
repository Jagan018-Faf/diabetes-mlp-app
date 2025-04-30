import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App (MLP Model)")

st.write("Please enter the patient's health details below:")

def user_input():
    Pregnancies = st.number_input("Pregnancies", min_value=0)
    Glucose = st.number_input("Glucose", min_value=0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0)
    Insulin = st.number_input("Insulin", min_value=0)
    BMI = st.number_input("BMI", min_value=0.0)
    DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    Age = st.number_input("Age", min_value=0)
    data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
    return data

input_data = user_input()

if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"Prediction Result: {result}")
