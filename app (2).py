import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# Load Preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Load ANN Model
model = load_model("architecture.keras")

# Load Dataset (for dropdown values)
df = pd.read_csv("heart_disease_dataset.csv")

st.set_page_config(page_title="Heart Disease Prediction", page_icon=None)

st.title("Heart Disease Prediction Using ANN")
st.write("Enter the patient details below.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=40)

    gender = st.selectbox("Gender", df["gender"].unique())

    resting_bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=220, value=120)

    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=450, value=200)

    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    resting_ecg = st.selectbox("Resting ECG", df["resting_ecg"].unique())

with col2:

    max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)

    exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])

    smoking_status = st.selectbox("Smoking Status", df["smoking_status"].unique())

    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

    physical_activity = st.selectbox("Physical Activity Level", df["physical_activity"].unique())

    family_history = st.selectbox("Family History", [0, 1])

# Create DataFrame
data = pd.DataFrame({
    "age":[age],
    "gender":[gender],
    "resting_bp":[resting_bp],
    "cholesterol":[cholesterol],
    "fasting_blood_sugar":[fasting_blood_sugar],
    "resting_ecg":[resting_ecg],
    "max_heart_rate":[max_heart_rate],
    "exercise_angina":[exercise_angina],
    "smoking_status":[smoking_status],
    "bmi":[bmi],
    "physical_activity":[physical_activity],
    "family_history":[family_history]
})

# Convert categorical columns
data["gender"] = data["gender"].astype(object)
data["resting_ecg"] = data["resting_ecg"].astype(object)
data["smoking_status"] = data["smoking_status"].astype(object)
data["physical_activity"] = data["physical_activity"].astype(object)

if st.button("Predict"):

    # Preprocess
    processed_data = preprocessor.transform(data)

    # Prediction
    probability = model.predict(processed_data)

    prediction = np.where(probability > 0.5, 1, 0)

    if prediction[0][0] == 1:
        st.error("The patient is likely to have Heart Disease.")
    else:
        st.success("The patient is unlikely to have Heart Disease.")

    st.info(f"Probability of Heart Disease: {probability[0][0] * 100:.2f}%")
