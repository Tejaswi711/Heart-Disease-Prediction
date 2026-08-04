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
df = pd.read_csv("ANN_Diabetes_Dataset_75000Rows_13Columns_NoNull.csv")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction Using ANN")
st.write("Enter the patient details below.")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)

    Gender = st.selectbox("Gender", df["Gender"].unique())

    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

    Glucose = st.number_input("Glucose", min_value=50, max_value=300, value=100)

    BloodPressure = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)

    Insulin = st.number_input("Insulin", min_value=0, max_value=500, value=100)

with col2:

    HbA1c = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=5.5)

    PhysicalActivity = st.number_input(
        "Physical Activity (Hours/Week)",
        min_value=0.0,
        max_value=20.0,
        value=5.0
    )

    FamilyHistory = st.selectbox(
        "Family History",
        df["FamilyHistory"].unique()
    )

    Smoking = st.selectbox(
        "Smoking",
        df["Smoking"].unique()
    )

    StressLevel = st.slider("Stress Level", 1, 10, 5)

    SleepHours = st.number_input(
        "Sleep Hours",
        min_value=2.0,
        max_value=12.0,
        value=7.0
    )

# Create DataFrame
data = pd.DataFrame({
    "Age":[Age],
    "Gender":[Gender],
    "BMI":[BMI],
    "Glucose":[Glucose],
    "BloodPressure":[BloodPressure],
    "Insulin":[Insulin],
    "HbA1c":[HbA1c],
    "PhysicalActivity":[PhysicalActivity],
    "FamilyHistory":[FamilyHistory],
    "Smoking":[Smoking],
    "StressLevel":[StressLevel],
    "SleepHours":[SleepHours]
})

# Convert categorical columns
data["Gender"] = data["Gender"].astype(object)
data["FamilyHistory"] = data["FamilyHistory"].astype(object)
data["Smoking"] = data["Smoking"].astype(object)

if st.button("Predict"):

    # Preprocess
    processed_data = preprocessor.transform(data)

    # Prediction
    probability = model.predict(processed_data)

    prediction = np.where(probability > 0.5, 1, 0)

    if prediction[0][0] == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is unlikely to have Diabetes.")

    st.info(f"Probability of Diabetes: {probability[0][0] * 100:.2f}%")