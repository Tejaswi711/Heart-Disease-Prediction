# Heart Disease Prediction Using Artificial Neural Network (ANN)

## Project Overview

This project predicts whether a patient is likely to have diabetes using an Artificial Neural Network (ANN). The model is trained using patient health-related features and helps classify patients as diabetic or non-diabetic.

---

## Business Problem

Diabetes is one of the most common chronic diseases worldwide. Early identification of individuals at risk can help healthcare professionals provide timely treatment and preventive care. Manual diagnosis can be time-consuming and prone to errors when handling large amounts of patient data. Therefore, an automated prediction system is required.

---

## Problem Statement

Develop an Artificial Neural Network (ANN) model to predict whether a patient has diabetes using health-related features such as:

- Age
- Gender
- BMI
- Glucose
- Blood Pressure
- Insulin
- HbA1c
- Physical Activity
- Family History
- Smoking
- Stress Level
- Sleep Hours

---

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

---

## Project Workflow

1. Load Dataset
2. Data Preprocessing
3. Handle Missing Values
4. Encode Categorical Variables
5. Feature Scaling
6. Train-Test Split
7. Build ANN Model
8. Train Model
9. Evaluate Model
10. Classification Report
11. Confusion Matrix
12. Deploy using Streamlit

---

## Model Architecture

- Input Layer
- Hidden Layer (ReLU)
- Hidden Layer (ReLU)
- Output Layer (Sigmoid)

Optimizer:
- Adam

Loss Function:
- Binary Crossentropy

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Files

```
Healthcare.ipynb
app.py
architecture.keras
preprocessor.pkl
requirements.txt
README.md
```

---

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

## Conclusion

This project demonstrates how Artificial Neural Networks can be applied to healthcare data for diabetes prediction. The model was trained using patient health information and evaluated using multiple classification metrics. The trained ANN achieved good predictive performance and can assist in early diabetes detection.
