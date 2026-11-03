import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart Stroke Prediction by akarsh")
st.markdown("Provide the following details to check your heart stroke risk:")

# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
chol= st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fbs= st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


# When Predict is clicked
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'Cholesterol': chol,
        'FastingBS': fbs,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + cp: 1,
        'RestingECG_' + restecg: 1,
        'ST_Slope_' + slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")