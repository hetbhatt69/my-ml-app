import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("CASA Ratio Prediction")

# Example inputs (replace with your actual feature names after RFE)
features = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4',
            'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature}", step=0.1)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted CASA Ratio: {prediction:.4f}")
