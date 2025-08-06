import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

# Title
st.title("CASA Ratio Prediction App")
st.write("🔍 Predict the CASA Ratio using trained Random Forest Regression model.")

# Input feature form (example with 10 features)
# Replace these with your actual selected features
feature_names = [f"feature_{i}" for i in range(100)]  # adjust to real feature names

input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter value for {feature}:", step=0.1)

# When the button is clicked
if st.button("Predict CASA Ratio"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted CASA Ratio: {prediction:.4f}")
