import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scalers
poly_model = joblib.load('model_poly_reg.pkl')
scaler_features = joblib.load('scaler_features.pkl')
scaler_target = joblib.load('scaler_target.pkl')
poly_reg = joblib.load('poly_reg.pkl')

# Define the feature names
column_names = [
    'PPG_Signal', 'Patient_Id', 'Heart_Rate', 'Systolic_Peak',
    'Diastolic_Peak', 'Pulse_Area', 'index', 'Gender', 'Height', 'Weight',
    'Age Range'
]

# Streamlit App Title
st.title("Glucose Level Prediction App")

# Input Section
st.header("Enter the Input Features")

input_values = []
for column_name in column_names:
    value = st.number_input(f"Enter value for {column_name}:", step=0.1, format="%.2f")
    input_values.append(value)

# Prediction Button
if st.button("Predict Glucose Level"):
    try:
        # Convert input into a DataFrame
        input_df = pd.DataFrame([input_values], columns=column_names)

        # Transform the input features
        input_scaled = scaler_features.transform(input_df)
        input_poly = poly_reg.transform(input_scaled)

        # Predict the scaled glucose level
        output_scaled = poly_model.predict(input_poly)

        # Convert the prediction back to the original scale
        output = scaler_target.inverse_transform(output_scaled)

        # Display the prediction
        st.success(f"Predicted Glucose Level: {output[0][0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
