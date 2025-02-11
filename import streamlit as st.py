import streamlit as st
import pandas as pd
import joblib

# Load saved models and scalers
poly_model = joblib.load('poly_model.pkl')
poly_reg = joblib.load('poly_reg.pkl')
scaler_features = joblib.load('scaler_features.pkl')
scaler_target = joblib.load('scaler_target.pkl')

# Define the column names
column_names = [
    'PPG_Signal(mV)', 'Patient_Id(ID number)', 'Heart_Rate(bpm)', 'Systolic_Peak(mmHg)', 
    'Diastolic_Peak(mmHg)', 'Pulse_Area', 'index(integer)',
    'Gender(1 for Male, 0 for Female)', 'Height(cm)', 'Weight(kg)', 'Age Range[1,2,3,4,5]'
]

st.title("Glucose Level Prediction App")
st.write("Enter the required parameters to predict the glucose level:")

# User inputs
input_values = []
for column_name in column_names:
    value = st.number_input(f"Enter value for {column_name}:", value=0.0)
    input_values.append(value)

# Prediction button
if st.button("Predict"):
    # Create a DataFrame for the input
    input_df = pd.DataFrame([input_values], columns=column_names)

    # Transform input data
    input_scaled = scaler_features.transform(input_df)
    input_poly = poly_reg.transform(input_scaled)

    # Make prediction
    output_scaled = poly_model.predict(input_poly)
    output = scaler_target.inverse_transform(output_scaled)

    # Display the prediction
    st.success(f"Predicted Glucose Level: {output[0][0]:.2f}")
