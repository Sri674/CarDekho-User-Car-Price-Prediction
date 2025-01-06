import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

# Load your pre-trained model, scaler, and encoder using joblib
best_rf_model = joblib.load(r'H:\\CarDekho_session\\env\\best_rf_model.pkl')  # Load the Random Forest model
encoders = joblib.load(r'H:\\CarDekho_session\\env\\categorical_encoders.pkl')  # Load the encoders for categorical features
scaler = joblib.load(r'H:\\CarDekho_session\\env\\scaler.pkl')  # Load the scaler for numerical features
scalerTR = joblib.load(r'H:\\CarDekho_session\\env\\scalerTR.pkl')  # In case you have another scaler for transformation

# Define the feature names (13 features) as per the dataset you provided
feature_names = [
    "fuel type", "Body type", "Kms Driven", "transmission", "Ownership", "oem", 
    "price", "Seats", "Max Power", "Torque", "Mileage", "City", "Engine_category", "Car_Age"
]

st.image("https://stimg.cardekho.com/pwa/img/carDekho-newLogo.svg", width=300)  

st.title('Car Price Prediction') 

# Input for categorical features
fuel_type = st.selectbox("Fuel Type", options=encoders['fuel type'].classes_)
body_type = st.selectbox("Body Type", options=encoders['Body type'].classes_)
transmission = st.selectbox("Transmission", options=encoders['transmission'].classes_)
ownership = st.selectbox("Ownership", options=encoders['Ownership'].classes_)
oem = st.selectbox("OEM", options=encoders['oem'].classes_)
city = st.selectbox("City", options=encoders['City'].classes_)
engine_category = st.selectbox("Engine Category", options=encoders['Engine_category'].classes_)

# Input for numerical features
kms_driven = st.number_input("KMS Driven", min_value=0)
seats = st.number_input("Seats", min_value=2, max_value=10)
max_power = st.number_input("Max Power (in bhp)", min_value=0.0)
torque = st.number_input("Torque (in Nm)", min_value=0.0)
mileage = st.number_input("Mileage (in kmpl)", min_value=0.0)
car_age = st.number_input("Car Age (in years)", min_value=0, max_value=22)


# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    "fuel type": [fuel_type],
    "Body type": [body_type],
    "Kms Driven": [kms_driven],
    "transmission": [transmission],
    "Ownership": [ownership],
    "oem": [oem],
    "Seats": [seats],
    "Max Power": [max_power],
    "Torque": [torque],
    "Mileage": [mileage],
    "City": [city],
    "Engine_category": [engine_category],
    "Car_Age": [car_age]
}) 

# Transform categorical features using the encoders
for col in ['fuel type', 'Body type', 'transmission', 'Ownership', 'oem', 'City', 'Engine_category']:
    input_data[col] = encoders[col].transform(input_data[col])

# Scale numerical features using the scaler
numerical_features = ['Kms Driven', 'Max Power', 'Torque', 'Mileage']
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Make the prediction (this is for the scaled version of price)
scaled_predicted_price = best_rf_model.predict(input_data)

# Transform the scaled predicted price back to the original scale using scalerTR
predicted_price = scalerTR.inverse_transform(scaled_predicted_price.reshape(-1, 1))

# Display the predicted car price
st.write(f"Predicted Car Price: {predicted_price[0][0]:,.2f}")

# Optional: Display the prepared input data
st.write("Prepared input data for prediction:")
st.write(input_data)






























