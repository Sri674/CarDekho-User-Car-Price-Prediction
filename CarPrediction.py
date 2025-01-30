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
b_encoders=pd.read_csv(r"H:\\CarDekho_session\\env\\overall_totalcities_df21.csv") # loading datset before encoding

# Define the feature names (13 features) as per the dataset you provided
feature_names = [
    "fuel type", "Body type", "Kms Driven", "transmission", "Ownership", "oem", 
    "price", "Seats", "Max Power", "Torque", "Mileage", "City", "Engine_category", "Car_Age"
]

st.image("https://stimg.cardekho.com/pwa/img/carDekho-newLogo.svg", width=300)  

st.title('Car Price Prediction') 

# Input for categorical features 
oem = st.selectbox("OEM", options=b_encoders['oem'].unique()) 
oem_df= b_encoders[b_encoders["oem"]== oem]
fuel_type = st.selectbox("Fuel Type", options=oem_df['fuel type'].unique())
body_type = st.selectbox("Body Type", options=oem_df['Body type'].unique())
transmission = st.selectbox("Transmission", options=oem_df['transmission'].unique())
ownership = st.selectbox("Ownership", options=oem_df['Ownership'].unique()) 
city = st.selectbox("City", options=oem_df['City'].unique())
engine_category = st.selectbox("Engine Category", options=oem_df['Engine_category'].unique()) 
seats = st.selectbox("Number of Seats", options=oem_df["Seats"].unique())

# Input for numerical features
kms_driven = st.slider("KMS Driven", min_value=oem_df["Kms Driven"].min(),max_value=oem_df["Kms Driven"].max())  
max_power = st.slider("Max Power (in bhp)", min_value=oem_df["Max Power"].min(),max_value=oem_df["Max Power"].max())
torque = st.slider("Torque (in Nm)", min_value=oem_df["Torque"].min(),max_value=oem_df["Torque"].max())
mileage = st.slider("Mileage (in kmpl)", min_value=oem_df["Mileage"].min(),max_value=oem_df["Mileage"].max())
car_age = st.slider("Car Age (in years)", min_value=oem_df["Car_Age"].min(),max_value=oem_df["Car_Age"].max()) 


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






























