
import streamlit as st
import pickle
import numpy as np

# Load the trained RandomForest model and scaler
with open('crop.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('sccrop.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create the web app
st.title('Crop Recommendation App')

# Input fields
N = st.number_input('Nitrogen', min_value=0.0, value=0.0, step=1.0)
P = st.number_input('Phosphorus', min_value=0.0, value=0.0, step=1.0)
K = st.number_input('Potassium', min_value=0.0, value=30.0, step=1.0)
temperature = st.number_input('Temperature', min_value=0.0, value=25.0, step=0.1)
humidity = st.number_input('Humidity', min_value=0.0, value=50.0, step=0.1)
ph = st.number_input('PH', min_value=0.0, value=6.5, step=0.1)
rainfall = st.number_input('Rainfall', min_value=0.0, value=100.0, step=1.0)

# Button to trigger prediction
if st.button('Predict Crop'):
    # Prepare the feature vector
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=np.float64)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Crop Recommendation
    predicted_crop = model.predict(features_scaled)

    # Display the prediction
    st.write(f'Predicted Crop: {predicted_crop[0]}')
