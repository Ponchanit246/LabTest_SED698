
import numpy as np 
import pandas as pd 
import streamlit as st
import pickle

# Load model and encoders from the .pkl file
with open('model_penguin_66130701723.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Title for the Streamlit app
st.title('Penguin Species Prediction App')

# User inputs
st.header('Enter the penguin characteristics for prediction:')
island = st.selectbox('Island', island_encoder.classes_)
sex = st.selectbox('Sex', sex_encoder.classes_)
bill_length_mm = st.number_input('Bill Length (mm)', min_value=0.0, step=0.1)
bill_depth_mm = st.number_input('Bill Depth (mm)', min_value=0.0, step=0.1)
flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0, step=1.0)
body_mass_g = st.number_input('Body Mass (g)', min_value=0.0, step=10.0)

# Prepare data for prediction
# Encoding categorical data
island_encoded = island_encoder.transform([island])[0]
sex_encoded = sex_encoder.transform([sex])[0]

# Prepare input data
input_data = np.array([[bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island_encoded, sex_encoded]])

# Ensure input data is a 2D array
if input_data.ndim == 1:
    input_data = input_data.reshape(1, -1)

# Check for NaN values
if np.any(pd.isnull(input_data)):
    st.error("Input data contains missing values. Please fill all input fields.")
else:
    # Make predictions
    prediction = model.predict(input_data)
    predicted_species = species_encoder.inverse_transform(prediction)
    st.success(f'The predicted species is: {predicted_species[0]}')


# Make predictions upon button click
if st.button('Predict'):
    prediction = model.predict(input_data)
    predicted_species = species_encoder.inverse_transform(prediction)
    st.success(f'The predicted species is: {predicted_species[0]}')
