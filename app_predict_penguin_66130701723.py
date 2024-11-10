
import numpy as np 
import pandas as pd 
import streamlit as st
import pickle

# Load model and encoders
with open('model_penguin_66130701723.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Title for the Streamlit app
st.title('Penguin Species Prediction App')

# Load your DataFrame directly from a CSV file
# Replace 'Uncleaned_employees_final_dataset.csv' with the actual file name or path
df = pd.read_csv('penguins_size.csv')
st.dataframe(df.head())  # Display the first few rows of the data

# Assuming you want to make predictions for each row in the DataFrame
for index, row in df.iterrows():
    # Extract values from the row (assuming column names match the expected input)
    island = row['island']
    sex = row['sex']
    bill_length_mm = row['bill_length_mm']
    bill_depth_mm = row['bill_depth_mm']
    flipper_length_mm = row['flipper_length_mm']
    body_mass_g = row['body_mass_g']

    # Categorical Data Encoding
    island_encoded = island_encoder.transform([island])[0]
    sex_encoded = sex_encoder.transform([sex])[0]

    # Prepare input data
    input_data = np.array([[bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island_encoded, sex_encoded]])

    # Make predictions
    prediction = model.predict(input_data)
    predicted_species = species_encoder.inverse_transform(prediction)
    st.write(f'Predicted species for row {index+1}: {predicted_species[0]}')
