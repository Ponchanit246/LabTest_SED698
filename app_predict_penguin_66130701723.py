
import streamlit as st
import pickle
import pandas as pd

# Load the model and encoders from the pickle file
with open('model_penguin_66130701723.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

st.title("Penguin Species Prediction App")

# Collect input from the user
island = st.selectbox("Island", ['Torgersen', 'Biscoe', 'Dream'])
culmen_length = st.number_input("Culmen Length (mm)", min_value=20.0, max_value=70.0, step=0.1)
culmen_depth = st.number_input("Culmen Depth (mm)", min_value=10.0, max_value=30.0, step=0.1)
flipper_length = st.number_input("Flipper Length (mm)", min_value=150.0, max_value=250.0, step=0.1)
body_mass = st.number_input("Body Mass (g)", min_value=2000, max_value=7000, step=50)
sex = st.selectbox("Sex", ['MALE', 'FEMALE'])

# Transform inputs
x_new = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length],
    'culmen_depth_mm': [culmen_depth],
    'flipper_length_mm': [flipper_length],
    'body_mass_g': [body_mass],
    'sex': [sex]
})

# Apply encoding transformations
try:
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])
except Exception as e:
    st.error(f"Encoding error: {e}")

st.write("Transformed Input Data:", x_new)

# Ensure feature order matches training data
x_new = x_new[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]

# Make prediction
if st.button("Predict"):
    try:
        y_pred_new = model.predict(x_new)
        result = species_encoder.inverse_transform(y_pred_new)
        st.write('Predicted Species:', result[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
