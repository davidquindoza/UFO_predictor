import streamlit as st
import os
import numpy as np
import pickle

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model file
model_path = os.path.join(script_directory, "ufo-model.pkl")

# Load the machine learning model
model = pickle.load(open(model_path, "rb"))

st.title('🛸 UFO Predictor App 👽')

st.markdown("""
This app performs a simple prediction on where the nearest location of a UFO appearance 
has occured using a regression model. 

""")

with st.form("my_form"):
    seconds = st.slider("Form slider", min_value=0.0, max_value=60.0)
    
    latitude = st.text_input('Latitude:')
    
    longitude = st.text_input('Longitude:')

    submitted = st.form_submit_button('Predict the Location!')

if submitted:
    # Convert latitude and longitude to float
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        st.error("Please enter valid latitude and longitude.")
        st.stop()

    # Perform prediction using the loaded model and user input
    int_features = [seconds, latitude, longitude]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # Define country labels
    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    # Display the prediction result
    st.write(f'Likely country: {countries[int(prediction[0])]}')
