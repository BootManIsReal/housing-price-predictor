import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('housing_model.pkl')

def preprocess_input(data):
    data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)
    model_columns = model.feature_names_in_
    data = data[model_columns]
    return data

def main():
    st.title("🏡 Housing Price Prediction App")
    st.write("This app predicts housing prices based on input features.")

    col1, col2 = st.columns(2)
    
    with col1:
        longitude = st.number_input("Longitude", value=-122.23)
        latitude = st.number_input("Latitude", value=37.88)
        housing_median_age = st.number_input("Housing Median Age", value=41.0)
        total_rooms = st.number_input("Total Rooms", value=880.0)
        total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
    
    with col2:
        population = st.number_input("Population", value=322.0)
        households = st.number_input("Households", value=126.0)
        median_income = st.number_input("Median Income", value=8.3252)
        ocean_proximity = st.selectbox("Ocean Proximity", ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"])

    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })

    input_data = preprocess_input(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.subheader("Prediction")
        st.write(f"The predicted housing price is: **${prediction[0]:,.2f}**")

if __name__ == "__main__":
    main()
