import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

@st.cache_data
def load_data():
    data = pd.read_csv("Data.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Prev_Close'] = data.groupby('Index')['Close'].shift(1)
    data['Price_Range'] = data['High'] - data['Low']
    data['Price_Change'] = data['Close'] - data['Open']
    data.dropna(inplace=True)
    label_encoder = joblib.load("saved-model/label_encoder.joblib")
    scaler = joblib.load("saved-model/scaler.joblib")
    data['Index_Encoded'] = label_encoder.transform(data['Index'])
    
    return data, label_encoder, scaler

def prepare_prediction_input(encoded_index, open_price, high_price, low_price, prev_close):
    price_range = high_price - low_price
    price_change = open_price - prev_close
    input_data = pd.DataFrame({
        'Index_Encoded': [encoded_index],
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Prev_Close': [prev_close],
        'Price_Range': [price_range],
        'Price_Change': [price_change]
    })
    
    return input_data

st.title("StockSage: Next Day Stock Price Predictor")

data, label_encoder, scaler = load_data()
st.write("### Dataset Overview")
st.write(data.head())

@st.cache_resource
def load_saved_model():
    return joblib.load("saved-model/random_forest_model.joblib")

st.write("### Predict the Next Day's Closing Price")
index_choice = st.selectbox("Select Index", label_encoder.classes_)
open_price = st.number_input("Enter Open Price", value=0.0)
high_price = st.number_input("Enter High Price", value=0.0)
low_price = st.number_input("Enter Low Price", value=0.0)
prev_close_price = st.number_input("Enter Previous Close Price", value=0.0)

if st.button("Predict"):
    encoded_index = label_encoder.transform([index_choice])[0]
    user_input = prepare_prediction_input(
        encoded_index, open_price, high_price, low_price, prev_close_price
    )
    user_input_scaled = scaler.transform(user_input)
    model = load_saved_model()
    prediction = model.predict(user_input_scaled)
    st.write(f"Predicted Close Price: {prediction[0]:.2f}")

if index_choice:
    st.write(f"### Historical Data for {index_choice}")
    index_data = data[data['Index'] == index_choice].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=index_data, x='Date', y='Close', ax=ax)
    ax.set_title(f"{index_choice} Historical Close Prices")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)