# import the libraries
import pandas as pd
from tensorflow.python import tf2
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.express as px

start = "2010-01-01"
end = "2022-01-31"


st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = yf .download(user_input, start, end)

#Describing Data
st.subheader("Data from 2010-2021")
st.write(df.describe())

#Visualizations
st.subheader("Closing Price vs Time Chart")
fig = px.line(x= df.index, y=df.Close)
fig.update_layout(xaxis_title='Date',
                yaxis_title='Closing Price')
st.plotly_chart(fig)

ma100 = df.Close.rolling(100).mean()
# Visualize the closing prices and 100days MA
st.subheader("Closing Price vs Time Chart with 100MA")
fig = px.line(x=df.index, y=df.Close)
fig.add_scatter(x=df.index, y=ma100, mode='lines', name ="MA 100days")
fig.update_layout(xaxis_title='Date',
                yaxis_title='Close Price and 100days MA')
st.plotly_chart(fig)

ma200 = df.Close.rolling(200).mean()
st.subheader("Closing Price vs Time Chart with 200MA")
# Visualize the closing prices and 200days MA
fig = px.line(x = df.index, y=df.Close)
fig.add_scatter(x=df.index, y=ma100, mode='lines', name ="MA 100days")
fig.add_scatter(x=df.index, y=ma200, mode='lines', name ="MA 200days")
fig.update_layout(xaxis_title='Date',
                yaxis_title='Close Price and 200days MA')
st.plotly_chart(fig)

# splitting data into training and testing
data_training = pd.DataFrame(df["Close"][0 : int(len(df)*0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70) : ])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load my model
model = load_model("keras_modell.h5")

# Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100 : i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test),  np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, "b", label ="Original Price")
plt.plot (y_predicted, "r", label ="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

