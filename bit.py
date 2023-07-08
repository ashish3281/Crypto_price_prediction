import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA

# Load the cryptocurrency price data
cryptos = {
    'Bitcoin': 'BTC-USD.csv',
    'Ethereum': 'ETH-USD.csv',
}

selected_crypto = st.sidebar.selectbox('Select Cryptocurrency', list(cryptos.keys()))
crypto_file = cryptos[selected_crypto]
data = pd.read_csv(crypto_file)

# Create a train and test split
train_size = int(len(data) * 0.75)
test_size = len(data) - train_size
train_data = data[:train_size]
test_data = data[train_size:]

# Convert the data to NumPy arrays
train_data_np = train_data['Close'].values
test_data_np = test_data['Close'].values

# Convert the NumPy arrays to Tensors
train_data_tensor = tf.convert_to_tensor(train_data_np)
test_data_tensor = tf.convert_to_tensor(test_data_np)

# Reshape the Tensors to match the expected input shape of the LSTM model
train_data_tensor = tf.reshape(train_data_tensor, (train_data_tensor.shape[0], 1, 1))
test_data_tensor = tf.reshape(test_data_tensor, (test_data_tensor.shape[0], 1, 1))

# Create a LSTM model
def create_lstm_model():
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=(1, 1)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(32),
        keras.layers.Dense(16),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])
    return model

# Create an ARIMA model
def create_arima_model():
    model = ARIMA(train_data['Close'], order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit

# Train and make predictions based on the selected model
selected_model = st.sidebar.selectbox('Select Model', ['LSTM', 'ARIMA'])

if selected_model == 'LSTM':
    # Train the LSTM model
    model = create_lstm_model()
    epochs = st.sidebar.slider('Number of Epochs', min_value=1, max_value=200, value=100, step=10)
    history = model.fit(train_data_tensor, train_data_tensor, epochs=epochs, batch_size=32)

    # Make predictions
    predictions = model.predict(test_data_tensor)
elif selected_model == 'ARIMA':
    # Train the ARIMA model
    model = create_arima_model()

    # Make predictions
    predictions = model.forecast(test_size)

# Calculate the root mean squared error
rmse = np.sqrt(np.mean((predictions - test_data['Close'].values)**2))

# Display the predictions in Streamlit
st.title(f'{selected_crypto} Stock Price Prediction')

# Model Performance
st.subheader('Model Performance')
st.text(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Plot the actual prices
st.subheader(f'Actual {selected_crypto} Stock Prices')
plt.figure(figsize=(12, 6))
plt.plot(test_data['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f'Actual {selected_crypto} Stock Prices')
st.pyplot(plt)

# Plot the predicted prices
st.subheader(f'{selected_crypto} Stock Price Predictions')
plt.figure(figsize=(12, 6))
plt.plot(test_data['Date'], predictions, label='Predicted')
plt.plot(test_data['Date'], test_data['Close'], label='Actual')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f'{selected_crypto} Stock Price Predictions ({selected_model})')
plt.legend()
st.pyplot(plt)

# Interactive Chart
st.subheader('Interactive Chart')
num_points = st.slider('Number of Points', min_value=10, max_value=len(test_data), value=100, step=10)
selected_actual = test_data['Close'][:num_points]
selected_predicted = predictions[:num_points]

plt.figure(figsize=(12, 6))
plt.plot(selected_actual, label='Actual')
plt.plot(selected_predicted, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'{selected_crypto} Stock Price Predictions ({selected_model})')
plt.legend()
st.pyplot(plt)

# Data Table
st.subheader('Data Table')
st.write(test_data.head(num_points))
