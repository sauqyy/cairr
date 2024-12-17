import os
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from datetime import datetime
import tensorflow as tf
import pandas as pd
import joblib

app = Flask(__name__)

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])  # Multiple features
        y.append(data[i + time_step, 0])  # Predicting the closing price
    return np.array(X), np.array(y)

def add_technical_indicators(data):
    # Add RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Add Moving Average
    data['MA'] = data['Close'].rolling(14).mean()
    
    # Fill missing values
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='ffill', inplace=True)
    
    return data

def train_model(ticker, interval):
    # Download data
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start='2020-01-01', end=end_date, interval=interval)

    if data.empty:
        raise ValueError("No data was downloaded. Please check the ticker symbol or date range.")
    
    # Add technical indicators
    data = add_technical_indicators(data)

    # Prepare features (Close, RSI, MA)
    features = data[['Close', 'RSI', 'MA']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    # Save scaler
    if not os.path.exists('model'):
        os.makedirs('model')
    joblib.dump(scaler, 'model/scaler.save')

    # Create training and testing datasets
    train_size = int(len(scaled_features) * 0.8)
    train_data = scaled_features[:train_size]
    test_data = scaled_features[train_size - 60:]

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    # Reshape input to [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Build the model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(60, X_train.shape[2])))
    model.add(Dropout(0.2))  # Dropout layer
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))  # Dropout layer
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))  # L2 regularization
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    # Save the model
    model.save('model/lstm_model.h5')

    # Predict next price
    last_60_days = scaled_features[-60:]
    X_input = last_60_days.reshape(1, 60, X_train.shape[2])
    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(
        np.concatenate([predicted_price, np.zeros((1, 2))], axis=1)
    )[0][0]

    # Plot the chart
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-100:], data['Close'][-100:], label='Close Price', color='blue')
    plt.axhline(y=predicted_price, color='red', linestyle='--', label='Predicted Price')
    plt.title(f"{ticker} Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()

    if not os.path.exists('static'):
        os.makedirs('static')
    chart_path = 'static/chart.png'
    plt.savefig(chart_path)
    plt.close()

    return f"Predicted Price: {predicted_price:.2f}", chart_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def new_page():
    return render_template('search.html')


@app.route('/train', methods=['POST'])
def train():
    ticker = request.form['ticker']
    interval = request.form['timeframe']
    try:
        message, chart_path = train_model(ticker, interval)
        return render_template('result.html', message=message, chart_path=chart_path)
    except Exception as e:
        return render_template('result.html', message=f"Error: {str(e)}", chart_path=None)

if __name__ == '__main__':
    app.run(debug=True)
