import os
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from datetime import datetime

app = Flask(__name__)

# Fungsi create_dataset untuk menyiapkan data untuk pelatihan dan pengujian
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def train_model(ticker, interval):
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start='2020-01-01', end=end_date, interval=interval)

    if data.empty:
        raise ValueError("No data was downloaded. Please check the ticker symbol or date range.")

    close_prices = data['Close'].values.reshape(-1, 1)
    if close_prices.size == 0:
        raise ValueError("The 'Close' prices data is empty. Please check the ticker symbol or date range.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:]

    X_train, y_train = create_dataset(train_data)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    if not os.path.exists('model'):
        os.makedirs('model')
    model.save('model/lstm_model.h5')

    last_60_days = scaled_data[-60:]
    X_input = last_60_days.reshape(1, 60, 1)
    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_price)

    current_price = close_prices[-1][0]
    price_difference = predicted_price[0][0] - current_price
    percentage_change = (price_difference / current_price) * 100

    if percentage_change > 2:
        signal = "Strong Buy"
    elif 0 < percentage_change <= 2:
        signal = "Buy"
    elif -2 <= percentage_change < 0:
        signal = "Sell"
    else:
        signal = "Strong Sell"

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, close_prices, label='Current Price', color='blue')
    plt.axhline(y=current_price, color='orange', linestyle='--', label='Current Price')
    plt.axhline(y=predicted_price[0][0], color='red', linestyle='--', label='Predicted Price')
    plt.title(f"{ticker} Price Chart (Real-Time - {interval})")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    chart_path = 'static/chart.png'
    plt.savefig(chart_path)
    plt.close()

    return f"Predicted Price: {predicted_price[0][0]:.2f} | Current Price: {current_price:.2f} | Signal: {signal}", chart_path

def calculate_win_rate_and_mae(ticker, interval):
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start='2020-01-01', 
                       nd=end_date, interval=interval)

    close_prices = data['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

    train_size = int(len(scaled_data) * 0.8)
    test_data = scaled_data[train_size - 60:]  # Data test termasuk 60 data sebelumnya

    X_test, y_test = create_dataset(test_data)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = load_model('model/lstm_model.h5')
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Sesuaikan panjang close_prices dengan panjang predicted_prices
    actual_prices = close_prices[train_size + 60:]  # Mulai dari data test

    if len(predicted_prices) != len(actual_prices):
        print(f"Warning: Length mismatch between predicted and actual prices.")
        print(f"Predicted: {len(predicted_prices)}, Actual: {len(actual_prices)}")
        return None, None

    # Calculate MAE
    mae = np.mean(np.abs(predicted_prices - actual_prices))

    # Calculate Win Rate
    signals = []
    for i in range(len(predicted_prices) - 1):
        if predicted_prices[i + 1] > actual_prices[i]:
            signals.append("Buy")
        else:
            signals.append("Sell")

    initial_balance = 10000
    balance = initial_balance
    position = 0
    total_trades = 0
    winning_trades = 0
    take_profit_percentage = 5
    stop_loss_percentage = 3

    for i in range(len(signals)):
        if signals[i] == "Buy" and balance >= actual_prices[i]:
            buy_price = actual_prices[i]
            position = balance / buy_price
            balance = 0
            take_profit_price = buy_price * (1 + take_profit_percentage / 100)
            stop_loss_price = buy_price * (1 - stop_loss_percentage / 100)

        elif signals[i] == "Sell" and position > 0:
            sell_price = actual_prices[i]
            if sell_price >= take_profit_price:
                winning_trades += 1
            balance = position * sell_price
            position = 0
            total_trades += 1

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    return mae, win_rate

@app.route('/')
def home():
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
        mae, win_rate = calculate_win_rate_and_mae(ticker, interval)
    except ValueError as e:
        message = str(e)
        chart_path = None
        mae = None
        win_rate = None
    return render_template('result.html', message=message, chart_path=chart_path, mae=mae, win_rate=win_rate)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
