import os
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from datetime import datetime
import tensorflow as tf
import joblib

app = Flask(__name__)

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
        raise ValueError("The 'Close' prices data is empty.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    if not os.path.exists('model'):
        os.makedirs('model')
    joblib.dump(scaler, 'model/scaler.save')

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:]

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    class TrainingProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f'Epoch {epoch + 1} - Loss: {logs["loss"]:.4f}')

    print("Starting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        callbacks=[TrainingProgress()],
        verbose=1
    )

    model.save('model/lstm_model.h5')

    last_60_days = scaled_data[-60:]
    X_input = last_60_days.reshape(1, 60, 1)
    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_price)

    current_price = close_prices[-1][0]
    take_profit = predicted_price[0][0]

    price_difference = abs(take_profit - current_price)
    stop_loss_distance = price_difference / 3

    if take_profit > current_price:
        signal = "Buy"
        stop_loss = current_price - stop_loss_distance
    else:
        signal = "Sell"
        stop_loss = current_price + stop_loss_distance

    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-100:], close_prices[-100:], label='Current Price', color='blue')
    plt.axhline(y=current_price, color='orange', linestyle='--', label='Current Price')
    plt.axhline(y=take_profit, color='green', linestyle='--', label='Take Profit (Predicted Price)')
    plt.axhline(y=stop_loss, color='red', linestyle='--', label='Stop Loss')
    plt.title(f"{ticker} Price Chart (Real-Time - {interval})")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()

    if not os.path.exists('static'):
        os.makedirs('static')
    chart_path = 'static/chart.png'
    plt.savefig(chart_path)
    plt.close()

    mae, win_rate = calculate_metrics(model, X_test, y_test, scaler, close_prices[train_size:])

    return (f"Predicted Price (Take Profit): {take_profit:.2f} | Current Price: {current_price:.2f} | "
            f"Stop Loss: {stop_loss:.2f} | Signal: {signal}"), \
            chart_path, mae, win_rate

def calculate_metrics(model, X_test, y_test, scaler, actual_prices):
    try:
        y_pred = model.predict(X_test)

        y_pred_transformed = scaler.inverse_transform(y_pred)
        y_test_transformed = scaler.inverse_transform([y_test])[0]

        mae = np.mean(np.abs(y_pred_transformed - y_test_transformed.reshape(-1, 1)))

        signals = [
            "Buy" if y_pred_transformed[i + 1] > actual_prices[i] else "Sell"
            for i in range(len(y_pred_transformed) - 1)
        ]

        total_trades = len(signals)
        winning_trades = sum(
            1 for i in range(total_trades)
            if (
                (signals[i] == "Buy" and actual_prices[i + 1] > actual_prices[i]) or
                (signals[i] == "Sell" and actual_prices[i + 1] < actual_prices[i])
            )
        )

        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        return mae, win_rate

    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        return 0, 0

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
        message, chart_path, mae, win_rate = train_model(ticker, interval)
        return render_template('result.html',
                             message=message,
                             chart_path=chart_path,
                             mae=f"{mae:.4f}" if mae is not None else None,
                             win_rate=f"{win_rate:.2f}" if win_rate is not None else None,
                             ticker=ticker)
    except Exception as e:
        return render_template('result.html',
                             message=f"Error: {str(e)}",
                             chart_path=None,
                             mae=None,
                             win_rate=None)

if __name__ == '__main__':
    app.run()
