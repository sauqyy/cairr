from flask import Flask, render_template, request
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Fungsi untuk melatih model dan menghitung winrate
def train_model_and_calculate_winrate(ticker, interval):
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start='2020-01-01', end=end_date, interval=interval)

    if data.empty:
        raise ValueError("No data was downloaded. Please check the ticker symbol or date range.")

    close_prices = data['Close'].values.reshape(-1, 1)
    if close_prices.size == 0:
        raise ValueError("The 'Close' prices data is empty. Please check the ticker symbol or date range.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)

    predicted_prices = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Menghitung winrate
    total_trades = 0
    winning_trades = 0

    for i in range(1, len(predicted_prices)):
        actual_change = close_prices[i] - close_prices[i - 1]
        predicted_change = predicted_prices[i] - predicted_prices[i - 1]

        if (predicted_change > 0 and actual_change > 0) or (predicted_change < 0 and actual_change < 0):
            winning_trades += 1
        total_trades += 1

    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
    else:
        win_rate = 0

    return win_rate, predicted_prices, close_prices

# Route utama
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def new_page():
    return render_template('search.html')

# Route untuk melatih model
@app.route('/train', methods=['POST'])
def train():
    ticker = request.form['ticker']
    interval = request.form['timeframe']
    try:
        win_rate, predicted_prices, close_prices = train_model_and_calculate_winrate(ticker, interval)
        message = f"Training completed. Win Rate: {win_rate:.2f}%"
        chart_path = 'static/chart.png'

        # Membuat grafik
        plt.figure(figsize=(10, 5))
        plt.plot(close_prices, label='Actual Prices', color='blue')
        plt.plot(predicted_prices, label='Predicted Prices', color='red')
        plt.title(f"{ticker} Prediction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
    except ValueError as e:
        message = str(e)
        chart_path = None
        win_rate = None
    return render_template('result.html', message=message, chart_path=chart_path, win_rate=win_rate)

if __name__ == '__main__':
    app.run(debug=True)
