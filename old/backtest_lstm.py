import sys
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime

ticker = sys.argv[1] if len(sys.argv) > 1 else 'BTC-USD'

data = yf.download(ticker, start='2022-01-01', end='2024-12-01')
if data.empty:
    raise ValueError(f"Failed to fetch data for {ticker}. Please check the ticker symbol.")
close_prices = data['Close'].values.reshape(-1, 1)

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

signals = []
for i in range(len(predicted_prices) - 1):
    if predicted_prices[i + 1] > close_prices[i]:
        signals.append("Buy")
    else:
        signals.append("Sell")

initial_balance = 10000
balance = initial_balance
position = 0
pure_profit = 0
pure_loss = 0
take_profit_percentage = 5
stop_loss_percentage = 3
trade_log = []
total_trades = 0
winning_trades = 0
stop_loss_hits = 0

for i in range(len(signals)):
    if signals[i] == "Buy" and balance >= close_prices[i]:
        buy_price = float(close_prices[i])
        position = balance / buy_price
        balance = 0
        take_profit_price = buy_price * (1 + take_profit_percentage / 100)
        stop_loss_price = buy_price * (1 - stop_loss_percentage / 100)
        trade_log.append(f"Buy at {buy_price:.2f}, Take Profit set at {take_profit_price:.2f}, Stop Loss set at {stop_loss_price:.2f}")

    elif signals[i] == "Sell" and position > 0:
        current_price = float(close_prices[i])
        if current_price >= take_profit_price:
            sell_price = take_profit_price
            trade_log.append(f"Sell at {take_profit_price:.2f} (Take Profit reached)")
            winning_trades += 1
        elif current_price <= stop_loss_price:
            sell_price = stop_loss_price
            trade_log.append(f"Sell at {stop_loss_price:.2f} (Stop Loss reached)")
            stop_loss_hits += 1
        else:
            sell_price = current_price
            trade_log.append(f"Sell at {current_price:.2f}")

        sell_value = position * sell_price
        profit_or_loss = sell_value - (position * buy_price)

        if profit_or_loss > 0:
            pure_profit += profit_or_loss
        else:
            pure_loss += abs(profit_or_loss)

        balance = sell_value
        position = 0
        total_trades += 1

total_profit_loss = pure_profit - pure_loss

if total_trades > 0:
    win_rate = (winning_trades / total_trades) * 100
else:
    win_rate = 0

print(f"Win Rate: {win_rate:.2f}%")
print(f"Total Profit-Loss: ${total_profit_loss:.2f}")
