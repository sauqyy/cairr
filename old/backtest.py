import yfinance as yf

def calculate_win_rate(ticker, start_date="2022-01-01", end_date="2023-12-31", ratio=3):
    # Download historical data
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

    if data.empty:
        raise ValueError("No data was downloaded. Please check the ticker symbol or date range.")

    close_prices = data['Close'].values
    total_trades = 0
    wins = 0
    losses = 0

    for i in range(len(close_prices) - 1):  # Loop through each day
        current_price = float(close_prices[i])
        predicted_price = float(close_prices[i + 1])  # Assume simple next-day price as "prediction"

        # Calculate take profit and stop loss
        price_difference = abs(predicted_price - current_price)
        stop_loss_distance = price_difference / ratio

        if predicted_price > current_price:  # Signal: Buy
            take_profit = predicted_price
            stop_loss = current_price - stop_loss_distance

            # Debugging logs
            print(f"Trade {i+1}: BUY - Current: {current_price:.2f}, Predicted: {predicted_price:.2f}, "
                  f"Take Profit: {take_profit:.2f}, Stop Loss: {stop_loss:.2f}")

            # Check conditions
            if close_prices[i + 1] >= take_profit:  # Hit take profit
                wins += 1
                print(f"Result: WIN (Take Profit Hit on Day {i+2})")
            elif close_prices[i + 1] <= stop_loss:  # Hit stop loss
                losses += 1
                print(f"Result: LOSS (Stop Loss Hit on Day {i+2})")

        else:  # Signal: Sell
            take_profit = predicted_price
            stop_loss = current_price + stop_loss_distance

            # Debugging logs
            print(f"Trade {i+1}: SELL - Current: {current_price:.2f}, Predicted: {predicted_price:.2f}, "
                  f"Take Profit: {take_profit:.2f}, Stop Loss: {stop_loss:.2f}")

            # Check conditions
            if close_prices[i + 1] <= take_profit:  # Hit take profit
                wins += 1
                print(f"Result: WIN (Take Profit Hit on Day {i+2})")
            elif close_prices[i + 1] >= stop_loss:  # Hit stop loss
                losses += 1
                print(f"Result: LOSS (Stop Loss Hit on Day {i+2})")

        total_trades += 1

    # Calculate win rate
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    # Debugging final results
    print(f"Total Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%")

    return win_rate, total_trades, wins, losses


# Example usage
ticker = "AAPL"  # Replace with desired ticker
win_rate, total_trades, wins, losses = calculate_win_rate(ticker)
print(f"Total Trades: {total_trades}")
print(f"Total Wins: {wins}")
print(f"Total Losses: {losses}")
print(f"Win Rate: {win_rate:.2f}%")
