from data_loader import load_stock_data
from visualization import plot_stock_price, decompose_time_series
from indicators import add_moving_averages, add_rsi, add_macd, plot_indicators
from forecasting import train_arima, plot_forecast
from evaluation import evaluate_forecast
import matplotlib.pyplot as plt

# Load stock data
ticker = "AAPL"
df = load_stock_data(ticker)

# Apply technical indicators
if df is not None:
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)

    # Plot indicators
    plot_stock_price(df, ticker)
    decompose_time_series(df, ticker)
    plot_indicators(df, ticker)

    # Train ARIMA model and make predictions
    forecast = train_arima(df, steps=30)

    if forecast is not None:
        plot_forecast(df, forecast)
        evaluate_forecast(df, forecast)

    # Keep all plots open
    plt.pause(0.1)
    input("Press Enter to close all plots...")  # Prevents auto-closing
else:
    print("Failed to load stock data.")
