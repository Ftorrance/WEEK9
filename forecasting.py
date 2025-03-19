import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def train_arima(df, order=(5, 1, 0), steps=30):
    """
    Trains an ARIMA model and forecasts stock prices.

    Parameters:
        df (pd.DataFrame): Stock price data.
        order (tuple): ARIMA order (p, d, q).
        steps (int): Number of future days to predict.

    Returns:
        pd.Series: Forecasted values.
    """
    try:
        # Train ARIMA model
        model = ARIMA(df['Close'], order=order)
        model_fit = model.fit()

        # Forecast future prices
        forecast = model_fit.forecast(steps=steps)

        return forecast
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        return None

def plot_forecast(df, forecast, steps=30):
    """
    Plots historical stock prices along with the ARIMA forecast.

    Parameters:
        df (pd.DataFrame): Stock price data.
        forecast (pd.Series): Forecasted prices.
        steps (int): Number of future days to predict.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Close'], label="Historical Prices", color="blue")

    # Generate future date range
    future_dates = pd.date_range(df.index[-1], periods=steps+1, freq="B")[1:]

    # Plot forecast
    plt.plot(future_dates, forecast, label="ARIMA Forecast", color="red", linestyle="dashed")
    
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title("Stock Price Forecast Using ARIMA")
    plt.legend()
    plt.grid()
    plt.show(block=False)
