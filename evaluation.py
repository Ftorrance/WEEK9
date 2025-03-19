import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(df, forecast, steps=30):
    """
    Evaluates the ARIMA forecast using MAE and RMSE.

    Parameters:
        df (pd.DataFrame): Stock price data.
        forecast (pd.Series): Forecasted prices.
        steps (int): Number of days forecasted.

    Returns:
        None (prints MAE and RMSE).
    """
    try:
        # Get actual values for comparison
        actual = df['Close'][-steps:]

        # Ensure forecast and actual are aligned
        if len(actual) != len(forecast):
            print("Warning: Forecast and actual data lengths do not match.")
            return

        # Calculate error metrics
        mae = mean_absolute_error(actual, forecast)
        rmse = np.sqrt(mean_squared_error(actual, forecast))

        # Print results
        print(f"\n🔍 Model Performance Evaluation:")
        print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
        print(f"📊 Root Mean Squared Error (RMSE): {rmse:.2f}")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
