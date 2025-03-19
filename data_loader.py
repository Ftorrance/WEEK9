import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start="2020-01-01", end="2025-01-01"):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start (str): Start date in "YYYY-MM-DD" format.
        end (str): End date in "YYYY-MM-DD" format.

    Returns:
        pd.DataFrame: Stock price data with 'Close' prices.
    """
    try:
        # Download stock data
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)

        # Check if MultiIndex exists
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close'][ticker]  # Select only the 'Close' column for the ticker
        else:
            df = df['Close']  # Single ticker case

        # Convert to DataFrame for consistency
        df = df.to_frame(name='Close')

        # Drop missing values
        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
