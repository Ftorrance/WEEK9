import pandas as pd
import matplotlib.pyplot as plt

def add_moving_averages(df, short_window=20, long_window=50):
    """
    Adds Simple Moving Averages (SMA) to the stock DataFrame.

    Parameters:
        df (pd.DataFrame): Stock price data.
        short_window (int): Period for short SMA.
        long_window (int): Period for long SMA.

    Returns:
        pd.DataFrame: Data with additional SMA columns.
    """
    df['SMA_20'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_50'] = df['Close'].rolling(window=long_window).mean()
    return df

def add_rsi(df, window=14):
    """
    Adds the Relative Strength Index (RSI) to the DataFrame.

    Parameters:
        df (pd.DataFrame): Stock price data.
        window (int): Lookback period for RSI.

    Returns:
        pd.DataFrame: Data with RSI column.
    """
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Adds the MACD indicator to the DataFrame.

    Parameters:
        df (pd.DataFrame): Stock price data.
        short_window (int): Period for short EMA.
        long_window (int): Period for long EMA.
        signal_window (int): Signal line period.

    Returns:
        pd.DataFrame: Data with MACD and Signal columns.
    """
    df['EMA_12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

import pandas as pd
import matplotlib.pyplot as plt

def plot_indicators(df, ticker):
    plt.figure(figsize=(12, 8))

    # Price and Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label="Close Price", color="blue")
    plt.plot(df.index, df['SMA_20'], label="20-day SMA", color="red", linestyle="dashed")
    plt.plot(df.index, df['SMA_50'], label="50-day SMA", color="green", linestyle="dashed")
    plt.title(f"{ticker} - Price & Moving Averages")
    plt.legend()
    plt.grid()

    # RSI
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['RSI'], label="RSI", color="purple")
    plt.axhline(70, linestyle="dashed", color="red", alpha=0.5)  # Overbought level
    plt.axhline(30, linestyle="dashed", color="green", alpha=0.5)  # Oversold level
    plt.title(f"{ticker} - Relative Strength Index (RSI)")
    plt.legend()
    plt.grid()

    # MACD
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['MACD'], label="MACD", color="black")
    plt.plot(df.index, df['Signal'], label="Signal Line", color="red", linestyle="dashed")
    plt.title(f"{ticker} - MACD Indicator")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show(block=False)  # Allow other plots to open
