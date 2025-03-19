import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_stock_price(df, ticker):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Close'], label=f"{ticker} Stock Price", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{ticker} Stock Closing Price Over Time")
    plt.legend()
    plt.grid()
    plt.show(block=False)  # Allow other plots to open

def decompose_time_series(df, ticker, period=252):
    result = seasonal_decompose(df['Close'], model='additive', period=period)

    plt.figure(figsize=(12, 8))
    
    plt.subplot(411)
    plt.plot(df['Close'], label='Original', color='blue')
    plt.legend()

    plt.subplot(412)
    plt.plot(result.trend, label='Trend', color='red')
    plt.legend()

    plt.subplot(413)
    plt.plot(result.seasonal, label='Seasonality', color='green')
    plt.legend()

    plt.subplot(414)
    plt.plot(result.resid, label='Residuals', color='black')
    plt.legend()

    plt.suptitle(f"Time Series Decomposition of {ticker}", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)  # Allow other plots to open
