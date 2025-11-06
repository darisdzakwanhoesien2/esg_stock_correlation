# modules/data_loader.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ----------------------------------------------------------
# 1️⃣ Load ESG data
# ----------------------------------------------------------
def load_esg_data(filename: str = "esg_scores.csv") -> pd.DataFrame:
    """
    Loads ESG scores from a CSV file located in /data directory.
    Returns a pandas DataFrame with standardized column names.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    
    return df


# ----------------------------------------------------------
# 2️⃣ Download stock price data using yfinance
# ----------------------------------------------------------
def download_stock_data(ticker: str, start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    """
    Downloads daily stock data using yfinance API.
    Adds daily returns and volatility columns.
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    df.reset_index(inplace=True)
    df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    df["Ticker"] = ticker
    df["Daily_Return"] = df["Adj_Close"].pct_change()
    df["Volatility_7D"] = df["Daily_Return"].rolling(window=7).std()
    df["Volatility_30D"] = df["Daily_Return"].rolling(window=30).std()

    return df


# ----------------------------------------------------------
# 3️⃣ Save stock data for caching
# ----------------------------------------------------------
def save_stock_data(df: pd.DataFrame, filename: str = "stock_prices.csv"):
    """
    Saves stock data to /data directory.
    """
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"✅ Saved stock data to {path}")


# ----------------------------------------------------------
# 4️⃣ Load cached stock data
# ----------------------------------------------------------
def load_cached_stock_data(filename: str = "stock_prices.csv") -> pd.DataFrame:
    """
    Loads cached stock prices from /data.
    """
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        return df
    else:
        print(f"⚠️ No cached stock data found. Please download first.")
        return pd.DataFrame()
