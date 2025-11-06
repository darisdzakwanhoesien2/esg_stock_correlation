# modules/preprocessor.py
"""
Preprocessor Module for ESG–Stock Correlation App
-------------------------------------------------
Aligns, merges, and enriches ESG and stock datasets for analysis.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ==========================================================
# 1️⃣ Load ESG and Stock Data
# ==========================================================
def load_data(esg_file="esg_scores.csv", stock_file="stock_prices.csv"):
    esg_path = os.path.join(DATA_DIR, esg_file)
    stock_path = os.path.join(DATA_DIR, stock_file)

    esg_df = pd.read_csv(esg_path, parse_dates=["Date"])
    stock_df = pd.read_csv(stock_path, parse_dates=["Date"])

    print(f"📄 Loaded ESG data: {len(esg_df)} rows, {esg_df['Ticker'].nunique()} companies")
    print(f"📄 Loaded Stock data: {len(stock_df)} rows, {stock_df['Ticker'].nunique()} companies")

    return esg_df, stock_df


# ==========================================================
# 2️⃣ Temporal Alignment (Expand annual ESG → daily)
# ==========================================================
def align_esg_to_daily(esg_df, stock_df):
    """
    Align ESG scores (annual) to match the daily frequency of stock data.
    Each company's ESG score is forward-filled across trading days.
    """
    daily_records = []
    tickers = sorted(esg_df["Ticker"].unique())

    for ticker in tickers:
        stock_dates = stock_df.loc[stock_df["Ticker"] == ticker, "Date"]
        esg_company = esg_df[esg_df["Ticker"] == ticker].sort_values("Date")

        if esg_company.empty or stock_dates.empty:
            continue

        # Expand ESG data to full daily date range (based on stock dates)
        esg_daily = pd.DataFrame({"Date": stock_dates})
        esg_daily = esg_daily.merge(esg_company, how="left", on="Date")

        # Forward-fill ESG scores until next annual value
        esg_daily = esg_daily.ffill().bfill()
        esg_daily["Ticker"] = ticker
        daily_records.append(esg_daily)

    esg_expanded = pd.concat(daily_records, ignore_index=True)
    print(f"🕓 Expanded ESG data to daily frequency: {len(esg_expanded)} rows")
    return esg_expanded


# ==========================================================
# 3️⃣ Merge ESG + Stock Datasets
# ==========================================================
def merge_esg_stock(esg_df_daily, stock_df):
    """
    Merges daily ESG and stock datasets on Date + Ticker.
    """
    merged = pd.merge(stock_df, esg_df_daily, on=["Date", "Ticker"], how="inner")
    merged.sort_values(["Ticker", "Date"], inplace=True)
    print(f"🔗 Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    return merged


# ==========================================================
# 4️⃣ Feature Engineering
# ==========================================================
def add_features(df):
    """
    Adds derived ESG–stock relationship metrics:
    - ESG_Trend (yearly change)
    - ESG_ZScores (sector normalization)
    - RiskAdjustedReturn
    - ESG_Return_Ratio
    """
    # ESG Trend (rolling difference)
    df["ESG_Trend"] = df.groupby("Ticker")["Overall_ESG_Score"].diff()

    # Sector-wise z-score normalization for ESG
    def zscore(x): return (x - x.mean()) / x.std(ddof=0)
    df["E_ZScore"] = df.groupby("Sector")["Environmental_Score"].transform(zscore)
    df["S_ZScore"] = df.groupby("Sector")["Social_Score"].transform(zscore)
    df["G_ZScore"] = df.groupby("Sector")["Governance_Score"].transform(zscore)
    df["ESG_ZScore"] = df.groupby("Sector")["Overall_ESG_Score"].transform(zscore)

    # Risk-adjusted return = return / 30D volatility
    df["RiskAdjustedReturn"] = df["Daily_Return"] / (df["Volatility_30D"] + 1e-6)

    # ESG-Return ratio: how ESG movement scales vs. returns
    df["ESG_Return_Ratio"] = df["ESG_Trend"] / (df["Daily_Return"] + 1e-6)

    # Sector mean ESG for benchmarking
    sector_avg = df.groupby("Sector")["Overall_ESG_Score"].transform("mean")
    df["Sector_ESG_Avg"] = sector_avg
    df["Above_Sector_Avg"] = (df["Overall_ESG_Score"] > sector_avg).astype(int)

    print("⚙️ Added derived ESG-stock metrics and normalization features")
    return df


# ==========================================================
# 5️⃣ Export Clean Dataset
# ==========================================================
def export_clean_data(df, filename="merged_esg_stock.csv"):
    """
    Saves the processed merged dataset to /data.
    """
    out_path = os.path.join(DATA_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"✅ Clean merged dataset saved to: {out_path}")


# ==========================================================
# 6️⃣ End-to-End Preprocessing Function
# ==========================================================
def preprocess_all():
    """
    Full pipeline:
    - Load ESG + stock data
    - Align ESG to daily frequency
    - Merge and compute features
    - Export cleaned dataset
    """
    esg_df, stock_df = load_data()
    esg_daily = align_esg_to_daily(esg_df, stock_df)
    merged = merge_esg_stock(esg_daily, stock_df)
    enriched = add_features(merged)
    export_clean_data(enriched)
    return enriched


# ==========================================================
# 7️⃣ Run Standalone
# ==========================================================
if __name__ == "__main__":
    print("🚀 Running ESG–Stock preprocessing pipeline...")
    df = preprocess_all()
    print("🎯 Final dataset ready for analysis.")
    print(df.head(5))
