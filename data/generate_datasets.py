# data/generate_datasets.py
"""
Synthetic ESG & Stock Dataset Generator (Config-Driven)
-------------------------------------------------------
Generates esg_scores.csv and stock_prices.csv using a parameterized JSON config.
You can later externalize the config to a separate JSON file for easy fine-tuning.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ==========================================================
# 1️⃣ CONFIGURATION (embedded JSON — editable anytime)
# ==========================================================
CONFIG = {
    "Environmental": {
        "CarbonIntensity": {"mean_offset": 120, "std": 30, "min": 10, "max": 300},
        "RenewableEnergy": {"scale_divisor": 1.2, "std": 15}
    },
    "Social": {
        "GenderDiversity": {"base": 35, "sensitivity": 0.5, "std": 5, "min": 10, "max": 60}
    },
    "Governance": {
        "BoardIndependence": {"base": 70, "sensitivity": 0.333, "std": 5, "min": 30, "max": 95}
    },
    "Controversy": {
        "base": 90, "sensitivity": 0.333, "std": 5, "min": 50, "max": 95
    },
    "ESG_Weights": {"E": 0.4, "S": 0.3, "G": 0.3}
}

# ==========================================================
# 2️⃣ BASIC SETTINGS
# ==========================================================
OUTPUT_DIR = os.path.dirname(__file__)

COMPANIES = [
    {"Ticker": "AAPL", "Company": "Apple Inc.", "Sector": "Technology", "Country": "USA"},
    {"Ticker": "TSLA", "Company": "Tesla Inc.", "Sector": "Automotive", "Country": "USA"},
    {"Ticker": "BP", "Company": "BP Plc", "Sector": "Energy", "Country": "UK"},
    {"Ticker": "MSFT", "Company": "Microsoft Corp.", "Sector": "Technology", "Country": "USA"},
    {"Ticker": "HSBC", "Company": "HSBC Holdings", "Sector": "Financials", "Country": "UK"},
    {"Ticker": "NOKIA", "Company": "Nokia Oyj", "Sector": "Telecom", "Country": "Finland"},
]

YEARS = [2020, 2021, 2022, 2023, 2024]


# ==========================================================
# 3️⃣ ESG DATA GENERATOR
# ==========================================================
def generate_esg_scores(config: dict = CONFIG):
    np.random.seed(42)
    records = []

    for c in COMPANIES:
        base_env = np.random.uniform(55, 85)
        base_soc = np.random.uniform(60, 80)
        base_gov = np.random.uniform(65, 90)

        for year in YEARS:
            # ---------------- Environmental ----------------
            e = np.clip(base_env + np.random.normal(0, 4), 0, 100)
            carbon_cfg = config["Environmental"]["CarbonIntensity"]
            renewable_cfg = config["Environmental"]["RenewableEnergy"]

            carbon_intensity = np.clip(
                np.random.normal(carbon_cfg["mean_offset"] - e, carbon_cfg["std"]),
                carbon_cfg["min"], carbon_cfg["max"]
            )
            renewable_ratio = np.clip(
                np.random.normal(e / renewable_cfg["scale_divisor"], renewable_cfg["std"]),
                0, 100
            )

            # ---------------- Social ----------------
            s = np.clip(base_soc + np.random.normal(0, 4), 0, 100)
            gender_cfg = config["Social"]["GenderDiversity"]

            gender_diversity = np.clip(
                np.random.normal(
                    gender_cfg["base"] + (s - 65) * gender_cfg["sensitivity"],
                    gender_cfg["std"]
                ),
                gender_cfg["min"], gender_cfg["max"]
            )

            # ---------------- Governance ----------------
            g = np.clip(base_gov + np.random.normal(0, 3), 0, 100)
            board_cfg = config["Governance"]["BoardIndependence"]

            board_independence = np.clip(
                np.random.normal(
                    board_cfg["base"] + (g - 70) * board_cfg["sensitivity"],
                    board_cfg["std"]
                ),
                board_cfg["min"], board_cfg["max"]
            )

            # ---------------- Controversy ----------------
            contr_cfg = config["Controversy"]
            overall = round(
                config["ESG_Weights"]["E"] * e +
                config["ESG_Weights"]["S"] * s +
                config["ESG_Weights"]["G"] * g, 2
            )
            controversy = np.clip(
                np.random.normal(
                    contr_cfg["base"] - (100 - overall) * contr_cfg["sensitivity"],
                    contr_cfg["std"]
                ),
                contr_cfg["min"], contr_cfg["max"]
            )

            # ---------------- Record ----------------
            records.append({
                "Date": f"{year}-12-31",
                "Ticker": c["Ticker"],
                "Company": c["Company"],
                "Sector": c["Sector"],
                "Country": c["Country"],
                "Environmental_Score": round(e, 2),
                "Social_Score": round(s, 2),
                "Governance_Score": round(g, 2),
                "Overall_ESG_Score": overall,
                "Carbon_Intensity": round(carbon_intensity, 1),
                "Renewable_Energy_Ratio": round(renewable_ratio, 1),
                "Gender_Diversity": round(gender_diversity, 1),
                "Board_Independence": round(board_independence, 1),
                "ESG_Controversy_Score": round(controversy, 1),
                "ESG_Data_Source": "Synthetic ESG Generator (Config-driven)"
            })

    df = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_DIR, "esg_scores.csv")
    df.to_csv(out_path, index=False)

    # Save config snapshot for traceability
    with open(os.path.join(OUTPUT_DIR, "esg_config_snapshot.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"✅ ESG dataset saved to: {out_path}")
    print(f"⚙️ Config snapshot saved to: data/esg_config_snapshot.json")
    return df


# ==========================================================
# 4️⃣ STOCK DATA GENERATOR (simplified placeholder)
# ==========================================================
def generate_stock_prices():
    np.random.seed(123)
    records = []
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)

    for c in COMPANIES:
        price = np.random.uniform(50, 200)
        dates = pd.date_range(start_date, end_date, freq="B")
        daily_returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [price]

        for r in daily_returns:
            prices.append(prices[-1] * (1 + r))
        prices = prices[1:]

        for i, date in enumerate(dates):
            open_p = prices[i] * np.random.uniform(0.99, 1.01)
            close_p = prices[i]
            high_p = max(open_p, close_p) * np.random.uniform(1.00, 1.02)
            low_p = min(open_p, close_p) * np.random.uniform(0.98, 1.00)
            volume = np.random.randint(1_000_000, 5_000_000)
            market_cap = close_p * 1e7

            records.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Ticker": c["Ticker"],
                "Open": round(open_p, 2),
                "High": round(high_p, 2),
                "Low": round(low_p, 2),
                "Close": round(close_p, 2),
                "Adj_Close": round(close_p, 2),
                "Volume": volume,
                "Market_Cap": round(market_cap, 0),
                "Daily_Return": round(daily_returns[i], 5),
                "Sector": c["Sector"],
                "Country": c["Country"],
                "Source": "Synthetic Stock Generator v1.0",
            })

    df = pd.DataFrame(records)
    df["Volatility_7D"] = df.groupby("Ticker")["Daily_Return"].rolling(7).std().reset_index(0, drop=True)
    df["Volatility_30D"] = df.groupby("Ticker")["Daily_Return"].rolling(30).std().reset_index(0, drop=True)

    out_path = os.path.join(OUTPUT_DIR, "stock_prices.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Stock dataset saved to: {out_path}")
    return df


# ==========================================================
# 5️⃣ MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("🚀 Generating synthetic ESG & stock datasets using config-driven parameters...")
    esg_df = generate_esg_scores(CONFIG)
    stock_df = generate_stock_prices()
    print("🎯 Done! Datasets ready in /data folder.")
