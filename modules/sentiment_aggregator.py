# modules/sentiment_aggregator.py
"""
Weighted Daily Sentiment Aggregator
-----------------------------------
Aggregates per-article multi-model sentiment data into
daily ticker-level averages weighted by media_reliability.

Input:
    data/news_sentiment_raw_scored.csv
Output:
    data/news_sentiment_weighted.csv
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ==========================================================
# Helper: Weighted average with NaN-safe handling
# ==========================================================
def weighted_avg(values, weights):
    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)
    mask = ~np.isnan(values)
    if mask.sum() == 0 or weights[mask].sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


# ==========================================================
# Core Aggregation Function
# ==========================================================
def aggregate_weighted_sentiment(input_file="news_sentiment_raw_scored.csv",
                                 output_file="news_sentiment_weighted.csv"):
    """
    Aggregate daily weighted sentiment across all models and text fields.
    """
    input_path = os.path.join(DATA_DIR, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Missing {input_file}. Please run sentiment analysis first.")

    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["media_reliability"] = df["media_reliability"].clip(0, 1).fillna(0.7)

    models = ["vader", "finbert", "roberta", "deberta"]
    text_fields = ["headline", "snippet", "full_text"]

    results = []
    grouped = df.groupby(["Date", "Ticker"])

    print(f"🧮 Aggregating {len(df)} articles across {len(grouped)} (Date, Ticker) groups...")

    for (date, ticker), group in grouped:
        weights = group["media_reliability"].values
        record = {
            "Date": date.strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "n_articles": len(group),
            "avg_reliability": float(np.nanmean(weights))
        }

        # Weighted mean sentiment for each text type per model
        for field in text_fields:
            for model in models:
                col = f"{field}_sent_{model}"
                if col in group.columns:
                    record[f"weighted_{field}_{model}"] = weighted_avg(group[col].values, weights)
                else:
                    record[f"weighted_{field}_{model}"] = np.nan

        # Cross-model mean for each text type
        for field in text_fields:
            cols = [f"weighted_{field}_{m}" for m in models]
            vals = [record[c] for c in cols if not pd.isna(record[c])]
            record[f"{field}_mean_all_models"] = np.nanmean(vals) if len(vals) > 0 else np.nan

        # Grand average sentiment (across all text fields and models)
        all_vals = [record[f"{field}_mean_all_models"] for field in text_fields if not pd.isna(record[f"{field}_mean_all_models"])]
        record["cross_model_mean_sentiment"] = np.nanmean(all_vals) if len(all_vals) > 0 else np.nan

        results.append(record)

    out_df = pd.DataFrame(results)
    out_path = os.path.join(DATA_DIR, output_file)
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved weighted daily sentiment → {out_path}")
    print(f"Records: {len(out_df)}")
    print(out_df.head())
    return out_df


# ==========================================================
# Run standalone
# ==========================================================
if __name__ == "__main__":
    start = datetime.now()
    print("🚀 Starting Weighted Daily Aggregation ...")
    aggregate_weighted_sentiment()
    print(f"✅ Completed in {(datetime.now() - start).seconds} seconds.")
