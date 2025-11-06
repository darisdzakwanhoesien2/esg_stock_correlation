# modules/sentiment_multimode_sota.py
"""
============================================================
Sentiment Analysis Module (Multi-Text, Multi-Mode, SOTA)
============================================================

Features:
- Analyzes headline, snippet, and full_text
- Supports multiple sentiment engines:
    1. VADER (Lexicon-based)
    2. FinBERT (Finance domain)
    3. RoBERTa (General short text)
    4. DeBERTa-v3 (SOTA general)
- Produces numeric scores [-1, 1] and categorical labels
- Saves results to: data/news_sentiment_raw_scored.csv
- Records metadata snapshot (run time, models, fields)
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import pipeline


# ==========================================================
# Configuration
# ==========================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Map of model identifiers
MODEL_MAP = {
    "finbert": "yiyanghkust/finbert-tone",
    "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "deberta": "microsoft/deberta-v3-base",
}


# ==========================================================
# Helper: VADER
# ==========================================================
def get_vader():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()


def score_vader(text):
    """
    Compute VADER sentiment score for one text.
    Returns float in [-1,1].
    """
    if not text or not isinstance(text, str):
        return np.nan, "neutral"
    s = get_vader().polarity_scores(text)["compound"]
    label = "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
    return s, label


# ==========================================================
# Helper: Transformer Models
# ==========================================================
def load_transformer(model_name):
    """
    Lazy-load a transformer model for sentiment analysis.
    """
    # device=-1 for CPU, device=0 for GPU
    # use_fast=False to avoid issues with some tokenizers
    return pipeline("sentiment-analysis", model=model_name, device=-1, use_fast=False)


def score_transformer(pipe, text):
    """
    Compute transformer sentiment score.
    Normalize to [-1,1] numeric and return label.
    """
    if not text or not isinstance(text, str):
        return np.nan, "neutral"
    try:
        res = pipe(text[:512])[0]
        label = res["label"].lower()
        score = float(res["score"])
        if "pos" in label:
            val = score
        elif "neg" in label:
            val = -score
        else:
            val = 0.0
        return val, label
    except Exception:
        return np.nan, "neutral"


# ==========================================================
# Core Sentiment Function
# ==========================================================
def compute_sentiment(news_df: pd.DataFrame, modes=None):
    """
    Compute sentiment for each text type (headline, snippet, full_text)
    across multiple modes (VADER, FinBERT, RoBERTa, DeBERTa).
    """
    if modes is None:
        modes = ["vader", "finbert", "roberta", "deberta"]

    df = news_df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # Lazy-load models
    pipes = {}
    for m in modes:
        if m != "vader":
            pipes[m] = load_transformer(MODEL_MAP[m])

    # Sentiment computation loop
    text_fields = ["headline", "snippet", "full_text"]
    for mode in modes:
        print(f"\n🧠 Processing mode: {mode.upper()}")
        for field in text_fields:
            col_score = f"{field}_sent_{mode}"
            col_label = f"{field}_label_{mode}"
            scores, labels = [], []

            for text in tqdm(df[field].fillna("").astype(str), desc=f"{mode}-{field}"):
                if mode == "vader":
                    s, l = score_vader(text)
                else:
                    s, l = score_transformer(pipes[mode], text)
                scores.append(s)
                labels.append(l)

            df[col_score] = scores
            df[col_label] = labels

    print("\n✅ Sentiment computation completed for all modes and fields.")
    return df


# ==========================================================
# Save Outputs and Metadata
# ==========================================================
def save_results(df, filename="news_sentiment_raw_scored.csv"):
    out_path = os.path.join(DATA_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"💾 Saved sentiment results → {out_path}")
    return out_path


def save_metadata(modes):
    metadata = {
        "run_time": datetime.utcnow().isoformat() + "Z",
        "modes": modes,
        "models": {m: MODEL_MAP.get(m, "vader_builtin") for m in modes},
        "fields": ["headline", "snippet", "full_text"],
        "output_file": "news_sentiment_raw_scored.csv"
    }
    path = os.path.join(DATA_DIR, "sentiment_run_snapshot.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Saved run metadata snapshot → {path}")


# ==========================================================
# Main Execution
# ==========================================================
if __name__ == "__main__":
    input_path = os.path.join(DATA_DIR, "news_sentiment_raw.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError("❌ news_sentiment_raw.csv not found. Please create it first.")

    print("📥 Reading input:", input_path)
    news_df = pd.read_csv(input_path)

    # Run analysis
    modes = ["vader", "finbert", "roberta", "deberta"]
    scored_df = compute_sentiment(news_df, modes=modes)

    # Save outputs
    save_results(scored_df)
    save_metadata(modes)

    print("\n🔹 Preview of sentiment output:")
    print(scored_df.head(2))