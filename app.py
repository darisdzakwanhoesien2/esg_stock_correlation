# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# Import internal pipeline modules
from modules.sentiment_multimode_sota import compute_sentiment, save_results, save_metadata
from modules.sentiment_aggregator import aggregate_weighted_sentiment

# -----------------------------
# Config and setup
# -----------------------------
st.set_page_config(layout="wide", page_title="ESG × Stock × Sentiment Dashboard")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_NEWS = os.path.join(DATA_DIR, "news_sentiment_raw.csv")
SCORED_NEWS = os.path.join(DATA_DIR, "news_sentiment_raw_scored.csv")
WEIGHTED_NEWS = os.path.join(DATA_DIR, "news_sentiment_weighted.csv")
MERGED_ESG = os.path.join(DATA_DIR, "merged_esg_stock.csv")

# -----------------------------
# Title
# -----------------------------
st.title("🌍 ESG × Stock × Sentiment — Automated Workflow Dashboard")
st.markdown("Run full analysis pipeline — from sentiment scoring to visualization — directly inside Streamlit.")

# -----------------------------
# Sidebar Workflow Controller
# -----------------------------
st.sidebar.header("⚙️ Workflow Controls")
run_sentiment = st.sidebar.button("1️⃣ Run Sentiment Analysis")
run_aggregate = st.sidebar.button("2️⃣ Run Weighted Aggregation")
refresh_dashboard = st.sidebar.button("🔄 Refresh Dashboard Data")

modes_selected = st.sidebar.multiselect(
    "Select Sentiment Models",
    options=["vader", "finbert", "roberta", "deberta"],
    default=["vader", "finbert", "roberta", "deberta"],
)

# -----------------------------
# Step 1 — Run Sentiment Analysis
# -----------------------------
if run_sentiment:
    st.subheader("🧠 Step 1 — Running Sentiment Analysis")
    if not os.path.exists(RAW_NEWS):
        st.error("Missing `news_sentiment_raw.csv`. Please create it first in /data.")
        st.stop()

    import pandas as pd
    df_news = pd.read_csv(RAW_NEWS, sep=',', quotechar='"', on_bad_lines='warn')
    progress = st.progress(0)
    st.info(f"Loaded {len(df_news)} articles. Beginning multi-model sentiment scoring...")

    start_time = time.time()
    with st.spinner("Running multi-model sentiment analysis..."):
        scored_df = compute_sentiment(df_news, modes=modes_selected)
        save_results(scored_df)
        save_metadata(modes_selected)
        progress.progress(100)

    elapsed = time.time() - start_time
    st.success(f"✅ Sentiment analysis completed in {elapsed:.1f} seconds!")
    st.session_state["sentiment_done"] = True

# -----------------------------
# Step 2 — Weighted Aggregation
# -----------------------------
if run_aggregate:
    st.subheader("⚖️ Step 2 — Aggregating Weighted Sentiment")
    if not os.path.exists(SCORED_NEWS):
        st.error("Missing `news_sentiment_raw_scored.csv`. Run Step 1 first.")
        st.stop()

    with st.spinner("Computing reliability-weighted daily averages..."):
        aggregated_df = aggregate_weighted_sentiment(
            input_file="news_sentiment_raw_scored.csv",
            output_file="news_sentiment_weighted.csv",
        )
        st.success(f"✅ Weighted aggregation complete — {len(aggregated_df)} daily records created.")
        st.session_state["aggregated_done"] = True

# -----------------------------
# Step 3 — Dashboard Visualization
# -----------------------------
if refresh_dashboard or ("aggregated_done" in st.session_state):
    st.subheader("📊 Step 3 — Sentiment & ESG Correlation Dashboard")

    try:
        merged_path = MERGED_ESG
        weighted_path = WEIGHTED_NEWS

        if not os.path.exists(weighted_path):
            st.error("❌ Missing weighted sentiment file. Please run Step 2 first.")
            st.stop()

        df_sent = pd.read_csv(weighted_path, parse_dates=["Date"])
        df_sent["Date"] = pd.to_datetime(df_sent["Date"])
        st.info(f"Loaded {len(df_sent)} sentiment records across {df_sent['Ticker'].nunique()} companies.")

        if os.path.exists(merged_path):
            df_esg = pd.read_csv(merged_path, parse_dates=["Date"])
            df = pd.merge(df_esg, df_sent, on=["Date", "Ticker"], how="left")
            st.success("Merged ESG + Stock + Sentiment data successfully.")
        else:
            df = df_sent.copy()
            st.warning("No merged ESG file found — displaying sentiment-only view.")

        # --- Visualization Section ---
        st.markdown("### 📈 Sentiment and ESG Over Time")

        tickers = sorted(df["Ticker"].dropna().unique())
        ticker_sel = st.selectbox("Select company", tickers)
        df_t = df[df["Ticker"] == ticker_sel].sort_values("Date")

        import plotly.graph_objects as go
        fig = go.Figure()

        if "Close" in df_t.columns:
            df_t["Close_norm"] = df_t["Close"] / df_t["Close"].iloc[0] * 100
            fig.add_trace(go.Scatter(x=df_t["Date"], y=df_t["Close_norm"],
                                     name="Stock Price (Indexed)", line=dict(color="blue")))

        fig.add_trace(go.Scatter(x=df_t["Date"], y=df_t["cross_model_mean_sentiment"],
                                 name="Cross-model Sentiment", yaxis="y2", line=dict(color="green")))
        fig.update_layout(
            title=f"{ticker_sel} — Stock vs Sentiment",
            yaxis2=dict(overlaying="y", side="right", title="Sentiment (-1..1)", range=[-1, 1]),
            xaxis_title="Date"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        st.markdown("### 🔗 Correlation Matrix (Sentiment × ESG × Stock)")
        num_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]]
        corr = df[num_cols].corr()
        import plotly.express as px
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Dashboard Error: {e}")

# -----------------------------
# Notes
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("✅ **Usage Flow:**")
st.sidebar.markdown("1. Upload or verify `news_sentiment_raw.csv` in /data")
st.sidebar.markdown("2. Click **Run Sentiment Analysis**")
st.sidebar.markdown("3. Click **Run Weighted Aggregation**")
st.sidebar.markdown("4. Click **Refresh Dashboard** to visualize results")