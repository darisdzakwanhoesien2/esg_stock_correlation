# modules/correlation.py
"""
Correlation Analysis Module
----------------------------
Analyzes the relationships between ESG factors and financial metrics.
Computes overall, sector-level, and rolling correlations, and prepares
Plotly visualizations for the Streamlit app.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ==========================================================
# 1️⃣ Load merged dataset
# ==========================================================
def load_merged_data(filename="merged_esg_stock.csv"):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"📈 Loaded merged dataset with {len(df)} rows and {df['Ticker'].nunique()} tickers")
    return df


# ==========================================================
# 2️⃣ Static Correlation Matrix
# ==========================================================
def compute_static_correlation(df, method="pearson"):
    """
    Computes the static correlation between ESG and financial metrics.
    """
    numeric_cols = [
        "Environmental_Score", "Social_Score", "Governance_Score",
        "Overall_ESG_Score", "Daily_Return", "Volatility_30D",
        "RiskAdjustedReturn"
    ]
    corr_matrix = df[numeric_cols].corr(method=method)
    print(f"✅ Computed {method.title()} correlation matrix")
    return corr_matrix


# ==========================================================
# 3️⃣ Rolling (Time-Based) Correlation
# ==========================================================
def compute_rolling_correlation(df, window=90, ticker="AAPL"):
    """
    Computes rolling correlation between ESG and Daily_Return for one ticker.
    Returns a time series of correlation values.
    """
    df_ticker = df[df["Ticker"] == ticker].sort_values("Date")
    result = (
        df_ticker["Overall_ESG_Score"]
        .rolling(window)
        .corr(df_ticker["Daily_Return"])
    )
    corr_df = pd.DataFrame({"Date": df_ticker["Date"], "Rolling_Corr": result})
    print(f"📊 Computed {window}-day rolling correlation for {ticker}")
    return corr_df


# ==========================================================
# 4️⃣ Sector-Level Correlation Summary
# ==========================================================
def compute_sector_correlations(df):
    """
    Aggregates correlations per sector between ESG and stock metrics.
    """
    results = []
    for sector, group in df.groupby("Sector"):
        corr = group[[
            "Overall_ESG_Score", "Daily_Return", "Volatility_30D", "RiskAdjustedReturn"
        ]].corr().loc["Overall_ESG_Score", ["Daily_Return", "Volatility_30D", "RiskAdjustedReturn"]]
        results.append({
            "Sector": sector,
            "Corr_ESG_Return": corr["Daily_Return"],
            "Corr_ESG_Volatility": corr["Volatility_30D"],
            "Corr_ESG_RiskAdjReturn": corr["RiskAdjustedReturn"]
        })
    summary = pd.DataFrame(results)
    print("🏭 Sector-level correlation summary computed")
    return summary


# ==========================================================
# 5️⃣ Visualization: Heatmap
# ==========================================================
def plot_correlation_heatmap(corr_matrix, title="ESG–Financial Correlation Matrix"):
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=title
    )
    fig.update_layout(height=500, margin=dict(l=40, r=40, t=50, b=40))
    return fig


# ==========================================================
# 6️⃣ Visualization: Rolling Correlation Line
# ==========================================================
def plot_rolling_correlation(corr_df, ticker="AAPL", window=90):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=corr_df["Date"],
        y=corr_df["Rolling_Corr"],
        mode="lines",
        line=dict(width=2),
        name=f"{ticker} {window}-Day Rolling Corr"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Rolling ESG–Return Correlation ({ticker}, {window}-day window)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        height=400
    )
    return fig


# ==========================================================
# 7️⃣ Visualization: Sector Correlation Bar Chart
# ==========================================================
def plot_sector_correlations(summary_df):
    melted = summary_df.melt(id_vars="Sector", var_name="Metric", value_name="Correlation")
    fig = px.bar(
        melted,
        x="Sector",
        y="Correlation",
        color="Metric",
        barmode="group",
        title="Sector-level ESG–Financial Correlations"
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    return fig


# ==========================================================
# 8️⃣ Quick Test Runner
# ==========================================================
if __name__ == "__main__":
    print("🚀 Running ESG correlation analysis module...")
    df = load_merged_data()

    pearson_corr = compute_static_correlation(df, method="pearson")
    spearman_corr = compute_static_correlation(df, method="spearman")

    rolling_corr = compute_rolling_correlation(df, ticker="AAPL", window=90)
    sector_summary = compute_sector_correlations(df)

    # Optional: visualize (uncomment if running interactively)
    # fig1 = plot_correlation_heatmap(pearson_corr)
    # fig2 = plot_rolling_correlation(rolling_corr, ticker="AAPL")
    # fig3 = plot_sector_correlations(sector_summary)
    # fig1.show(); fig2.show(); fig3.show()

    print("🎯 Correlation analysis complete.")
