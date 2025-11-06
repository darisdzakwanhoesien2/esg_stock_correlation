# modules/visualization.py
"""
Visualization Module
--------------------
Reusable interactive Plotly/Altair charts for ESG–Stock Correlation App.
Includes ESG distribution plots, sector radar charts, ESG–return trends, and
combined overlays for Streamlit dashboards.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt


# ==========================================================
# 1️⃣ ESG Distribution by Sector
# ==========================================================
def plot_esg_distribution(df, score_col="Overall_ESG_Score"):
    """
    Visualize ESG score distribution across sectors.
    """
    fig = px.box(
        df,
        x="Sector",
        y=score_col,
        color="Sector",
        points="all",
        title=f"Distribution of {score_col} by Sector",
    )
    fig.update_layout(height=500, xaxis_tickangle=-30)
    return fig


# ==========================================================
# 2️⃣ ESG vs Stock Return Scatter Plot
# ==========================================================
def plot_esg_vs_return(df, x="Overall_ESG_Score", y="Daily_Return", sector=None):
    """
    Plots relationship between ESG and financial performance (daily returns).
    Optionally filter by sector.
    """
    if sector:
        df = df[df["Sector"] == sector]
        title = f"{x} vs {y} ({sector})"
    else:
        title = f"{x} vs {y} (All Sectors)"

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="Sector",
        hover_data=["Company", "Ticker", "Country"],
        trendline="ols",
        opacity=0.6,
        title=title,
    )
    fig.update_layout(height=500)
    return fig


# ==========================================================
# 3️⃣ ESG Evolution Over Time (Line Chart)
# ==========================================================
def plot_esg_time_trend(df, ticker="AAPL"):
    """
    Show evolution of ESG score and stock price over time for one ticker.
    """
    df_ticker = df[df["Ticker"] == ticker].sort_values("Date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ticker["Date"],
        y=df_ticker["Overall_ESG_Score"],
        name="Overall ESG",
        mode="lines",
        line=dict(width=3, color="green")
    ))
    fig.add_trace(go.Scatter(
        x=df_ticker["Date"],
        y=df_ticker["Close"] / df_ticker["Close"].iloc[0] * 100,  # normalize to %
        name="Stock Price (Indexed)",
        mode="lines",
        line=dict(width=2, color="blue", dash="dash")
    ))

    fig.update_layout(
        title=f"ESG and Stock Price Evolution – {ticker}",
        xaxis_title="Date",
        yaxis_title="Normalized Score / Price Index (100 = start)",
        height=450,
        legend_title="Metric"
    )
    return fig


# ==========================================================
# 4️⃣ Sector Radar Chart (E, S, G Balance)
# ==========================================================
def plot_sector_radar(df, sector="Technology"):
    """
    Radar chart showing mean E/S/G scores for one sector.
    """
    df_sector = df[df["Sector"] == sector]
    mean_scores = (
        df_sector[["Environmental_Score", "Social_Score", "Governance_Score"]].mean()
    )

    categories = ["Environmental_Score", "Social_Score", "Governance_Score"]
    values = mean_scores.values.tolist() + [mean_scores.values[0]]  # loop back to start

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=["E", "S", "G", "E"],
        fill="toself",
        line_color="green",
        opacity=0.7,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=f"Average ESG Profile – {sector}",
        height=400
    )
    return fig


# ==========================================================
# 5️⃣ Altair Interactive ESG–Volatility Chart
# ==========================================================
def plot_esg_vs_volatility_altair(df):
    """
    Altair scatterplot showing ESG vs Volatility_30D with interactive tooltip.
    """
    chart = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("Overall_ESG_Score", title="Overall ESG Score"),
            y=alt.Y("Volatility_30D", title="30-Day Volatility"),
            color="Sector",
            tooltip=["Company", "Ticker", "Sector", "Overall_ESG_Score", "Volatility_30D"],
        )
        .properties(title="ESG vs Volatility (Altair Interactive View)", height=400)
        .interactive()
    )
    return chart


# ==========================================================
# 6️⃣ ESG Trend vs Returns (Dual Axis)
# ==========================================================
def plot_esg_trend_vs_return(df, ticker="AAPL"):
    """
    Shows ESG_Trend and Daily_Return together (dual axis view).
    """
    df_ticker = df[df["Ticker"] == ticker].sort_values("Date")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_ticker["Date"],
        y=df_ticker["Daily_Return"],
        name="Daily Return",
        marker_color="blue",
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=df_ticker["Date"],
        y=df_ticker["ESG_Trend"],
        name="ESG Trend",
        yaxis="y2",
        line=dict(color="green", width=2)
    ))

    fig.update_layout(
        title=f"ESG Trend vs Daily Return – {ticker}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Daily Return"),
        yaxis2=dict(
            title="ESG Trend",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        height=450
    )
    return fig
