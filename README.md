esg_stock_app/
│
├── app.py                     # 🌐 Streamlit dashboard (sentiment × ESG × stock)
│
├── data/
│   ├── esg_scores.csv                      # simulated ESG dataset
│   ├── stock_prices.csv                    # stock pricing dataset
│   ├── merged_esg_stock.csv                # merged ESG + stock data
│   ├── news_sentiment_raw.csv              # ✅ full textual dataset (you asked for)
│   ├── news_sentiment_raw_scored.csv       # sentiment outputs (VADER/FinBERT/RoBERTa/DeBERTa)
│   ├── news_sentiment_weighted.csv         # daily aggregated weighted sentiment
│   └── sentiment_run_snapshot.json         # metadata of latest model run
│
├── modules/
│   ├── data_loader.py                      # handles CSV loading and validation
│   ├── preprocessor.py                     # merges ESG + stock datasets
│   ├── sentiment_multimode_sota.py         # sentiment scoring (4 modes × 3 text levels)
│   ├── sentiment_aggregator.py             # weighted daily aggregation
│   ├── visualization.py                    # optional — reusable charts
│   └── correlation.py                      # correlation analysis helpers
│
├── requirements.txt
└── README.md



# Step 1 — Analyze sentiment using all models
python -m modules.sentiment_multimode_sota

# Step 2 — Aggregate daily weighted sentiment
python -m modules.sentiment_aggregator

# Step 3 — Launch Streamlit dashboard
streamlit run app.py

# esg_stock_correlation
