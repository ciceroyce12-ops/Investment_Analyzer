import json
import os
import numpy as np
import pandas as pd
import yfinance as yf

os.makedirs("data", exist_ok=True)

# Define a multi-asset global universe (Indonesia + US + Commodities)
tickers = {
    "BBCA.JK": "Bank Central Asia (Indonesia)",
    "BBRI.JK": "Bank Rakyat Indonesia (Indonesia)",
    "TLKM.JK": "Telkom Indonesia (Indonesia)",
    "SPY": "S&P 500 ETF (US)",
    "QQQ": "Nasdaq 100 ETF (US)",
    "GLD": "Gold Trust (Global Commodity)",
    "BTC-USD": "Bitcoin (Crypto)",
}

asset_data = []

for ticker, name in tickers.items():
  try:
    df = yf.download(ticker, period="2y", progress=False)["Close"]
    if isinstance(df, pd.DataFrame):
      df = df.iloc[:, 0]  # Handle multi-index columns if any

    returns = df.pct_change().dropna()
    ann_return = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))

    # Calculate Sharpe Ratio (assuming risk-free rate ~ 5%)
    sharpe = (
        float((ann_return - 0.05) / ann_vol) if ann_vol > 0 else 0.0
    )

    # Simple composite quantitative score (0 to 100)
    score = round(max(0, min(100, 50 + (ann_return * 100) - (ann_vol * 30))), 1)

    asset_data.append({
        "ticker": ticker,
        "name": name,
        "score": score,
        "annual_return": round(ann_return * 100, 2),
        "volatility": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
    })
  except Exception as e:
    print(f"Error processing {ticker}: {e}")

# Sort assets by score descending (Top Opportunities)
asset_data = sorted(asset_data, key=lambda x: x["score"], reverse=True)

output = {
    "top_opportunities": asset_data[:4],  # The Top 4
    "full_universe": asset_data,
}

with open("data/assets.json", "w") as f:
  json.dump(output, f, indent=4)

print("Quantitative multi-asset engine executed successfully!")
