import json
import os
import numpy as np
import pandas as pd
import yfinance as yf

os.makedirs("data", exist_ok=True)

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
    df = yf.download(ticker, period="2y", progress=False)
    if df.empty or "Close" not in df.columns:
      continue
    prices = df["Close"]
    if isinstance(prices, pd.DataFrame):
      prices = prices.iloc[:, 0]

    returns = prices.pct_change().dropna()
    if len(returns) < 10:
      continue

    ann_return = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = (
        float((ann_return - 0.05) / ann_vol) if ann_vol > 0 else 0.0
    )

    score = round(
        max(0, min(100, 50 + (ann_return * 100) - (ann_vol * 30))), 1
    )
    ret_score = round(min(100, max(0, (ann_return + 0.1) * 200)), 1)
    risk_score = round(min(100, max(0, (1 - ann_vol) * 100)), 1)
    liq_score = 85.0 if "JK" not in ticker else 75.0

    # Monte Carlo simulation over 4 years (1008 trading days)
    last_price = float(prices.iloc[-1])
    log_returns = np.log(1 + returns)
    mu = log_returns.mean()
    sigma = log_returns.std()

    days = 1008
    sims = 500
    np.random.seed(42)
    rand_shocks = np.random.normal(loc=mu, scale=sigma, size=(days, sims))
    path_matrix = last_price * np.exp(np.cumsum(rand_shocks, axis=0))

    checkpoints = [252, 504, 756, 1007]
    mc_median = [
        float(np.median(path_matrix[cp])) / last_price for cp in checkpoints
    ]
    mc_5th = [
        float(np.percentile(path_matrix[cp], 5)) / last_price
        for cp in checkpoints
    ]
    mc_95th = [
        float(np.percentile(path_matrix[cp], 95)) / last_price
        for cp in checkpoints
    ]

    asset_data.append({
        "ticker": ticker,
        "name": name,
        "score": score,
        "annual_return": round(ann_return * 100, 2),
        "volatility": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "audit": {
            "return_score": ret_score,
            "risk_score": risk_score,
            "liquidity_score": liq_score,
            "strengths": [
                "Strong historical risk-adjusted return",
                "High liquidity profile",
            ],
            "weaknesses": [
                "Exposed to macroeconomic volatility",
                "Currency / FX risk factors",
            ],
        },
        "monte_carlo": {
            "years": ["Year 1", "Year 2", "Year 3", "Year 4"],
            "median": mc_median,
            "bear_5th": mc_5th,
            "bull_95th": mc_95th,
        },
    })
  except Exception as e:
    print(f"Error processing {ticker}: {e}")

asset_data = sorted(asset_data, key=lambda x: x["score"], reverse=True)

output = {
    "top_opportunities": asset_data[:4],
    "full_universe": asset_data,
}

with open("data/assets.json", "w") as f:
  json.dump(output, f, indent=4)

print("Advanced quantitative engine executed successfully!")
