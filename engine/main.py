import json
import os
import numpy as np
import pandas as pd
import yfinance as yf

os.makedirs("data", exist_ok=True)

# Comprehensive global asset universe spanning all major asset classes
universe = {
    # Indonesian Equities
    "BBCA.JK": {
        "name": "Bank Central Asia",
        "category": "Indonesian Equities",
    },
    "BBRI.JK": {
        "name": "Bank Rakyat Indonesia",
        "category": "Indonesian Equities",
    },
    "TLKM.JK": {"name": "Telkom Indonesia", "category": "Indonesian Equities"},
    "ASII.JK": {"name": "Astra International", "category": "Indonesian Equities"},
    # Global / US Equities & ETFs
    "SPY": {"name": "S&P 500 ETF", "category": "Global Equities"},
    "QQQ": {"name": "Nasdaq 100 ETF", "category": "Global Equities"},
    "VT": {"name": "Vanguard Total World Stock", "category": "Global Equities"},
    # Fixed Income & Bonds
    "BND": {"name": "Total Bond Market ETF", "category": "Fixed Income"},
    "TLT": {"name": "20+ Year Treasury Bond ETF", "category": "Fixed Income"},
    # Real Estate (REITs)
    "VNQ": {"name": "Vanguard Real Estate ETF", "category": "Real Estate"},
    # Commodities
    "GLD": {"name": "Gold Trust", "category": "Commodities"},
    "SLV": {"name": "Silver Trust", "category": "Commodities"},
    # Crypto
    "BTC-USD": {"name": "Bitcoin", "category": "Crypto"},
    "ETH-USD": {"name": "Ethereum", "category": "Crypto"},
}

asset_data = []

for ticker, meta in universe.items():
  try:
    df = yf.download(ticker, period="3y", progress=False)
    if df.empty or "Close" not in df.columns:
      continue

    prices = df["Close"]
    if isinstance(prices, pd.DataFrame):
      prices = prices.iloc[:, 0]

    returns = prices.pct_change().dropna()
    if len(returns) < 50:
      continue

    ann_return = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = (
        float((ann_return - 0.04) / ann_vol) if ann_vol > 0 else 0.0
    )  # 4% risk-free rate assumption

    # Composite Score (0-100)
    score = round(max(0, min(100, 50 + (ann_return * 80) - (ann_vol * 25))), 1)

    # 4-Year Monte Carlo Simulation Projections (10,000 paths scaled to starting capital of 10,000,000)
    last_price = float(prices.iloc[-1])
    log_rets = np.log(1 + returns)
    mu, sigma = log_rets.mean(), log_rets.std()

    sim_results = []
    np.random.seed(42)
    for _ in range(1000):  # Fast simulation sample
      shocks = np.random.normal(mu, sigma, 1008)  # 4 years = 1008 trading days
      path_val = 10000000 * np.prod(np.exp(shocks))
      sim_results.append(path_val)

    p5 = float(np.percentile(sim_results, 5))
    p25 = float(np.percentile(sim_results, 25))
    median = float(np.percentile(sim_results, 50))
    p75 = float(np.percentile(sim_results, 75))
    p95 = float(np.percentile(sim_results, 95))

    asset_data.append({
        "ticker": ticker,
        "name": meta["name"],
        "category": meta["category"],
        "score": score,
        "annual_return": round(ann_return * 100, 2),
        "volatility": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "monte_carlo": {
            "bear_5th": round(p5, -3),
            "base_median": round(median, -3),
            "bull_95th": round(p95, -3),
        },
    })
  except Exception as e:
    print(f"Error for {ticker}: {e}")

# Sort by score descending
asset_data = sorted(asset_data, key=lambda x: x["score"], reverse=True)

output = {
    "top_opportunities": asset_data[:4],
    "full_universe": asset_data,
}

with open("data/assets.json", "w") as f:
  json.dump(output, f, indent=4)

print("Global multi-asset engine executed successfully!")
