import json
import os
import numpy as np
import pandas as pd
import yfinance as yf

os.makedirs("data", exist_ok=True)

# Comprehensive global asset universe with currency & fee profiles
universe = {
    "BBCA.JK": {
        "name": "Bank Central Asia",
        "category": "Indonesian Equities",
        "currency": "IDR",
        "fee": 0.0015,
    },
    "BBRI.JK": {
        "name": "Bank Rakyat Indonesia",
        "category": "Indonesian Equities",
        "currency": "IDR",
        "fee": 0.0015,
    },
    "TLKM.JK": {
        "name": "Telkom Indonesia",
        "category": "Indonesian Equities",
        "currency": "IDR",
        "fee": 0.0015,
    },
    "SPY": {
        "name": "S&P 500 ETF",
        "category": "Global Equities",
        "currency": "USD",
        "fee": 0.0025,
    },
    "QQQ": {
        "name": "Nasdaq 100 ETF",
        "category": "Global Equities",
        "currency": "USD",
        "fee": 0.0025,
    },
    "VT": {
        "name": "Vanguard Total World",
        "category": "Global Equities",
        "currency": "USD",
        "fee": 0.0025,
    },
    "BND": {
        "name": "Total Bond Market ETF",
        "category": "Fixed Income",
        "currency": "USD",
        "fee": 0.0010,
    },
    "GLD": {
        "name": "Gold Trust",
        "category": "Commodities",
        "currency": "USD",
        "fee": 0.0020,
    },
    "BTC-USD": {
        "name": "Bitcoin",
        "category": "Crypto",
        "currency": "USD",
        "fee": 0.0050,
    },
}

# Fetch USD/IDR exchange rate history for unified FX normalization
try:
  fx_df = yf.download("IDR=X", period="3y", progress=False)["Close"]
  if isinstance(fx_df, pd.DataFrame):
    fx_df = fx_df.iloc[:, 0]
  current_fx = float(fx_df.iloc[-1]) if not fx_df.empty else 15800.0
except Exception:
  current_fx = 15800.0

asset_data = []

for ticker, meta in universe.items():
  try:
    df = yf.download(ticker, period="3y", progress=False)
    if df.empty or "Close" not in df.columns:
      continue
    prices = df["Close"]
    if isinstance(prices, pd.DataFrame):
      prices = prices.iloc[:, 0]

    # Convert USD asset prices to IDR equivalent using historical FX rates
    if meta["currency"] == "USD":
      try:
        fx_aligned = (
            yf.download("IDR=X", period="3y", progress=False)["Close"]
            .reindex(prices.index)
            .ffill()
        )
        if isinstance(fx_aligned, pd.DataFrame):
          fx_aligned = fx_aligned.iloc[:, 0]
        prices = prices * fx_aligned
      except Exception:
        prices = prices * current_fx

    returns = prices.pct_change().dropna()
    if len(returns) < 50:
      continue

    ann_return = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = float((ann_return - 0.06) / ann_vol) if ann_vol > 0 else 0.0

    # Composite quantitative score penalized by transaction friction & volatility
    score = round(
        max(
            0,
            min(
                100,
                50
                + (ann_return * 65)
                - (ann_vol * 20)
                - (meta["fee"] * 200),
            ),
        ),
        1,
    )

    # 4-Year Monte Carlo Projections in IDR
    log_rets = np.log(1 + returns)
    mu, sigma = log_rets.mean(), log_rets.std()
    sim_results = []
    np.random.seed(42)
    for _ in range(1000):
      shocks = np.random.normal(mu, sigma, 1008)
      path_val = 10000000 * np.prod(np.exp(shocks))
      sim_results.append(path_val)

    p5 = float(np.percentile(sim_results, 5))
    median = float(np.percentile(sim_results, 50))
    p95 = float(np.percentile(sim_results, 95))

    # Risk Audit (Why NOT this investment?)
    strengths = [
        f"Robust Sharpe ratio ({sharpe:.2f})"
        if sharpe > 0.7
        else "Acceptable liquidity",
        f"Annualized IDR return of +{round(ann_return*100, 1)}%",
    ]
    weaknesses = [
        f"Volatility index at {round(ann_vol*100, 1)}%",
        f"Estimated transaction friction: {meta['fee']*100}%",
    ]
    failure_condition = (
        "Model ranking degrades if market volatility surges > 30%"
        if ann_vol > 0.2
        else "Sensitive to rapid IDR currency appreciation"
    )

    asset_data.append({
        "ticker": ticker,
        "name": meta["name"],
        "category": meta["category"],
        "currency": meta["currency"],
        "score": score,
        "annual_return": round(ann_return * 100, 2),
        "volatility": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "fee_pct": meta["fee"] * 100,
        "monte_carlo": {
            "bear_5th": round(p5, -3),
            "base_median": round(median, -3),
            "bull_95th": round(p95, -3),
        },
        "audit": {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "failure_condition": failure_condition,
        },
    })
  except Exception as e:
    print(f"Error processing {ticker}: {e}")

asset_data = sorted(asset_data, key=lambda x: x["score"], reverse=True)

output = {
    "top_opportunities": asset_data[:4],
    "full_universe": asset_data,
    "fx_rate_usd_idr": current_fx,
}

with open("data/assets.json", "w") as f:
  json.dump(output, f, indent=4)

print("Institutional engine pipeline completed successfully!")
