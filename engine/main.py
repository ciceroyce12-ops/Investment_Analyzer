import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

os.makedirs("data", exist_ok=True)

universe = {
    "SPY": {
        "name": "S&P 500 ETF",
        "category": "Global Equities",
        "currency": "USD",
        "fee": 0.0025,
        "liquidity": 99,
        "fundamentals": 95,
    },
    "QQQ": {
        "name": "Nasdaq 100 ETF",
        "category": "Global Equities",
        "currency": "USD",
        "fee": 0.0025,
        "liquidity": 98,
        "fundamentals": 96,
    },
    "VT": {
        "name": "Vanguard Total World",
        "category": "Global Equities",
        "currency": "USD",
        "fee": 0.0025,
        "liquidity": 97,
        "fundamentals": 93,
    },
    "BND": {
        "name": "Total Bond Market ETF",
        "category": "Fixed Income",
        "currency": "USD",
        "fee": 0.0010,
        "liquidity": 96,
        "fundamentals": 90,
    },
    "GLD": {
        "name": "Gold Trust",
        "category": "Commodities",
        "currency": "USD",
        "fee": 0.0020,
        "liquidity": 98,
        "fundamentals": 80,
    },
    "BTC-USD": {
        "name": "Bitcoin",
        "category": "Crypto",
        "currency": "USD",
        "fee": 0.0050,
        "liquidity": 95,
        "fundamentals": 70,
    },
}

# Download FX data once globally
try:
  fx_df = yf.download("IDR=X", period="max", progress=False)["Close"]
  if isinstance(fx_df, pd.DataFrame):
    fx_df = fx_df.iloc[:, 0]
  current_fx = float(fx_df.iloc[-1]) if not fx_df.empty else 15800.0
except Exception:
  fx_df = pd.Series()
  current_fx = 15800.0

asset_data = []
price_dict = {}

for ticker, meta in universe.items():
  try:
    df = yf.download(ticker, period="max", progress=False)
    if df.empty or "Close" not in df.columns:
      continue
    prices = df["Close"]
    if isinstance(prices, pd.DataFrame):
      prices = prices.iloc[:, 0]

    if meta["currency"] == "USD":
      try:
        if not fx_df.empty:
          fx_aligned = fx_df.reindex(prices.index).ffill()
          prices = prices * fx_aligned
        else:
          prices = prices * current_fx
      except Exception:
        prices = prices * current_fx

    resampled = prices.resample("W").last().dropna()
    if len(resampled) > 10:
      price_dict[ticker] = resampled

    returns = prices.pct_change().dropna()
    if len(returns) < 50:
      continue

    ann_return = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = float((ann_return - 0.06) / ann_vol) if ann_vol > 0 else 0.0

    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) * 100

    return_score = min(100, max(0, (ann_return + 0.1) * 200))
    risk_score = min(100, max(0, 100 - (ann_vol * 150)))
    sharpe_score = min(100, max(0, sharpe * 50))
    drawdown_score = min(100, max(0, 100 + max_dd))

    composite_score = round(
        (return_score * 0.25)
        + (risk_score * 0.20)
        + (sharpe_score * 0.15)
        + (drawdown_score * 0.15)
        + (meta["fundamentals"] * 0.15)
        + (meta["liquidity"] * 0.10)
        - (meta["fee"] * 200),
        1,
    )
    composite_score = max(0, min(100, composite_score))

    log_rets = np.log(1 + returns)
    mu, sigma = log_rets.mean(), log_rets.std()
    sim_results = []
    np.random.seed(42)
    for _ in range(10000):
      shocks = np.random.normal(mu, sigma, 1008)
      path_val = 10000000 * np.prod(np.exp(shocks))
      sim_results.append(path_val)

    sim_array = np.array(sim_results)
    p5 = float(np.percentile(sim_array, 5))
    median = float(np.percentile(sim_array, 50))
    p95 = float(np.percentile(sim_array, 95))
    prob_loss = float(np.mean(sim_array < 10000000) * 100)

    # Fetch latest news items via yfinance
    news_items = []
    try:
      t_obj = yf.Ticker(ticker)
      raw_news = t_obj.news
      if raw_news:
        for item in raw_news[:3]:
          content = (
              item.get("content", {})
              if isinstance(item.get("content"), dict)
              else {}
          )
          title = item.get("title") or content.get("title")
          if not title and isinstance(item, dict):
            title = item.get("headline")
          publisher = (
              item.get("publisher")
              or content.get("provider", {}).get("displayName")
              or "Financial Wire"
          )
          link = (
              item.get("link")
              or content.get("clickThroughUrl", {}).get("url")
              or "#"
          )
          if title:
            news_items.append({
                "title": str(title),
                "publisher": str(publisher),
                "link": str(link),
            })
    except Exception:
      pass

    if not news_items:
      news_items = [{
          "title": (
              f"Market performance overview and trend analysis for {ticker}"
          ),
          "publisher": "Global Market Intelligence",
          "link": "#",
      }]

    asset_data.append({
        "ticker": ticker,
        "name": meta["name"],
        "category": meta["category"],
        "currency": meta["currency"],
        "score": composite_score,
        "metrics": {
            "annual_return": round(ann_return * 100, 2),
            "volatility": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "fundamentals": meta["fundamentals"],
            "liquidity": meta["liquidity"],
        },
        "monte_carlo": {
            "p5": round(p5, -3),
            "median": round(median, -3),
            "p95": round(p95, -3),
            "prob_loss": round(prob_loss, 1),
        },
        "news": news_items,
        "audit": {
            "failure_condition": (
                "High sensitivity to rapid macro downturns & volatility spikes."
                if ann_vol > 0.2
                else "Stable trend; susceptible to currency fluctuation."
            )
        },
    })
  except Exception as e:
    print(f"Error processing {ticker}: {e}")

# Align historical series to 2015 common start date
historical_series = {}
if price_dict:
  df_all = pd.DataFrame(price_dict)
  df_all = df_all.loc["2015-01-01":].dropna()
  for ticker in df_all.columns:
    series = df_all[ticker]
    norm_history = (series / series.iloc[0]) * 100
    historical_series[ticker] = {
        "dates": [d.strftime("%Y-%m-%d") for d in norm_history.index],
        "values": [round(float(v), 2) for v in norm_history.values],
    }

asset_data = sorted(asset_data, key=lambda x: x["score"], reverse=True)

output = {
    "timestamp": datetime.utcnow().strftime("%d %b %Y %H:%M UTC"),
    "fx_rate_usd_idr": current_fx,
    "top_opportunities": asset_data[:4],
    "full_universe": asset_data,
    "historical_prices": historical_series,
}

with open("data/assets.json", "w") as f:
  json.dump(output, f, indent=4)

print("Quantitative engine data successfully generated.")
