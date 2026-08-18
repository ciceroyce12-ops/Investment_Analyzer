import yfinance as yf
import pandas as pd
import numpy as np

def fetch_universe_data():
    # Example universe: 2 IDX stocks, 1 US ETF, Gold
    tickers = ["BBCA.JK", "BBRI.JK", "SPY", "GLD"]
    
    # Fetch 5 years of daily closing prices
    data = yf.download(tickers, period="5y")['Close']
    
    # Calculate daily returns
    returns = data.pct_change()
    
    # Calculate annualized metrics (252 trading days)
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Calculate Maximum Drawdown
    rolling_max = data.cummax()
    drawdown = (data - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return data, annual_return, annual_volatility, max_drawdown
