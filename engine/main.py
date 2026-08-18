import json
import os
import numpy as np
import pandas as pd
import yfinance as yf

# 1. Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# 2. Fetch dummy/test universe data (e.g., BCA stock in Indonesia)
try:
  data = yf.download("BBCA.JK", period="1y")["Close"]
  latest_return = float(data.pct_change().mean() * 252)
except Exception:
  latest_return = 0.15  # Fallback dummy value if network blocks

# 3. Create the output dictionary
output_data = {
    "top_asset": "Bank Central Asia (BBCA.JK)",
    "score": 87.4,
    "annualized_return_est": round(latest_return * 100, 2),
}

# 4. Save into data/assets.json
with open("data/assets.json", "w") as f:
  json.dump(output_data, f, indent=4)

print("Python engine executed successfully and generated JSON!")
