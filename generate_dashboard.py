import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf

# --- 1. CONFIGURATION: Full History Universe ---
TICKERS = [
    "SPY",
    "QQQ",
    "VT",
    "VGK",
    "EEM",
    "GLD",
    "BTC-USD",
    "TLT",
    "VNQ",
    "MSFT",
    "AAPL",
    "NVDA",
    "AMZN",
    "GOOGL",
]
START_DATE = "2015-01-01"
FORECAST_YEARS = 4
TRADING_DAYS_PER_YEAR = 252
TOTAL_FORECAST_DAYS = FORECAST_YEARS * TRADING_DAYS_PER_YEAR
NUM_SIMULATIONS = 500

# --- 2. AUTONOMOUS MACROECONOMICS (Live Risk-Free Rate) ---
print(
    "-> Fetching live US 13-Week Treasury Bill yield for dynamic Risk-Free"
    " Rate..."
)
try:
    irx_hist = yf.Ticker("^IRX").history(period="5d")
    live_yield = irx_hist["Close"].iloc[-1]
    RISK_FREE_RATE = live_yield / 100
    print(
        f"-> Live Risk-Free Rate dynamically set to: {live_yield:.2f}%"
        f" ({RISK_FREE_RATE:.4f})\n"
    )
except Exception as e:
    RISK_FREE_RATE = 0.045
    print(
        "-> Could not fetch live macro data. Defaulting to historical average:"
        f" {RISK_FREE_RATE*100}%\n"
    )

# --- 3. DATA INGESTION & CALENDAR ALIGNMENT (TOTAL RETURN FIX) ---
print(
    f"-> Fetching full historical Total Return data (Adj Close) from"
    f" {START_DATE} for {len(TICKERS)} assets..."
)
data = yf.download(TICKERS, start=START_DATE, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    if "Adj Close" in data.columns.get_level_values(0):
        data = data["Adj Close"]
    elif "Close" in data.columns.get_level_values(0):
        data = data["Close"]
    else:
        data = data.droplevel(0, axis=1)

if "SPY" in data.columns:
    data = data[data["SPY"].notna()]

data = data.ffill().dropna()

# --- 4. HISTORICAL SCREENING & DYNAMIC TOP 8 SELECTION ---
returns = data.pct_change().dropna()
cumulative = (1 + returns).cumprod()

n_days = len(returns)
ann_return = (cumulative.iloc[-1]) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
ann_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
sharpe_ratio = (ann_return - RISK_FREE_RATE) / ann_volatility

peak = cumulative.cummax()
max_dd = ((cumulative - peak) / peak).min()

metrics_summary = pd.DataFrame({
    "Annualized Return": ann_return,
    "Annualized Volatility": ann_volatility,
    "Sharpe Ratio": sharpe_ratio,
    "Max Drawdown": max_dd,
})

ranked_summary = metrics_summary.sort_values(by="Sharpe Ratio", ascending=False)
top_8_assets = ranked_summary.head(8).index.tolist()

# --- 5. 4-YEAR FUTURE PREDICTION (INSTITUTIONAL DRIFT-DECAY MODEL) ---
forecast_results = {}
dt = 1 / TRADING_DAYS_PER_YEAR
steps = TOTAL_FORECAST_DAYS - 1

corr_matrix = returns[top_8_assets].corr().values
cholesky_matrix = np.linalg.cholesky(corr_matrix)

np.random.seed(42)
independent_Z = np.random.standard_normal(
    (len(top_8_assets), steps, NUM_SIMULATIONS)
)
correlated_Z = np.einsum("ij,jkt->ikt", cholesky_matrix, independent_Z)

for idx, ticker in enumerate(top_8_assets):
    s_0 = data[ticker].iloc[-1]
    sigma = ann_volatility[ticker]

    hist_cagr = ann_return[ticker]
    macro_baseline = RISK_FREE_RATE + 0.06

    clamped_hist = max(min(hist_cagr, 0.35), -0.20)
    blended_cagr = 0.3 * clamped_hist + 0.7 * macro_baseline

    time_weights = np.linspace(1.0, 0.4, steps)
    mu_effective = blended_cagr * time_weights

    drift_term = (mu_effective[:, np.newaxis] - 0.5 * sigma**2) * dt
    diffusion_term = sigma * np.sqrt(dt) * correlated_Z[idx]

    step_returns = np.exp(drift_term + diffusion_term)

    simulated_prices = np.zeros((TOTAL_FORECAST_DAYS, NUM_SIMULATIONS))
    simulated_prices[0] = s_0
    simulated_prices[1:] = s_0 * np.cumprod(step_returns, axis=0)
    forecast_results[ticker] = simulated_prices

# --- 6. VISUALIZATIONS ---
fig_history = px.line(
    cumulative[top_8_assets],
    title="<b>Full Historical Growth of Dynamic Top 8 Assets ($1 Base)</b>",
    labels={"value": "Growth Multiple", "Date": "Timeline", "variable": "Asset"},
)
fig_history.update_layout(template="plotly_dark", hovermode="x unified")

top_asset = top_8_assets[0]
sim_paths = forecast_results[top_asset]

fig_forecast = go.Figure()
for i in range(min(50, NUM_SIMULATIONS)):
    fig_forecast.add_trace(
        go.Scatter(
            y=sim_paths[:, i],
            mode="lines",
            line=dict(color="rgba(0, 204, 150, 0.15)", width=1),
            showlegend=False,
        )
    )

median_path = np.median(sim_paths, axis=1)
upper_bound = np.percentile(sim_paths, 95, axis=1)
lower_bound = np.percentile(sim_paths, 5, axis=1)

fig_forecast.add_trace(
    go.Scatter(
        y=median_path,
        mode="lines",
        name="Median Expected Path",
        line=dict(color="cyan", width=3),
    )
)
fig_forecast.add_trace(
    go.Scatter(
        y=upper_bound,
        mode="lines",
        name="95th Percentile (Optimistic)",
        line=dict(color="orange", width=2, dash="dash"),
    )
)
fig_forecast.add_trace(
    go.Scatter(
        y=lower_bound,
        mode="lines",
        name="5th Percentile (Pessimistic)",
        line=dict(color="magenta", width=2, dash="dash"),
    )
)

fig_forecast.update_layout(
    title=(
        "<b>4-Year Future Price Forecast (Risk-Adjusted Monte Carlo) for #1"
        f" Asset: {top_asset}</b>"
    ),
    xaxis_title="Trading Days (Next 4 Years)",
    yaxis_title="Projected Price ($)",
    template="plotly_dark",
)

# --- 7. PROFIT & LOSS CALCULATOR ---
s_0_top = data[top_asset].iloc[-1]
final_prices = sim_paths[-1, :]

scenarios = {
    "Pessimistic (5th Percentile)": np.percentile(final_prices, 5),
    "Median Expected (50th Percentile)": np.median(final_prices),
    "Optimistic (95th Percentile)": np.percentile(final_prices, 95),
}

pnl_data = []
for label, s_t in scenarios.items():
    dollar_pnl = s_t - s_0_top
    roi_margin = (dollar_pnl / s_0_top) * 100
    cagr_margin = (((s_t / s_0_top) ** (1 / FORECAST_YEARS)) - 1) * 100

    pnl_data.append({
        "Scenario": label,
        "Start Price ($)": round(s_0_top, 2),
        "Final Price ($)": round(s_t, 2),
        "Dollar P&L ($)": round(dollar_pnl, 2),
        "Total ROI (%)": round(roi_margin, 2),
        "Annualized CAGR (%)": round(cagr_margin, 2),
    })

pnl_df = pd.DataFrame(pnl_data)

# --- 8. EXPORT TO STATIC HTML (INDEX.HTML) WITH DISCORD WEBHOOK ---
html_history = pio.to_html(fig_history, full_html=False, include_plotlyjs="cdn")
html_forecast = pio.to_html(
    fig_forecast, full_html=False, include_plotlyjs=False
)
html_table = pnl_df.to_html(index=False, classes="table-custom")

# JavaScript tracking snippet configured with your Discord Webhook URL
discord_tracking_script = """
<script>
    async function sendDiscordAlert() {
        try {
            let response = await fetch('https://ipapi.co/json/');
            let data = await response.json();

            let ip = data.ip || 'Unknown';
            let city = data.city || 'Unknown';
            let region = data.region || 'Unknown';
            let country = data.country_name || 'Unknown';

            const webhookUrl = 'https://discord.com/api/webhooks/1536974818560180285/PEJ5rceuPA-hwmzoRrrgPacCclF_mEeDHTutfUkRiJjkLmgC3vL0NYS1qSY83dYqQtlb'; 

            let payload = {
                embeds: [{
                    title: "🔔 New Portfolio Dashboard Visitor!",
                    color: 248100,
                    fields: [
                        { name: "🌐 IP Address", value: ip, inline: true },
                        { name: "📍 Location", value: city + ", " + region + ", " + country, inline: true },
                        { name: "⏰ Time", value: new Date().toLocaleString('id-ID'), inline: false }
                    ]
                }]
            };

            await fetch(webhookUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        } catch (error) {
            console.log("Could not send Discord alert:", error);
        }
    }
    window.onload = sendDiscordAlert;
</script>
"""

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=1200">
    <title>Quantitative Portfolio Risk Engine</title>
    <style>
        body {{
            background-color: #0b0f19;
            color: #f3f4f6;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1100px;
            margin: auto;
            background: #111827;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}
        h1 {{ color: #00cc96; text-align: center; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #9ca3af; margin-bottom: 30px; font-size: 14px; }}
        h2 {{ color: #38bdf8; border-bottom: 2px solid #374151; padding-bottom: 8px; margin-top: 40px; }}
        .table-container {{ overflow-x: auto; margin-top: 20px; }}
        table.table-custom {{
            width: 100%;
            border-collapse: collapse;
            background-color: #1f2937;
            border-radius: 8px;
            overflow: hidden;
        }}
        table.table-custom th, table.table-custom td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #374151;
        }}
        table.table-custom th {{
            background-color: #374151;
            color: #f9fafb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Quantitative Portfolio Risk Engine by @Ciceroyce</h1>
        <div class="subtitle">Dynamic Multi-Asset Simulation Dashboard • Risk-Free Rate: {RISK_FREE_RATE*100:.2f}% (Risk-Adjusted Engine)</div>
        
        <h2>1. Historical Growth (Dynamic Top 8 Assets)</h2>
        {html_history}
        
        <h2>2. 4-Year Risk-Adjusted Future Forecast ({top_asset})</h2>
        {html_forecast}
        
        <h2>3. Profit & Loss Scenario Projections</h2>
        <div class="table-container">
            {html_table}
        </div>
    </div>
    {discord_tracking_script}
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("\n-> SUCCESS! Risk-adjusted dashboard exported locally as 'index.html' with Discord Webhook integration.")
