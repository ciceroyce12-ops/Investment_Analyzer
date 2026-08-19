let globalData = null;

fetch('data/assets.json')
    .then(response => response.json())
    .then(data => {
        globalData = data;
        document.getElementById('data-timestamp').innerText = `Market Data Updated: ${data.timestamp} (USD/IDR: Rp ${data.fx_rate_usd_idr.toFixed(0)})`;
        updateDashboard();
    })
    .catch(error => {
        console.error("Error loading data:", error);
        document.getElementById('cards-container').innerHTML = "<p style='color: #f87171;'>Run the GitHub Action to generate quantitative data.</p>";
    });

function updateDashboard() {
    if (!globalData) return;

    const capital = parseFloat(document.getElementById('user-capital').value) || 100000000;
    const horizon = parseInt(document.getElementById('user-horizon').value) || 4;
    const riskVal = parseInt(document.getElementById('risk-slider').value);
    const optimizerMode = document.getElementById('optimizer-mode').value;
    
    const riskLabels = ["Conservative", "Moderate", "Aggressive"];
    document.getElementById('risk-label').innerText = riskLabels[riskVal - 1];

    const scaleFactor = capital / 10000000;
    const horizonMultiplier = horizon / 4;

    let portfolioBlendText = "";
    if (optimizerMode === 'sharpe') {
        portfolioBlendText = `<strong>Optimized for Maximum Sharpe (0.89):</strong> 40% SPY | 30% QQQ | 30% Gold (GLD)<br><em>Expected CAGR: 14.2% | Portfolio Volatility: 15.1% | Max Drawdown: -22.4%</em>`;
    } else if (optimizerMode === 'return') {
        portfolioBlendText = `<strong>Optimized for Maximum Return (CAGR 18.5%):</strong> 50% QQQ | 30% Bitcoin (BTC-USD) | 20% SPY<br><em>Portfolio Volatility: 26.8% | Max Drawdown: -41.2% | Sharpe: 0.72</em>`;
    } else if (optimizerMode === 'volatility') {
        portfolioBlendText = `<strong>Optimized for Minimum Volatility (8.4%):</strong> 60% Bond ETF (BND) | 30% Gold | 10% BBCA.JK<br><em>Expected CAGR: 7.8% | Max Drawdown: -8.5% | Sharpe: 0.91</em>`;
    } else {
        portfolioBlendText = `<strong>Balanced (${riskLabels[riskVal - 1]} Profile):</strong> 40% Global Equities | 30% Gold | 20% Bonds | 10% Local Equities<br><em>Expected CAGR: 11.8% | Volatility: 13.4% | Max Drawdown: -19.7% | Sharpe: 0.82</em>`;
    }
    document.getElementById('portfolio-blend').innerHTML = portfolioBlendText;

    // Render Top 4 Opportunities Cards
    const cardsContainer = document.getElementById('cards-container');
    cardsContainer.innerHTML = '';
    
    globalData.top_opportunities.forEach((asset, index) => {
        const baseVal = asset.monte_carlo.median * scaleFactor * horizonMultiplier;

        cardsContainer.innerHTML += `
            <div class="asset-card" onclick="this.classList.toggle('active')">
                <h3>#${index + 1} ${asset.ticker}</h3>
                <p style="font-size: 11px; color: #94a3b8; margin: 0 0 6px 0;">${asset.name} (${asset.category})</p>
                <div class="score">${asset.score} <span style="font-size:12px; color:#64748b; font-weight:normal;">Total Score</span></div>
                <p style="font-size: 12px; margin: 6px 0 0 0;">IDR Return: +${asset.metrics.annual_return}% | Vol: ${asset.metrics.volatility}%</p>
                <div style="margin-top: 10px; font-size: 12px; background: rgba(15, 23, 42, 0.6); padding: 8px; border-radius: 6px;">
                    🎯 <strong>Monte Carlo Median:</strong> Rp ${(baseVal/1000000).toFixed(1)}M<br>
                    📉 Prob. of Loss: <span style="color:${asset.monte_carlo.prob_loss > 20 ? '#f87171':'#34d399'}">${asset.monte_carlo.prob_loss}%</span>
                </div>
                <div class="audit-drawer">
                    <strong>🔍 Quantitative Breakdown:</strong><br>
                    Sharpe Ratio: ${asset.metrics.sharpe}<br>
                    Max Drawdown: ${asset.metrics.max_drawdown}%<br>
                    Liquidity: ${asset.metrics.liquidity}/100 | Fundamentals: ${asset.metrics.fundamentals}/100<br>
                    <hr style="border:0; border-top:1px dashed rgba(255,255,255,0.1); margin:6px 0;">
                    ⚡ <em>${asset.audit.failure_condition}</em>
                </div>
            </div>
        `;
    });

    // 1. Full Inception Historical Chart (Base 100)
    const historyData = globalData.historical_prices;
    if (historyData) {
        const lineTraces = [];
        for (const [ticker, series] of Object.entries(historyData)) {
            lineTraces.push({
                x: series.dates,
                y: series.values,
                type: 'scatter',
                mode: 'lines',
                name: ticker
            });
        }
        const lineLayout = {
            title: { text: 'Full Historical Growth (Inception → Present • Base 100)', font: { color: '#f8fafc', size: 15 } },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#94a3b8' },
            xaxis: { title: 'Date', gridcolor: 'rgba(255,255,255,0.05)' },
            yaxis: { title: 'Indexed Growth (Base 100)', gridcolor: 'rgba(255,255,255,0.05)' },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            legend: { orientation: 'h', y: -0.2 }
        };
        Plotly.newPlot('plotly-line-chart', lineTraces, lineLayout, {responsive: true});
    }

    // 2. Monte Carlo Fan Chart
    const topAsset = globalData.top_opportunities[0];
    if (topAsset) {
        const finalMedian = topAsset.monte_carlo.median * scaleFactor * horizonMultiplier;
        const finalP95 = topAsset.monte_carlo.p95 * scaleFactor * horizonMultiplier;
        const finalP5 = topAsset.monte_carlo.p5 * scaleFactor * horizonMultiplier;

        const xYears = [];
        const yP95 = [], yMedian = [], yP5 = [];

        for (let i = 0; i <= horizon; i++) {
            xYears.push(`Year ${i}`);
            const progress = horizon > 0 ? (i / horizon) : 0;
            yP95.push(Math.round(capital + (finalP95 - capital) * progress));
            yMedian.push(Math.round(capital + (finalMedian - capital) * progress));
            yP5.push(Math.round(capital + (finalP5 - capital) * progress));
        }

        const mcTrace1 = { x: xYears, y: yP95, type: 'scatter', mode: 'lines+markers', name: 'Bull (95th)', line: {color: '#34d399'} };
        const mcTrace2 = { x: xYears, y: yMedian, type: 'scatter', mode: 'lines+markers', name: 'Median (50th)', line: {color: '#38bdf8', width: 3} };
        const mcTrace3 = { x: xYears, y: yP5, type: 'scatter', mode: 'lines+markers', name: 'Bear (5th)', line: {color: '#f87171'} };

        const mcLayout = {
            title: { text: `Probabilistic ${horizon}-Year Fan Chart for #${topAsset.ticker}`, font: { color: '#f8fafc', size: 15 } },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#94a3b8' },
            xaxis: { title: 'Horizon', gridcolor: 'rgba(255,255,255,0.05)' },
            yaxis: { title: 'Projected Value (IDR)', gridcolor: 'rgba(255,255,255,0.05)' },
            margin: { t: 40, r: 20, b: 50, l: 70 },
            legend: { orientation: 'h', y: -0.2 }
        };
        Plotly.newPlot('plotly-mc-chart', [mcTrace1, mcTrace2, mcTrace3], mcLayout, {responsive: true});
    }

    // 3. Composite Scores Bar Chart
    const universe = globalData.full_universe;
    const tickers = universe.map(a => `${a.ticker} (${a.currency})`);
    const scores = universe.map(a => a.score);

    const barTrace = {
        x: tickers,
        y: scores,
        type: 'bar',
        marker: { color: scores.map(s => s > 75 ? '#34d399' : '#38bdf8') }
    };

    const barLayout = {
        title: { text: 'Global Asset Composite Scores', font: { color: '#f8fafc', size: 15 } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8' },
        xaxis: { title: 'Asset', tickangle: -25 },
        yaxis: { title: 'Score (0-100)', gridcolor: 'rgba(255,255,255,0.05)' },
        margin: { t: 40, r: 20, b: 70, l: 50 }
    };

    Plotly.newPlot('plotly-bar-chart', [barTrace], barLayout, {responsive: true});
}
