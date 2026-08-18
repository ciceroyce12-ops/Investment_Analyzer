let globalData = null;

fetch('data/assets.json')
    .then(response => response.json())
    .then(data => {
        globalData = data;
        updateDashboard();
    })
    .catch(error => {
        console.error("Error loading data:", error);
        document.getElementById('cards-container').innerHTML = "<p style='color: #f87171;'>Run the GitHub Action to generate global asset data.</p>";
    });

function updateDashboard() {
    if (!globalData) return;

    const capital = parseFloat(document.getElementById('user-capital').value) || 10000000;
    const horizon = parseFloat(document.getElementById('user-horizon').value) || 4;
    const riskVal = parseInt(document.getElementById('risk-slider').value);
    
    const riskLabels = ["Conservative", "Moderate", "Aggressive"];
    document.getElementById('risk-label').innerText = riskLabels[riskVal - 1];

    const scaleFactor = capital / 10000000;
    const horizonMultiplier = horizon / 4;

    // 1. Portfolio Allocator logic based on risk slider
    const universe = globalData.full_universe;
    let portfolioBlendText = "";
    if (riskVal === 1) {
        portfolioBlendText = `<strong>Conservative Profile:</strong> 50% Fixed Income (BND) | 30% Gold (GLD) | 20% Blue-Chip Equities`;
    } else if (riskVal === 2) {
        portfolioBlendText = `<strong>Moderate Profile:</strong> 40% Global Equities (SPY/VT) | 30% Gold | 20% Fixed Income | 10% Indonesian Equities`;
    } else {
        portfolioBlendText = `<strong>Aggressive Profile:</strong> 50% Growth Equities (QQQ) | 30% Crypto (BTC) | 20% Global Equities`;
    }
    document.getElementById('portfolio-blend').innerHTML = portfolioBlendText;

    // 2. Render Top 4 Cards with Clickable Audit Drawers
    const cardsContainer = document.getElementById('cards-container');
    cardsContainer.innerHTML = '';
    
    globalData.top_opportunities.forEach((asset, index) => {
        const baseVal = asset.monte_carlo.base_median * scaleFactor * horizonMultiplier;
        const bearVal = asset.monte_carlo.bear_5th * scaleFactor * horizonMultiplier;
        const bullVal = asset.monte_carlo.bull_95th * scaleFactor * horizonMultiplier;

        cardsContainer.innerHTML += `
            <div class="asset-card" onclick="this.classList.toggle('active')">
                <h3>#${index + 1} ${asset.ticker}</h3>
                <p style="font-size: 11px; color: #94a3b8; margin: 0 0 5px 0;">${asset.name} (${asset.category})</p>
                <div class="score">${asset.score} <span style="font-size:12px; color:#94a3b8;">Score</span></div>
                <p style="font-size: 12px; margin: 6px 0 0 0;">IDR Return: +${asset.annual_return}% | Fee: ${asset.fee_pct}%</p>
                <div class="simulation-box">
                    <strong>Monte Carlo (${horizon}Y):</strong><br>
                    🐻 Bear: Rp ${(bearVal/1000000).toFixed(1)}M<br>
                    🎯 Base: Rp ${(baseVal/1000000).toFixed(1)}M<br>
                    🐂 Bull: Rp ${(bullVal/1000000).toFixed(1)}M
                </div>
                <div class="audit-drawer">
                    <strong>🔍 Risk Audit ("Why NOT"):</strong><br>
                    ✓ ${asset.audit.strengths[0]}<br>
                    ⚠ ${asset.audit.weaknesses[0]}<br>
                    ⚡ <em>${asset.audit.failure_condition}</em><br>
                    <span style="color: #38bdf8; font-size: 10px;">(Click to collapse)</span>
                </div>
            </div>
        `;
    });

    // 3. Render Plotly Bar Chart
    const tickers = universe.map(a => `${a.ticker} (${a.currency})`);
    const scores = universe.map(a => a.score);

    const trace = {
        x: tickers,
        y: scores,
        type: 'bar',
        marker: { color: scores.map(s => s > 75 ? '#34d399' : '#38bdf8') }
    };

    const layout = {
        title: { text: `Global Asset Scores (FX Rate: 1 USD = Rp ${globalData.fx_rate_usd_idr.toFixed(0)})`, font: { color: '#f8fafc', size: 15 } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8' },
        xaxis: { title: 'Asset & Currency', tickangle: -25 },
        yaxis: { title: 'Composite Score (0-100)' },
        margin: { t: 40, r: 20, b: 70, l: 50 }
    };

    Plotly.newPlot('plotly-bar-chart', [trace], layout, {responsive: true});
}
