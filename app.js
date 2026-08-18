let globalData = null;

fetch('data/assets.json')
    .then(response => response.json())
    .then(data => {
        globalData = data;
        renderDashboard();
    })
    .catch(error => {
        console.error("Error loading data:", error);
        document.getElementById('cards-container').innerHTML = "<p style='color: #f87171;'>Run the GitHub Action to generate global asset data.</p>";
    });

function updateDashboard() {
    if (globalData) renderDashboard();
}

function renderDashboard() {
    const capital = parseFloat(document.getElementById('user-capital').value) || 10000000;
    const horizon = parseFloat(document.getElementById('user-horizon').value) || 4;
    const scaleFactor = capital / 10000000; // Scaled from default 10M base simulation
    const horizonMultiplier = horizon / 4; // Adjusts Monte Carlo spread based on years

    // 1. Render Top 4 Cards
    const cardsContainer = document.getElementById('cards-container');
    cardsContainer.innerHTML = '';
    
    globalData.top_opportunities.forEach((asset, index) => {
        const baseVal = asset.monte_carlo.base_median * scaleFactor * horizonMultiplier;
        const bearVal = asset.monte_carlo.bear_5th * scaleFactor * horizonMultiplier;
        const bullVal = asset.monte_carlo.bull_95th * scaleFactor * horizonMultiplier;

        cardsContainer.innerHTML += `
            <div class="asset-card">
                <h3>#${index + 1} ${asset.ticker}</h3>
                <p style="font-size: 11px; color: #94a3b8; margin: 0 0 5px 0;">${asset.name} (${asset.category})</p>
                <div class="score">${asset.score} <span style="font-size:12px; color:#94a3b8;">Score</span></div>
                <p style="font-size: 12px; margin: 6px 0 0 0;">Annual Return: +${asset.annual_return}%</p>
                <div class="simulation-box">
                    <strong>Monte Carlo (${horizon}Y):</strong><br>
                    🐻 Bear (5th): Rp ${(bearVal/1000000).toFixed(1)}M<br>
                    🎯 Base (Med): Rp ${(baseVal/1000000).toFixed(1)}M<br>
                    🐂 Bull (95th): Rp ${(bullVal/1000000).toFixed(1)}M
                </div>
            </div>
        `;
    });

    // 2. Render Plotly Bar Chart for all assets in the universe
    const universe = globalData.full_universe;
    const tickers = universe.map(a => `${a.ticker} (${a.category.split(' ')[0]})`);
    const scores = universe.map(a => a.score);

    const trace = {
        x: tickers,
        y: scores,
        type: 'bar',
        marker: {
            color: scores.map(s => s > 75 ? '#34d399' : '#38bdf8')
        }
    };

    const layout = {
        title: { text: 'Global Asset Quantitative Scores across All Asset Classes', font: { color: '#f8fafc', size: 16 } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8' },
        xaxis: { title: 'Asset & Category', tickangle: -30 },
        yaxis: { title: 'Composite Score (0-100)' },
        margin: { t: 40, r: 20, b: 80, l: 50 }
    };

    Plotly.newPlot('plotly-bar-chart', [trace], layout, {responsive: true});
}
