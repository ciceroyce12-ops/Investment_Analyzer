fetch('data/assets.json')
    .then(response => response.json())
    .then(data => {
        // 1. Render Top 4 Cards
        const cardsContainer = document.getElementById('cards-container');
        cardsContainer.innerHTML = '';
        
        data.top_opportunities.forEach((asset, index) => {
            cardsContainer.innerHTML += `
                <div class="asset-card">
                    <h3>#${index + 1} ${asset.ticker}</h3>
                    <p style="font-size: 12px; color: #94a3b8; margin: 0 0 8px 0;">${asset.name}</p>
                    <div class="score">${asset.score} <span style="font-size:12px; color:#94a3b8;">Score</span></div>
                    <p style="font-size: 13px; margin: 8px 0 0 0;">Return: +${asset.annual_return}%</p>
                    <p style="font-size: 13px; margin: 4px 0 0 0;">Sharpe: ${asset.sharpe}</p>
                </div>
            `;
        });

        // 2. Render Plotly Bar Chart for all assets in the universe
        const universe = data.full_universe;
        const tickers = universe.map(a => a.ticker);
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
            title: { text: 'Global Asset Quantitative Scores', font: { color: '#f8fafc' } },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#94a3b8' },
            xaxis: { title: 'Asset Ticker' },
            yaxis: { title: 'Composite Score (0-100)' },
            margin: { t: 40, r: 20, b: 40, l: 50 }
        };

        Plotly.newPlot('plotly-bar-chart', [trace], layout, {responsive: true});
    })
    .catch(error => {
        console.error("Error loading data:", error);
        document.getElementById('cards-container').innerHTML = "<p style='color: #f87171;'>Run the GitHub Action to generate multi-asset data.</p>";
    });
