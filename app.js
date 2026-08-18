// Look right here: because everything is in the root, it's just 'data/assets.json' !
fetch('data/assets.json')
    .then(response => response.json())
    .then(data => {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = `<p>Successfully loaded top asset: <strong>${data.top_asset}</strong> with a score of <strong>${data.score}</strong>!</p>`;
    })
    .catch(error => {
        document.getElementById('results').innerHTML = "<p style='color: #f87171;'>Waiting for the first GitHub Action run to generate data...</p>";
    });
