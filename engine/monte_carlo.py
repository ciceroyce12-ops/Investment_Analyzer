import numpy as np

def run_monte_carlo(prices, days=1008, simulations=10000):
    # 1008 trading days = 4 years
    log_returns = np.log(1 + prices.pct_change().dropna())
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    # Generate random paths
    simulated_paths = np.zeros((days, simulations))
    simulated_paths[0] = prices.iloc[-1]
    
    for t in range(1, days):
        random_shocks = np.random.normal(loc=mu, scale=sigma, size=simulations)
        simulated_paths[t] = simulated_paths[t-1] * np.exp(random_shocks)
        
    # Extract percentiles for the dashboard
    final_values = simulated_paths[-1]
    return {
        "5th": np.percentile(final_values, 5),
        "25th": np.percentile(final_values, 25),
        "median": np.median(final_values),
        "75th": np.percentile(final_values, 75),
        "95th": np.percentile(final_values, 95)
    }
