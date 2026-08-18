import pandas as pd

def calculate_scores(metrics_df):
    # metrics_df contains Return, Volatility, MaxDrawdown
    # 1. Normalize metrics using Z-scores
    z_scores = (metrics_df - metrics_df.mean()) / metrics_df.std()
    
    # 2. Invert negative metrics (lower volatility/drawdown is better)
    z_scores['Volatility'] = z_scores['Volatility'] * -1
    z_scores['MaxDrawdown'] = z_scores['MaxDrawdown'] * -1
    
    # 3. Apply configurable weights
    weights = {'Return': 0.40, 'Volatility': 0.30, 'MaxDrawdown': 0.30}
    
    # Calculate final composite score (0 to 100 scale mapping)
    scores = (z_scores * pd.Series(weights)).sum(axis=1)
    normalized_scores = 50 + (scores * 15) # Scale to a recognizable number
    
    return normalized_scores.clip(0, 100).round(1)
