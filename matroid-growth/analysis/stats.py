import numpy as np

def get_zipf_distribution(row_usage):
    """
    Transforms raw row-usage into (Rank, Frequency) coordinates.
    Essential for the Log-Log plots in the grant proposal.
    """
    # 1. Sort usage in descending order (High degree to low degree)
    counts = np.sort(row_usage)[::-1]
    
    # 2. Assign ranks (1, 2, 3...)
    # Filter out zero-usage rows to avoid log(0) issues
    counts = counts[counts > 0]
    ranks = np.arange(1, len(counts) + 1)
    
    return ranks, counts

def calculate_shannon_entropy(row_usage):
    """
    Measures the 'Complexity' of the row-usage distribution.
    Low Entropy = High Clustering (Dominant Hubs).
    High Entropy = Uniform Randomness.
    """
    total = np.sum(row_usage)
    if total == 0:
        return 0
        
    probs = row_usage / total
    # Shanon Entropy formula: -sum(p * log2(p))
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def get_rank_nullity_stats(matroid_data):
    """
    Returns the basic 'Shape' of the matroid:
    Columns (N), Rows (R), and the global Nullity.
    """
    n = len(matroid_data['columns'])
    r_final = matroid_data['R_final']
    # Total row-usage sum should equal N * k (if k is fixed)
    total_edges = np.sum(matroid_data['row_usage'])
    
    return {
        "n_columns": n,
        "r_rows": r_final,
        "avg_degree": total_edges / r_final if r_final > 0 else 0
    }

def calculate_clustering_metrics(columns, final_rank):
    """
    Analyzes how 'clumped' the columns are in the basis.
    """
    n_cols = len(columns)
    # Count how many unique columns were actually generated
    unique_cols = len(set(tuple(c) for c in columns))
    
    # Calculate column collision rate (Higher = more likely to hit F7)
    collision_rate = 1.0 - (unique_cols / n_cols)
    
    # Calculate Row Saturation: How many rows are actually being used?
    used_rows = set()
    for col in columns:
        used_rows.update(col)
    
    row_saturation = len(used_rows) / final_rank if final_rank > 0 else 0
    
    return {
        "collision_rate": collision_rate,
        "row_saturation": row_saturation,
        "efficiency": unique_cols / (2**final_rank if final_rank < 20 else 1e9)
    }