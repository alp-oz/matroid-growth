import numpy as np
from core.fields import GF2

def get_circuit_participation_by_birth(matroid_data, window_size=500):
    """
    Analyzes 'Nullity' (Circuit density) as a function of time.
    High Nullity = Many long circuits/dependencies.
    Low Nullity = Mostly independent (Banal) columns.
    """
    cols = matroid_data['columns']
    R = matroid_data['R_final']
    n = len(cols)
    
    time_series = []
    
    # We slide a window through the simulation's birth history
    # This detects if the matroid gets "denser" as it gets older
    for start in range(0, n - window_size, window_size):
        end = start + window_size
        subset = cols[start:end]
        
        # Nullity Calculation: (Number of columns) - (Linear Rank of those columns)
        # Higher nullity = more "tangled" dependencies in that window
        rank = GF2.get_rank(subset, R)
        nullity = len(subset) - rank
        
        time_series.append({
            'birth_window': (start, end),
            'nullity': nullity,
            'density': nullity / window_size
        })
        
    return time_series

def estimate_girth(columns, R_final, samples=50, max_girth=8):
    """
    The Girth is the size of the smallest circuit.
    - Girth 3: Locally dense (contains triangles or the Fano Plane).
    - High Girth: Sparse and 'diluted'.
    """
    for g in range(3, max_girth + 1):
        # We increase sampling for larger g because larger circuits are harder to find
        adjusted_samples = samples * (g - 1)
        
        for _ in range(adjusted_samples):
            # Pick 'g' random columns to test for dependency
            idx = np.random.choice(len(columns), size=g, replace=False)
            sample = [columns[i] for i in idx]
            
            # If Rank < size of sample, a circuit of size g exists
            if GF2.get_rank(sample, R_final) < g:
                return g
    return None # No small circuits found in the sample