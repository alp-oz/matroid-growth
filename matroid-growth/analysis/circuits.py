import numpy as np
from matroid_core.fields import GF2

def get_circuit_participation_by_birth(matroid_data, window_size=500):
    """
    Analyzes 'Nullity' (Circuit density) as a function of time.
    High Nullity = Many long circuits.
    Low Nullity = Mostly independent (Banal) columns.
    """
    cols = matroid_data['columns']
    R = matroid_data['R_final']
    n = len(cols)
    
    time_series = []
    
    # We slide a window through the simulation's birth history
    for start in range(0, n - window_size, window_size):
        end = start + window_size
        subset = cols[start:end]
        
        # Nullity = Number of columns - Rank of columns
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
    A small girth (3 or 4) means the matroid is 'locally tangled'.
    As R grows, we expect the Girth to increase (the 'Dilution' effect).
    """
    for g in range(3, max_girth + 1):
        for _ in range(samples):
            # Pick 'g' random columns
            idx = np.random.choice(len(columns), size=g, replace=False)
            sample = [columns[i] for i in idx]
            
            # If Rank < Number of elements, we found a circuit of size g
            if GF2.get_rank(sample, R_final) < g:
                return g
    return None # No small circuits found