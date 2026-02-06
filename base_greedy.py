"""
Project: Matroid Basis Evolution
Model Type: Binary Linear Matroid (GF2)
Logic: Basis Exchange Reproduction
Branch: basis_exchange
"""
import numpy as np
def run_greedy_model(vectors, rank_limit):
    basis = []
    for v in vectors:
        # Check if we have room AND if v is independent
        if len(basis) < rank_limit:
            if is_independent(basis, v):
                basis.append(v)
                print("Added to basis.")
        else:
            # In the old model, once it's full, we just ignore new vectors
            print("Basis full. Ignoring candidate.")
    return basis