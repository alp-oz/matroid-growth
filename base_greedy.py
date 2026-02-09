"""
Project: Matroid Basis Evolution
Model Type: Binary Linear Matroid (GF2)
Logic: Basis Exchange Reproduction
Branch: basis_exchange
"""
import numpy as np

def is_independent(basis, vector):
    if len(basis) == 0: return not np.all(vector == 0)
    matrix = np.array(basis + [vector])
    # In binary matroids, rank is checked via Gaussian elimination
    return np.linalg.matrix_rank(matrix) > len(basis)

def run_greedy():
    RANK = 3
    basis = []
    # 10 random 5D vectors
    universe = [np.random.randint(0, 2, 5) for _ in range(10)]

    print("--- Starting Greedy Model ---")
    for i, v in enumerate(universe):
        if len(basis) < RANK:
            if is_independent(basis, v):
                basis.append(v)
                print(f"Vector {i}: Added.")
        else:
            print(f"Vector {i}: Basis full. Ignored.")
    
    print(f"Final Greedy Basis size: {len(basis)}")

if __name__ == "__main__":
    run_greedy()