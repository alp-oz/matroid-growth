import numpy as np

def is_independent(basis, vector):
    if len(basis) == 0: return not np.all(vector == 0)
    matrix = np.array(basis + [vector])
    return np.linalg.matrix_rank(matrix) > len(basis)

def run_exchange():
    RANK = 3
    basis = []
    universe = [np.random.randint(0, 2, 5) for _ in range(10)]

    print("--- Starting Basis Exchange Model ---")
    for i, v in enumerate(universe):
        if len(basis) < RANK:
            if is_independent(basis, v):
                basis.append(v)
                print(f"Vector {i}: Added.")
        else:
            # THIS IS THE 'NEW' LOGIC:
            # Even if full, we 'swap' the oldest vector for the new one
            print(f"Vector {i}: Basis full. Swapping (Basis Exchange)...")
            basis.pop(0) # Remove oldest
            basis.append(v) # Add newest
    
    print(f"Final Exchange Basis size: {len(basis)}")

if __name__ == "__main__":
    run_exchange()