# %% Dependent elements alter dependencies: "Rich-get-Richer" + "Weak-get-Weaker"
import numpy as np

def run_model_2_refinement(steps=1000, initial_r=10, alpha=0.1, k=3, swap_rate=1):
    """
    Model 2: Growth + Refinement (Edge Swapping)
    swap_rate: How many 'refinements' happen per step.
    """
    r = initial_r
    # We use a dense matrix for Model 2 because we need to frequently update entries
    # (For very large scales, we would use a dok_matrix)
    matrix_A = np.zeros((r, 0), dtype=int)
    row_degrees = np.ones(r)

    for t in range(steps):
        # --- 1. GROWTH PHASE (Same as Model 1) ---
        if np.random.rand() < alpha:
            r += 1
            matrix_A = np.vstack([matrix_A, np.zeros((1, matrix_A.shape[1]), dtype=int)])
            row_degrees = np.append(row_degrees, 1)
        else:
            probs = row_degrees / row_degrees.sum()
            selected_rows = np.random.choice(range(r), size=min(k, r), replace=False, p=probs)
            new_col = np.zeros((r, 1), dtype=int)
            for row_idx in selected_rows:
                new_col[row_idx, 0] = 1
                row_degrees[row_idx] += 1
            matrix_A = np.hstack([matrix_A, new_col])

        # --- 2. REFINEMENT PHASE (The Swap) ---
        if matrix_A.shape[1] > 0: # Only if we have columns to swap
            for _ in range(swap_rate):
                # Pick a random column
                col_idx = np.random.randint(matrix_A.shape[1])
                ones_in_col = np.where(matrix_A[:, col_idx] == 1)[0]
                
                if len(ones_in_col) > 0:
                    # Find the 'weakest' row in this circuit (lowest degree)
                    weakest_row = ones_in_col[np.argmin(row_degrees[ones_in_col])]
                    
                    # Pick a 'hub' to move to
                    probs = row_degrees / row_degrees.sum()
                    target_row = np.random.choice(range(r), p=probs)
                    
                    # Perform the Swap (if it's a different row and not already in col)
                    if target_row != weakest_row and matrix_A[target_row, col_idx] == 0:
                        matrix_A[weakest_row, col_idx] = 0
                        matrix_A[target_row, col_idx] = 1
                        
                        # Update degrees
                        row_degrees[weakest_row] -= 1
                        row_degrees[target_row] += 1

    return matrix_A, row_degrees