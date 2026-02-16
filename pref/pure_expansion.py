# %%
import numpy as np
import matplotlib.pyplot as plt

def run_expanding_model(steps=1000, initial_rank=10, innovation_rate=0.1):
    # Start with a small Identity Matrix (The initial Basis)
    rank = initial_rank
    # matrix_A will grow in height (rows) and width (columns)
    matrix_A = np.zeros((rank, 0), dtype=int)
    
    # Track degrees of the rows
    row_degrees = np.ones(rank)

    for t in range(steps):
        # Decide: New Row (Innovation) or New Column (Dependency)?
        if np.random.rand() < innovation_rate:
            # --- ADD NEW BASIS ELEMENT (New Row) ---
            rank += 1
            # Add a new row of zeros to matrix_A
            new_row = np.zeros((1, matrix_A.shape[1]), dtype=int)
            matrix_A = np.vstack([matrix_A, new_row])
            # Add a new degree counter for this row
            row_degrees = np.append(row_degrees, 1)
        else:
            # --- ADD NEW DEPENDENCY (New Column) ---
            # Use Preferential Attachment to pick k rows
            k = 3
            probs = row_degrees / row_degrees.sum()
            
            # Avoid picking more rows than we actually have
            n_to_pick = min(k, rank)
            selected_rows = np.random.choice(range(rank), size=n_to_pick, replace=False, p=probs)
            
            # Create the new column
            new_col = np.zeros((rank, 1), dtype=int)
            for r in selected_rows:
                new_col[r, 0] = 1
                row_degrees[r] += 1 # The "Rich-get-Richer" part
            
            matrix_A = np.hstack([matrix_A, new_col])

    return matrix_A, row_degrees

# Run and peek at the structure
A, degrees = run_expanding_model()
print(f"Final Rank: {A.shape[0]}, Final Elements: {A.shape[1]}")


# %%  Taking advantage of sparsity for larger runs
import numpy as np
from scipy import sparse

def run_model_1_sparse(steps=1000, initial_r=10, alpha=0.1, k=3):
    """
    Model 1: Pure Growth / Expansion
    alpha: Innovation rate (P(new row))
    k: Number of 1s per new column
    """
    r = initial_r
    # We use a list of lists to build the sparse matrix efficiently
    # matrix_A will store coordinates of 1s: (row_index, col_index)
    rows = []
    cols = []
    
    # Track degrees: every row starts with degree 1 (from the Identity matrix)
    row_degrees = np.ones(r, dtype=int)
    current_col_idx = 0

    for t in range(steps):
        if np.random.rand() < alpha:
            # INNOVATION: Add a new Basis Element (Row)
            r += 1
            row_degrees = np.append(row_degrees, 1)
        else:
            # DEPENDENCY: Add a new Column with exactly k ones
            probs = row_degrees / row_degrees.sum()
            # Pick k unique rows based on preferential attachment
            selected_rows = np.random.choice(r, size=min(k, r), replace=False, p=probs)
            
            for row_idx in selected_rows:
                rows.append(row_idx)
                cols.append(current_col_idx)
                row_degrees[row_idx] += 1
            
            current_col_idx += 1

    # Construct the final Sparse Matrix A
    matrix_A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, current_col_idx), dtype=np.int8)
    
    return matrix_A, row_degrees

# Example Run
A_sparse, degrees = run_model_1_sparse(steps=2000, alpha=0.1, k=3)
print(f"Model 1 Summary: Rank = {A_sparse.shape[0]}, Dependent Elements = {A_sparse.shape[1]}")