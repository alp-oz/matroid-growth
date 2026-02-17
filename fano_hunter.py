# %%
import numpy as np
import itertools

def find_fano_minor(matrix_A):
    """
    Scans the matrix A to see if any 3 rows and 4 columns 
    form the non-identity part of the Fano Plane.
    """
    n_rows, n_cols = matrix_A.shape
    if n_rows < 3 or n_cols < 4:
        return None

    # Fano (F7) non-identity part is a 3x4 matrix containing:
    # [1, 1, 0, 1]
    # [1, 0, 1, 1]
    # [0, 1, 1, 1]
    # (or any permutation of these columns)
    fano_target = {tuple([1,1,0]), tuple([1,0,1]), tuple([0,1,1]), tuple([1,1,1])}

    # To keep it fast, we only check the most 'popular' rows (hubs)
    # as they are the most likely to participate in a minor.
    row_sums = np.sum(matrix_A, axis=1)
    top_rows = np.argsort(row_sums)[-10:] # Check top 10 hubs
    
    for row_indices in itertools.combinations(top_rows, 3):
        submatrix = matrix_A[list(row_indices), :]
        # Find unique columns in this 3-row submatrix
        unique_cols = set()
        for col_idx in range(n_cols):
            col = tuple(submatrix[:, col_idx])
            if col in fano_target:
                unique_cols.add(col)
        
        if unique_cols == fano_target:
            return row_indices # Found it!

    return None

def run_model_1_with_fano(steps=500, initial_rank=10, innovation_rate=0.1, k=3):
    rank = initial_rank
    matrix_A = np.zeros((rank, 0), dtype=int)
    row_degrees = np.ones(rank)

    for t in range(steps):
        if np.random.rand() < innovation_rate:
            rank += 1
            matrix_A = np.vstack([matrix_A, np.zeros((1, matrix_A.shape[1]), dtype=int)])
            row_degrees = np.append(row_degrees, 1)
        else:
            probs = row_degrees / row_degrees.sum()
            selected_rows = np.random.choice(range(rank), size=min(k, rank), replace=False, p=probs)
            
            new_col = np.zeros((rank, 1), dtype=int)
            for r in selected_rows:
                new_col[r, 0] = 1
                row_degrees[r] += 1
            matrix_A = np.hstack([matrix_A, new_col])

    # Hunting for the Fano Minor
    fano_rows = find_fano_minor(matrix_A)
    
    return matrix_A, row_degrees, fano_rows

# --- Execution ---
A, degrees, fano_location = run_model_1_with_fano(steps=1000, k=3)

print(f"Final Matrix Size: {A.shape}")
if fano_location:
    print(f"🎯 Fano Minor found! It involves Basis Elements (Rows): {fano_location}")
else:
    print("❌ No Fano Minor detected in the top hubs.")