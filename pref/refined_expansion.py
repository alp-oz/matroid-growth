# %% Dependent elements alter dependencies: "Rich-get-Richer" + "Weak-get-Weaker"
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph, hstack, vstack, csr_matrix

# STANDARD Refinement model (SLOW for large scales)

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

# %% Sparse version of Model 2 with Refinement
# ==========================================
# #region 2: REFINEMENT (SPARSE SWAPPING)
# ==========================================
def run_model_2_sparse_refinement(steps=1000, initial_r=10, alpha=0.1, k=3, swap_rate=1):
    """
    Model 2 using LIL matrix for efficient edge swapping.
    """
    r = initial_r
    # Initialize as a List-of-Lists sparse matrix
    # We start with 0 columns and grow it
    matrix_A = sparse.lil_matrix((r, 0), dtype=np.int8)
    row_degrees = np.ones(r, dtype=int)
    curr_col_idx = 0

    for t in range(steps):
        # --- 1. GROWTH PHASE ---
        if np.random.rand() < alpha:
            r += 1
            # Add a new row to the LIL matrix
            matrix_A.resize((r, matrix_A.shape[1]))
            row_degrees = np.append(row_degrees, 1)
        else:
            # Preferential attachment for new column
            probs = row_degrees / row_degrees.sum()
            sel = np.random.choice(r, size=min(k, r), replace=False, p=probs)
            
            # Expand columns by 1
            matrix_A.resize((r, curr_col_idx + 1))
            for row_idx in sel:
                matrix_A[row_idx, curr_col_idx] = 1
                row_degrees[row_idx] += 1
            curr_col_idx += 1

        # --- 2. REFINEMENT PHASE (Sparse Swap) ---
        if curr_col_idx > 0:
            for _ in range(swap_rate):
                col_to_fix = np.random.randint(curr_col_idx)
                
                # Get indices of rows that have a '1' in this column
                # In LIL format, matrix_A.T.rows[col_to_fix] gives this directly
                ones_in_col = matrix_A.getcol(col_to_fix).nonzero()[0]
                
                if len(ones_in_col) > 0:
                    # Find weakest row
                    degs_in_col = row_degrees[ones_in_col]
                    weakest_row = ones_in_col[np.argmin(degs_in_col)]
                    
                    # Pick a hub to move to
                    probs = row_degrees / row_degrees.sum()
                    target_row = np.random.choice(r, p=probs)
                    
                    # Swap logic
                    if target_row != weakest_row and matrix_A[target_row, col_to_fix] == 0:
                        matrix_A[weakest_row, col_to_fix] = 0
                        matrix_A[target_row, col_to_fix] = 1
                        
                        row_degrees[weakest_row] -= 1
                        row_degrees[target_row] += 1

    return matrix_A.tocsr(), row_degrees
# #endregion

# %%
# --- 1. Run the Refinement Model ---
# We use a higher swap_rate to see the 'Rich-Get-Richer' effect in action
matrix_A2, degrees2 = run_model_2_sparse_refinement(
    steps=10000, 
    initial_r=10, 
    alpha=0.01, 
    k=3, 
    swap_rate=2
)

# --- 2. The Connectivity Test ---
from scipy.sparse import csgraph, hstack, vstack, csr_matrix

def is_connected_sparse(A):
    r, n = A.shape
    if r == 0 or n == 0: return False
    
    # Construct the Bipartite Incidence Graph
    # Rows and Columns are nodes; A_ij = 1 is an edge.
    top = hstack([csr_matrix((r, r)), A])
    bottom = hstack([A.T, csr_matrix((n, n))])
    bipartite_adj = vstack([top, bottom])
    
    n_components, labels = csgraph.connected_components(bipartite_adj, directed=False)
    return n_components, n_components == 1

n_comps, connected = is_connected_sparse(matrix_A2)

print(f"Model 2 Results:")
print(f"Components: {n_comps}")
print(f"Connected: {connected}")
print(f"Max Row Degree: {degrees2.max()} (The 'Super-Hub')")