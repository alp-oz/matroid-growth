# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from collections import Counter

# ==========================================
# #region MODEL 0: DENSE (INEFFICIENT)
# ==========================================
def run_expanding_model_dense(steps=1000, initial_rank=10, innovation_rate=0.1):
    rank = initial_rank
    matrix_A = np.zeros((rank, 0), dtype=int)
    row_degrees = np.ones(rank)

    for t in range(steps):
        if np.random.rand() < innovation_rate:
            rank += 1
            new_row = np.zeros((1, matrix_A.shape[1]), dtype=int)
            matrix_A = np.vstack([matrix_A, new_row])
            row_degrees = np.append(row_degrees, 1)
        else:
            k = 3
            probs = row_degrees / row_degrees.sum()
            n_to_pick = min(k, rank)
            selected_rows = np.random.choice(range(rank), size=n_to_pick, replace=False, p=probs)
            new_col = np.zeros((rank, 1), dtype=int)
            for r in selected_rows:
                new_col[r, 0] = 1
                row_degrees[r] += 1
            matrix_A = np.hstack([matrix_A, new_col])
    return matrix_A, row_degrees
# #endregion

# ==========================================
# #region MODEL 1: SPARSE (EFFICIENT)
# ==========================================
def run_model_1_sparse(steps=1000, initial_r=10, alpha=0.1, k=3):
    r = initial_r
    rows, cols = [], []
    row_degrees = np.ones(r, dtype=int)
    current_col_idx = 0

    for t in range(steps):
        if np.random.rand() < alpha:
            r += 1
            row_degrees = np.append(row_degrees, 1)
        else:
            probs = row_degrees / row_degrees.sum()
            selected_rows = np.random.choice(r, size=min(k, r), replace=False, p=probs)
            for row_idx in selected_rows:
                rows.append(row_idx)
                cols.append(current_col_idx)
                row_degrees[row_idx] += 1
            current_col_idx += 1

    matrix_A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), 
                                 shape=(r, current_col_idx), dtype=np.int8)
    return matrix_A, row_degrees
# #endregion

# Sparse version of Model 2 with Refinement
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

# ==========================================
# #region MODEL 3: RANDOM BINARY (ERDŐS-RÉNYI)
# ==========================================
def run_random_binary(n_rows, n_cols, total_ones):
    # To compare fairly, we match the number of 1s in the matrix
    p = total_ones / (n_rows * n_cols)
    # Generate random matrix and sum rows to get degrees
    random_matrix = sparse.random(n_rows, n_cols, density=p, format='csr')
    row_degrees = np.array(random_matrix.getnnz(axis=1))
    return random_matrix, row_degrees
# #endregion

# %%
# %%
# ==========================================
# #region MAIN EXECUTION & VISUALIZATION (FINAL)
# ==========================================
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. CONTROLLED PARAMETERS ---
steps = 2000     
alpha = 0.1     
k_val = 3       
swap_rate = 2   

print(f"Starting simulations...")

# --- 2. RUN MODELS (Fixed Unpacking) ---

# Model 1: Pure Expansion 
_, deg_pure = run_model_1_sparse(steps=steps, initial_r=10, alpha=alpha, k=k_val)

# Model 2: Refinement (Swapping)
_, deg_refine = run_model_2_sparse_refinement(steps=steps, initial_r=10, alpha=alpha, k=k_val, swap_rate=swap_rate)

# Model 3: Random Binary 
n_rows = len(deg_pure)
n_cols = int(steps * (1 - alpha))
total_ones = n_cols * k_val
# FIXED: Added _, to unpack the tuple and get the array
_, deg_random = run_random_binary(n_rows, n_cols, total_ones)

# --- 3. VISUALIZATION ---

def get_dist(degrees):
    # Ensure degrees is a flat numpy array
    degrees = np.asarray(degrees).flatten()
    counts = Counter(degrees)
    ks = sorted(counts.keys())
    ps = [counts[k] / len(degrees) for k in ks]
    return ks, ps

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot A: Degree Distribution (Log-Log)
# ------------------------------------
for d, label, color, marker in zip(
    [deg_pure, deg_refine, deg_random], 
    ['Pure Expansion', 'Refinement (Swap)', 'Random Binary'],
    ['blue', 'green', 'red'], 
    ['o', 's', '^']
):
    k, p = get_dist(d)
    ax1.loglog(k, p, marker, label=label, color=color, alpha=0.6, markersize=8)

ax1.set_title("Degree Distribution (Log-Log Scale)")
ax1.set_xlabel("Degree $k$")
ax1.set_ylabel("$P(k)$")
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Plot B: Arrival Order (Log Y-Axis)
# ------------------------------------
indices_pure = range(len(deg_pure))
indices_refine = range(len(deg_refine))
indices_random = range(len(deg_random))

ax2.scatter(indices_pure, deg_pure, color='blue', s=10, alpha=0.4, label='Pure')
ax2.scatter(indices_refine, deg_refine, color='green', s=10, alpha=0.4, label='Refinement')
ax2.scatter(indices_random, deg_random, color='red', s=10, alpha=0.1, label='Random')

# CHANGED: Setting Y-axis to Logarithmic to see the "head" and "tail" clearly
ax2.set_yscale('log')

ax2.set_title("Degree vs. Arrival Order (Log Y-Axis)")
ax2.set_xlabel("Row Index (Arrival Order)")
ax2.set_ylabel("Final Degree (log scale)")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
# #endregion


# %%

# ==========================================
# #region MAIN EXECUTION & VISUALIZATION (WITH COLUMNS)
# ==========================================
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. CONTROLLED PARAMETERS ---
steps = 2000     
alpha = 0.1     
k_val = 3       
swap_rate = 2   

print("Running simulations...")

# --- 2. RUN MODELS (Keeping Matrices for Column Analysis) ---

# Model 1: Pure Expansion
A_pure, deg_pure_rows = run_model_1_sparse(steps=steps, alpha=alpha, k=k_val)

# Model 2: Refinement
A_refine, deg_refine_rows = run_model_2_sparse_refinement(steps=steps, alpha=alpha, k=k_val, swap_rate=swap_rate)

# Model 3: Random Binary
n_rows = len(deg_pure_rows)
n_cols = A_pure.shape[1]
total_ones = n_cols * k_val
A_rand, deg_random_rows = run_random_binary(n_rows, n_cols, total_ones)

# --- 3. EXTRACT COLUMN DEGREES ---
# We sum along axis 0 to see how many rows each column is connected to
deg_pure_cols = np.array(A_pure.sum(axis=0)).flatten()
deg_refine_cols = np.array(A_refine.sum(axis=0)).flatten()
deg_rand_cols = np.array(A_rand.sum(axis=0)).flatten()

# --- 4. VISUALIZATION (3 PANELS) ---

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

def get_dist(degrees):
    degrees = np.asarray(degrees).flatten()
    counts = Counter(degrees)
    ks = sorted(counts.keys())
    ps = [counts[k] / len(degrees) for k in ks]
    return ks, ps

# PLOT 1: Row Degree Distribution (Log-Log)
# ------------------------------------
for d, label, color, marker in zip(
    [deg_pure_rows, deg_refine_rows, deg_random_rows], 
    ['Pure Expansion', 'Refinement', 'Random'],
    ['blue', 'green', 'red'], ['o', 's', '^']
):
    k, p = get_dist(d)
    ax1.loglog(k, p, marker, label=label, color=color, alpha=0.6)
ax1.set_title("Row Degrees (Basis Popularity)")
ax1.set_xlabel("Degree $k$")
ax1.set_ylabel("$P(k)$")
ax1.legend()
ax1.grid(True, which="both", alpha=0.2)

# PLOT 2: Row Arrival Order (Log Y)
# ------------------------------------
ax2.scatter(range(len(deg_pure_rows)), deg_pure_rows, color='blue', s=8, alpha=0.3)
ax2.scatter(range(len(deg_refine_rows)), deg_refine_rows, color='green', s=8, alpha=0.3)
ax2.set_yscale('log')
ax2.set_title("Row Degree vs. Arrival Time")
ax2.set_xlabel("Row Index")
ax2.set_ylabel("Degree (Log)")
ax2.grid(True, alpha=0.3)

# PLOT 3: Column Degree Distribution (The New One!)
# ------------------------------------
# Using a histogram here because your model columns are mostly constant
bins = np.arange(0, max(deg_rand_cols) + 2) - 0.5
ax3.hist(deg_pure_cols, bins=bins, color='blue', alpha=0.5, label='Pure/Refine', density=True)
ax3.hist(deg_rand_cols, bins=bins, color='red', alpha=0.4, label='Random', density=True)
ax3.set_title("Column Degrees (Dependency Complexity)")
ax3.set_xlabel("Number of Rows per Column")
ax3.set_ylabel("Frequency")
ax3.legend()

plt.tight_layout()
plt.show()
# #endregion

