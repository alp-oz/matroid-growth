# %%
import numpy as np
import matplotlib.pyplot as plt

def run_main_growth_model(steps=3000, initial_rank=50, fixed_k=3):
    rank = initial_rank
    # matrix_A represents the relationship between Basis and Dependent elements
    matrix_A = np.zeros((rank, steps), dtype=int)
    basis_degrees = np.ones(rank)
    
    # Track the "Current Basis" - initially elements 0 to rank-1
    # This helps us track the identity of elements as they swap in and out
    current_basis_ids = list(range(rank))
    
    # 1. GROWTH WITH CONTINUOUS BASIS EXCHANGE
    for n in range(steps):
        # --- A. Standard Attachment ---
        prob = basis_degrees / basis_degrees.sum()
        selected_rows = np.random.choice(range(rank), size=fixed_k, replace=False, p=prob)
        
        for r in selected_rows:
            matrix_A[r, n] = 1
            # Note: basis_degrees[r] will be recalculated after pivot
            
        # --- B. THE EXCHANGE (Pivot) ---
        # The new element (column n) enters the basis.
        # It replaces the most "burdened" basis element it is connected to.
        pivot_row = selected_rows[np.argmax(basis_degrees[selected_rows])]
        
        # Binary Row Operations: XOR the pivot row into all other rows that use this element
        pivot_row_content = matrix_A[pivot_row, :].copy()
        for r in range(rank):
            if r != pivot_row and matrix_A[r, n] == 1:
                matrix_A[r, :] = np.bitwise_xor(matrix_A[r, :], pivot_row_content)
        
        # Update Degrees after the structure change
        basis_degrees = np.sum(matrix_A, axis=1) + 1
            
    # Calculate degrees for all elements
    dep_degrees = np.sum(matrix_A, axis=0) + 1 

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Basis Degree Distribution (Histogram)
    ax1.hist(basis_degrees, bins=20, color='orange', alpha=0.7)
    ax1.set_title(f"Dynamic Basis Degree (Fixed k={fixed_k})")
    ax1.set_xlabel("Degree")

    # Plot 2: Zipf's Law (Log-Log)
    all_degrees = np.sort(np.concatenate([basis_degrees, dep_degrees]))[::-1]
    ranks = np.arange(1, len(all_degrees) + 1)
    ax2.loglog(ranks, all_degrees, 'ro', markersize=3, alpha=0.5)
    
    # Add a reference line for Power Law (slope -1)
    ax2.loglog(ranks, all_degrees[0] / ranks, 'k--', alpha=0.3, label="Ideal Zipf (Slope -1)")
    
    ax2.set_title("Zipf's Law Plot (Continuous Exchange)")
    ax2.set_ylabel("Degree (Log)")
    ax2.set_xlabel("Rank (Log)")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

run_main_growth_model()
# %%