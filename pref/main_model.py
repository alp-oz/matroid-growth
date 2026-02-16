# %%
import numpy as np
import matplotlib.pyplot as plt

def run_main_growth_model(steps=3000, initial_rank=50, fixed_k=3):
    rank = initial_rank
    matrix_A = np.zeros((rank, steps), dtype=int)
    basis_degrees = np.ones(rank)
    
    # 1. GROWTH ONLY (No Basis Exchange)
    for n in range(steps):
        prob = basis_degrees / basis_degrees.sum()
        selected_rows = np.random.choice(range(rank), size=fixed_k, replace=False, p=prob)
        for r in selected_rows:
            matrix_A[r, n] = 1
            basis_degrees[r] += 1
            
    # Calculate degrees for pure growth
    dep_degrees = np.sum(matrix_A, axis=0) + 1 # Should all be k+1

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Degree Distribution (Histogram)
    ax1.hist(basis_degrees, bins=20, color='blue', alpha=0.7)
    ax1.set_title(f"Basis Degree Distribution (Fixed k={fixed_k})")
    ax1.set_xlabel("Degree")

    # Plot 2: Zipf's Law (Log-Log)
    all_degrees = np.sort(np.concatenate([basis_degrees, dep_degrees]))[::-1]
    ranks = np.arange(1, len(all_degrees) + 1)
    ax2.loglog(ranks, all_degrees, 'bo', markersize=3)
    ax2.set_title("Zipf's Law Plot (Growth Only)")
    ax2.set_ylabel("Degree (Log)")
    ax2.set_xlabel("Rank (Log)")
    
    plt.tight_layout()
    plt.show()

run_main_growth_model()



# %%
