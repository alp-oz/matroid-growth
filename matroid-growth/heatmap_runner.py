import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # You may need: pip install seaborn
from matroid_core.simulator import MatroidSimulator
from matroid_core.utils import convert_to_bitsets, check_fano_minor

def run_2d_sweep():
    print("--- 2D Phase Frontier: Gamma vs. Beta ---")
    
    # Grid Setup
    gammas = np.linspace(0.1, 0.9, 5)   # Y-axis: Growth Decay
    betas = np.linspace(0.0, 2.0, 5)    # X-axis: Attachment
    iterations = 5                     # Trials per cell (increase for smoother grant data)
    
    results = np.zeros((len(gammas), len(betas)))

    for i, g in enumerate(gammas):
        for j, b in enumerate(betas):
            print(f"Testing Gamma={g:.1f}, Beta={b:.1f}...", end="\r")
            fano_count = 0
            for _ in range(iterations):
                sim = MatroidSimulator(n_steps=2000, gamma=g, beta=b, swaps=20).run()
                if check_fano_minor(convert_to_bitsets(sim.col_to_rows)):
                    fano_count += 1
            results[i, j] = fano_count / iterations

    # Plotting the Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, annot=True, xticklabels=np.round(betas, 1), 
                yticklabels=np.round(gammas, 1), cmap="YlOrRd")
    
    plt.title("Fano Minor Existence Probability: Phase Frontier")
    plt.xlabel("Attachment Strength (Beta)")
    plt.ylabel("Growth Decay (Gamma)")
    
    plt.savefig('experiments/fano_heatmap.png')
    print("\n[SUCCESS] Heatmap saved to experiments/fano_heatmap.png")
    plt.show()

if __name__ == "__main__":
    run_2d_sweep()