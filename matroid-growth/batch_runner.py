import numpy as np
import matplotlib.pyplot as plt
from matroid_core.simulator import MatroidSimulator
from matroid_core.utils import convert_to_bitsets, check_fano_minor, calculate_binary_rank

def run_beta_sweep():
    print("--- Batch Experiment: Beta Sweep ($\beta$ vs. $F_7$ Probability) ---")
    
    # --- CONTROL KNOBS (Constants) ---
    n_steps = 3000      # Size of each simulation
    gamma = 0.7         # Growth decay
    swaps = 30          # Refinement strength
    iterations = 15     # Number of trials per beta point (for averaging)
    
    # --- THE VARIABLE KNOB (Beta) ---
    beta_values = np.linspace(0.0, 2.0, 11)  # 0.0, 0.2, 0.4 ... 2.0
    
    fano_probabilities = []
    rank_ratios = []

    for b in beta_values:
        print(f"Testing Beta = {b:.1f}...", end=" ")
        fano_count = 0
        ratios = []
        
        for _ in range(iterations):
            # Run simulation
            sim = MatroidSimulator(n_steps=n_steps, gamma=gamma, beta=b, swaps=swaps).run()
            bits = convert_to_bitsets(sim.col_to_rows)
            
            # Check for Fano minor
            if check_fano_minor(bits):
                fano_count += 1
            
            # Calculate Matroid Rank ratio
            rank = calculate_binary_rank(bits, sim.curr_r)
            ratios.append(rank / sim.curr_r)
            
        fano_probabilities.append(fano_count / iterations)
        rank_ratios.append(np.mean(ratios))
        print(f"-> P[Fano] = {fano_count/iterations:.2f}")

    # --- VISUALIZATION ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot 1: Fano Probability (The Phase Transition)
    color = 'tab:red'
    ax1.set_xlabel('Attachment Strength (Beta)')
    ax1.set_ylabel('Probability of Fano Minor', color=color)
    ax1.plot(beta_values, fano_probabilities, 'o-', color=color, linewidth=3, label='P[F7]')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Independence Ratio (The Structural Compression)
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Independence Ratio (Rank / Rows)', color=color)
    ax2.plot(beta_values, rank_ratios, 's--', color=color, alpha=0.6, label='r(M)/n')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Phase Transition in Growing Matroids\n(n={n_steps}, $\gamma$={gamma}, Swaps={swaps})")
    fig.tight_layout()
    
    # Save the plot for the grant proposal
    plt.savefig('experiments/phase_transition_plot.png')
    print("\n[SUCCESS] Phase transition plot saved to experiments/phase_transition_plot.png")
    plt.show()

if __name__ == "__main__":
    run_beta_sweep()