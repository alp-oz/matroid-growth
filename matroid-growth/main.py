import os
import csv
from datetime import datetime
from matroid_core.simulator import MatroidSimulator
from matroid_core.analysis import plot_zipf, get_connectivity_stats
from matroid_core.utils import convert_to_bitsets, check_fano_minor, check_wheel_minor_w3, calculate_binary_rank

def log_result(params, rank, fano, wheel, giant):
    """Saves experiment results to a CSV for research documentation."""
    os.makedirs('experiments', exist_ok=True)
    log_file = 'experiments/research_log.csv'
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Header for the research log
            writer.writerow(['Timestamp', 'n_steps', 'gamma', 'beta', 'swaps', 'Rank', 'Fano', 'Wheel', 'Giant_Pct'])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            params['n_steps'], params['gamma'], params['beta'], params['swaps'],
            rank, fano, wheel, f"{giant*100:.2f}"
        ])

def main():
    print("\n--- Matroid Growth Research Dashboard ---")
    print("[1] Random (Uniform) | [2] Scale-Free | [3] Refined (Swaps) | [4] Custom")
    choice = input("Select Model: ")
    
    # Preset configurations for reproducibility
    presets = {
        '1': {'n_steps': 5000, 'k': 3, 'C': 0.1, 'gamma': 0.0, 'beta': 0.0, 'swaps': 0},
        '2': {'n_steps': 5000, 'k': 3, 'C': 1.0, 'gamma': 0.7, 'beta': 1.0, 'swaps': 0},
        '3': {'n_steps': 5000, 'k': 3, 'C': 1.0, 'gamma': 0.7, 'beta': 1.0, 'swaps': 50}
    }

    p = presets.get(choice)

    if not p:
        print("\n--- Custom Parameter Entry ---")
        p = {
            'n_steps': int(input("Total steps (n): ")),
            'k': int(input("Column weight (k): ")), 
            'C': float(input("Growth Coeff (C): ")), 
            'gamma': float(input("Decay (gamma): ")),
            'beta': float(input("Attachment (beta): ")), 
            'swaps': int(input("Refinement Swaps: "))
        }

    print("\nSimulating growth process...", end="\r")
    
    # 1. Run the Stochastic Simulator
    sim = MatroidSimulator(**p).run()
    
    # 2. Extract Mathematical Properties
    bits = convert_to_bitsets(sim.col_to_rows)
    matroid_rank = calculate_binary_rank(bits, sim.curr_r)
    fano_exists = check_fano_minor(bits)
    wheel_exists = check_wheel_minor_w3(bits)
    _, giant_size = get_connectivity_stats(sim)

    # 3. Output Results (Compact Version)
    print(f"DONE: Rows={sim.curr_r} | Matroid Rank r(M)={matroid_rank} | Ratio={matroid_rank/sim.curr_r:.3f}")
    print(f"MINORS: Fano={str(fano_exists).upper()} | Wheel={str(wheel_exists).upper()} | Giant={giant_size*100:.1f}%")
    
    # 4. Log to Research Journal
    log_result(p, matroid_rank, fano_exists, wheel_exists, giant_size)
    print(f"-> Results appended to experiments/research_log.csv")
    print("-" * 60)

    # 5. Visualize Degree Distribution
    plot_zipf(sim)

if __name__ == "__main__":
    main()