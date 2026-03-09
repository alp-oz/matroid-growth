"""
MATROID PHASE TRANSITION PROBE
------------------------------
Consolidated tool to sweep parameters and detect the transition 
between Algebraic (Fano-rich) and Graphic (Wheel-rich) states.

KEY IMPORTS:
- check_fano_minor: Detects the F7 non-graphic minor.
- check_wheel_3_minor: Detects the W3 (K4 graph) minor.
- calculate_binary_rank: Monitors the actual basis size growth.

1. SELECT KNOB: Change 'knob_name' to "start_r", "beta", or "C".
2. K_PARAMS: 
   - Use an integer (e.g., 4) for fixed column size.
   - Use ("poisson", 4) for variable column size.
3. OUTPUT: Results are saved in timestamped sub-folders in /experiments/.
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from matroid_core.engine import MatroidEngine
from analysis.probe_minors import (
    convert_to_bitsets, 
    check_fano_minor, 
    check_wheel_3_minor, 
    calculate_binary_rank
)

def run_sweep():
    # --- 1. CONFIGURATION (Golden Values adjusted for visibility) ---
    BASE_CONFIG = {
        "n_steps": 1000,        
        "k_params": 4,          # Switch to ("poisson", 4) for variability
        "C": 0.0001,             # Increased from 0.0001 to allow row growth
        "gamma": 0,
        "beta": 0.8,            # Slightly lower beta to reduce hyper-clustering
        "start_r": 50
    }

    # --- 2. SELECT THE KNOB ---
    knob_name = "start_r" 
    # Expanded range to capture the transition: 
    # High (7-35) -> Low (50) -> High (100+)
    knob_values = [7, 18, 35, 50, 75, 100, 150]

    ITERATIONS = 15 # Increased for statistical validity
    fano_probs, wheel_probs, avg_ranks = [], [], []

    # --- 3. DIRECTORY & NAMING ---
    SHORT_CODES = {"start_r": "sr", "beta": "bt", "C": "cc", "n_steps": "ns"}
    code = SHORT_CODES.get(knob_name, knob_name[:2])
    date_str, time_str = time.strftime("%Y%m%d"), time.strftime("%H%M")
    
    run_folder = os.path.join("experiments", f"ex_{code}_{time_str}")
    os.makedirs(run_folder, exist_ok=True)
    base_filename = f"{code}_{date_str}_{time_str}"

    print(f"--- 🚀 Sweep: {knob_name} | Folder: {run_folder} ---")

    for val in knob_values:
        config = BASE_CONFIG.copy()
        config[knob_name] = val
        
        print(f"Testing {knob_name}={val}...", end=" ", flush=True)
        f_hits, w_hits, iteration_ranks = 0, 0, []
        
        for _ in range(ITERATIONS):
            engine = MatroidEngine(**config)
            data = engine.run()
            
            # --- DIAGNOSTIC BLOCK ---
            # Verifying if columns are "collapsing" into a dense subspace
            unique_cols = len(set(tuple(c) for c in data['columns']))
            bits = convert_to_bitsets(data['columns'])
            current_rank = calculate_binary_rank(bits)
            # ------------------------
            
            # Metric Tracking
            if check_fano_minor(bits): f_hits += 1
            if check_wheel_3_minor(bits): w_hits += 1
            iteration_ranks.append(current_rank)
        
        fano_probs.append(f_hits / ITERATIONS)
        wheel_probs.append(w_hits / ITERATIONS)
        avg_ranks.append(float(np.mean(iteration_ranks)))
        
        print(f"P[F7]={fano_probs[-1]:.2f} | P[W3]={wheel_probs[-1]:.2f} | AvgRank={avg_ranks[-1]:.1f}")

    # --- 4. PLOTTING & METADATA ---
    plt.figure(figsize=(10, 6))
    plt.plot(knob_values, fano_probs, 'o-', label="Fano (F7)", color='#e67e22', linewidth=2)
    plt.plot(knob_values, wheel_probs, 's--', label="Wheel (W3)", color='#2980b9', alpha=0.7)
    
    plt.title(f"Matroid Phase Transition: {knob_name}")
    plt.xlabel(f"Variable: {knob_name}")
    plt.ylabel("Minor Discovery Probability")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save output
    plt.savefig(os.path.join(run_folder, f"{base_filename}.png"))
    
    metadata = {
        "timestamp": f"{date_str}-{time_str}",
        "config": BASE_CONFIG,
        "knob": knob_name,
        "values": list(knob_values),
        "fano_probs": fano_probs,
        "wheel_probs": wheel_probs,
        "avg_ranks": avg_ranks
    }
    with open(os.path.join(run_folder, f"{base_filename}.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Sweep Finished. Result: {run_folder}/{base_filename}.png")
    plt.show()

if __name__ == "__main__":
    run_sweep()  # <--- This MUST be indented (4 spaces)