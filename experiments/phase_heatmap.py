import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
import seaborn as sns
from pathlib import Path

# Absolute imports to ensure Git structure works
from matroid_core.engine import MatroidEngine
from analysis.probe_minors import (
    convert_to_bitsets, 
    check_fano_minor, 
    check_wheel_3_minor
)

def run_heatmap():
    # 1. ROBUST PATHING
    # This finds the root of your git repo automatically
    base_path = Path(__file__).resolve().parent.parent
    timestamp = time.strftime("%Y%m%d-%H%M")
    run_folder = base_path / "experiments" / f"heatmap_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"--- 🚀 Heatmap Started. Saving to: {run_folder} ---")

    # 2. CONFIG
    BASE_CONFIG = {
        "n_steps": 1500, # Balanced for speed
        "k_params": 4,
        "C": 0.0001,
        "gamma": 0.0,
        "beta": 0.8,
        "start_r": 50
    }

    x_knob, x_values = "start_r", [20, 40, 60, 80, 100]
    y_knob, y_values = "beta", [1.4, 1.1, 0.8, 0.5, 0.2] 
    ITERATIONS = 5 

    fano_grid = np.zeros((len(y_values), len(x_values)))
    wheel_grid = np.zeros((len(y_values), len(x_values)))

    # 3. EXECUTION
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            config = BASE_CONFIG.copy()
            config[x_knob] = int(x_val)
            config[y_knob] = float(y_val)
            
            f_hits, w_hits = 0, 0
            print(f"Processing: Beta={y_val:.1f}, R={x_val}...", end="\r")
            
            for _ in range(ITERATIONS):
                engine = MatroidEngine(**config)
                data = engine.run()
                bits = convert_to_bitsets(data['columns'])
                if check_fano_minor(bits): f_hits += 1
                if check_wheel_3_minor(bits): w_hits += 1
            
            fano_grid[i, j] = f_hits / ITERATIONS
            wheel_grid[i, j] = w_hits / ITERATIONS

    # 4. ROBUST PLOTTING
    print("\n--- 📊 Finalizing Visuals ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(fano_grid, annot=True, fmt=".2f", xticklabels=x_values, 
                yticklabels=y_values, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=1)
    axes[0].set_title("P(Fano F7 Minor)")

    sns.heatmap(wheel_grid, annot=True, fmt=".2f", xticklabels=x_values, 
                yticklabels=y_values, ax=axes[1], cmap="YlGnBu", vmin=0, vmax=1)
    axes[1].set_title("P(Wheel W3 Minor)")

    plt.tight_layout()
    
    # Force save before show
    save_path = run_folder / "heatmap_comparison.png"
    plt.savefig(str(save_path), dpi=300)
    
    # Save a JSON manifest for Git tracking
    with open(run_folder / "results.json", "w") as f:
        json.dump({"fano": fano_grid.tolist(), "config": BASE_CONFIG}, f)

    print(f"✅ Saved successfully: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_heatmap()