import numpy as np
import matplotlib.pyplot as plt

# --- CORE IMPORTS ---
from matroid_core.engine import MatroidEngine
from analysis.stats import get_zipf_distribution
from analysis.connectivity import get_bipartite_connectivity
from analysis.circuits import get_circuit_participation_by_birth

# =================================================================
# 🎛️ THE RESEARCH KNOBS
# =================================================================
# INSTRUCTIONS for K_SETTING:
# 1. For fixed weight:    Set to an integer (e.g., 4)
# 2. For Poisson weight: Set to ("poisson", lambda) where lambda is the mean
# =================================================================
CONFIG = {
    "N_STEPS": 150,       # Reduced from 5000 (Stop before it clogs)
    "BETA": 0.4,           # Reduced from 0.85 (Less clumping)
    "K_SETTING": 3,        # Fixed at 3 (Sparse "Wheel-like" connections)
    "GAMMA": 0.2,          # Reduced from 0.5 (Keep row discovery active longer)
    "START_R": 300,        # Increased from 100 (More room to breathe)
    "WINDOW_SIZE": 100     # Smaller window for higher resolution
}

def run_research_suite():
    print(f"🚀 Initializing Matroid Growth...")
    print(f"Mode: {CONFIG['K_SETTING']} | Beta: {CONFIG['BETA']}")

    # 1. GENERATE: The Matroid
    engine = MatroidEngine(
        n_steps=CONFIG["N_STEPS"],
        beta=CONFIG["BETA"],
        k_params=CONFIG["K_SETTING"],
        gamma=CONFIG["GAMMA"],
        start_r=CONFIG["START_R"]
    )
    data = engine.run()
    
    # 2. ANALYZE: Statistics & Temporal Trends
    # --- THIS IS THE CRITICAL BLOCK ---
    ranks, counts = get_zipf_distribution(data['row_usage'])
    conn = get_bipartite_connectivity(data)
    
    # This generates the data for your "Banal Column" proof
    temporal_nullity = get_circuit_participation_by_birth(
        data, 
        window_size=CONFIG["WINDOW_SIZE"]
    )
    # ----------------------------------

    print(f"✅ Simulation Complete.")
    print(f"Final Basis (R): {data['R_final']}")
    print(f"Active Connectivity: {conn['active_giant_fraction']:.2%}")
    print(f"Total Components: {conn['total_components']} (Orphans: {conn['orphan_rows']})")

    # 3. RESEARCH-GRADE PLOTTING
    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PLOT A: Zipf Distribution (Degree of Row Usage)
    valid = counts > 0
    ax1.loglog(ranks[valid], counts[valid], 'o', markersize=4, color='#1f77b4', alpha=0.5)
    ax1.set_title(r"$\bf{A.}$ Degree Distribution (Zipf's Law)", fontsize=12)
    ax1.set_xlabel("Row Rank (Log Scale)")
    ax1.set_ylabel("Usage Frequency (Log Scale)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # PLOT B: Structural Dilution (The Banal Effect)
    # This now has access to 'temporal_nullity' defined in Step 2
    times = [t['birth_window'][0] for t in temporal_nullity]
    nulls = [t['nullity'] for t in temporal_nullity]
    
    ax2.plot(times, nulls, '-o', color='#d62728', markersize=5, linewidth=2)
    ax2.fill_between(times, nulls, color='#d62728', alpha=0.1)
    
    ax2.set_title(r"$\bf{B.}$ Structural Dilution (Banal Effect)", fontsize=12)
    ax2.set_xlabel("Column Birth Index (Time $t$)")
    ax2.set_ylabel("Local Nullity (Dependency Density)")
    ax2.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_research_suite()