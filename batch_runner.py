import numpy as np
import matplotlib.pyplot as plt

# --- CORE IMPORTS ---
# These now utilize the formalized package structure you created
from matroid_core.engine import MatroidEngine
from analysis.stats import get_zipf_distribution
from analysis.connectivity import get_bipartite_connectivity
from analysis.circuits import estimate_girth, get_circuit_participation_by_birth

# =================================================================
# 🎛️ THE RESEARCH KNOBS
# =================================================================
CONFIG = {
    "N_STEPS": 3000,       
    "BETA": 0.8,           
    
    # K_SETTING determines how many existing rows a new column connects to.
    # 1. FIXED: Set to an integer (e.g., 3) to choose exactly K rows.
    # 2. RANDOM: Set to ("poisson", lambda) to pick K from a Poisson distribution.
    "K_SETTING": 3,        
    
    "GAMMA": 0.05,          
    "START_R": 50,        
    "WINDOW_SIZE": 100     
}

def run_research_suite():
    print(f"🚀 Initializing Matroid Growth...")
    print(f"Mode: {CONFIG['K_SETTING']} | Beta: {CONFIG['BETA']}")

    # 1. GENERATE: The Matroid engine handles the preferential attachment logic
    # Updated to ensure C is passed (defaulting to 0.0001 if not in CONFIG)
    engine = MatroidEngine(
        n_steps=CONFIG["N_STEPS"],
        beta=CONFIG["BETA"],
        k_params=CONFIG["K_SETTING"],
        gamma=CONFIG["GAMMA"],
        start_r=CONFIG["START_R"],
        C=CONFIG.get("C", 0.0001) 
    )
    data = engine.run()
    
    # 2. ANALYZE: Topological and Statistical properties
    # 2a. The 'row_usage' key is now safely returned by the updated engine
    ranks, counts = get_zipf_distribution(data['row_usage'])
    conn = get_bipartite_connectivity(data)

    # 2b. Girth Estimation
    girth = estimate_girth(data['columns'], data['R_final'])

    # Measuring the 'Banal Effect': how dependencies grow over time
    temporal_nullity = get_circuit_participation_by_birth(
        data, 
        window_size=CONFIG["WINDOW_SIZE"]
    )

    print(f"\n✅ Simulation Complete.")
    print(f"Final Basis (R): {data['R_final']}")
    print(f"Estimated Girth: {girth if girth else '8+'}") 
    print(f"Active Connectivity: {conn['active_giant_fraction']:.2%}")
    print(f"Total Components: {conn['total_components']} (Orphans: {conn['orphan_rows']})")

    # 3. PLOTTING
    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PLOT A: Zipf Distribution
    valid = counts > 0
    ax1.loglog(ranks[valid], counts[valid], 'o', markersize=4, color='#1f77b4', alpha=0.5)
    ax1.set_title(r"$\bf{A.}$ Degree Distribution (Zipf's Law)", fontsize=12)
    ax1.set_xlabel("Row Rank (Log Scale)")
    ax1.set_ylabel("Usage Frequency (Log Scale)")
    
    # PLOT B: Structural Dilution (The Banal Effect)
    times = [t['birth_window'][0] for t in temporal_nullity]
    nulls = [t['nullity'] for t in temporal_nullity]
    
    ax2.plot(times, nulls, '-o', color='#d62728', markersize=5, linewidth=2)
    ax2.fill_between(times, nulls, color='#d62728', alpha=0.1)
    ax2.set_title(r"$\bf{B.}$ Structural Dilution (Banal Effect)", fontsize=12)
    ax2.set_xlabel("Column Birth Index (Time $t$)")
    ax2.set_ylabel("Local Nullity (Dependency Density)")

    plt.tight_layout()
    plt.show()

def main():
    run_research_suite()

if __name__ == "__main__":
    main()