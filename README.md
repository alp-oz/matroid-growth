Matroid Phase Transition Framework

A modular Python suite for simulating the growth of binary matroids and detecting the phase boundary between Graphic (network-representable) and Algebraic (vector-space-only) structures.
📂 Project Architecture
matroid_core/

    engine.py: The primary growth simulator using preferential attachment and basis expansion logic.

    fields.py: Implementation of Galois Field arithmetic (primarily GF(2) for binary matroids).

    minors.py: Definitions of fundamental forbidden minors (F7​, W3​, U2,4​).

analysis/

    probe_minors.py: Detection of forbidden minors using bitset-based dependency checks.

    circuits.py: Algorithms for identifying minimal dependent sets (circuits).

    connectivity.py: Tools for evaluating the k-connectivity of the generated matroid.

    stats.py: Metric tracking for Collision Rate, Row Saturation, and Clustering Coefficients.

    quantum.py: (Experimental) Analysis of matroid structures in relation to quantum stabilizer codes.

experiments/

    phase_transition.py: 1D parameter sweeps for basis size R and attachment bias β.

    heatmap.py: 2D visualization of the algebraic-to-graphic phase boundary.

    batch_runner.py: High-performance execution of multiple simulations using Python's multiprocessing.

🧬 Scientific Background

This framework probes the "Critical Density" of random binary matroids.

    The Graphic Phase (M(W3​)): At low densities or high discovery rates, the matroid mimics the cycle matroid of a graph. It is characterized by the presence of "Wheel" minors.

    The Algebraic Phase (F7​): As the attachment bias β increases, local clusters become dense, inevitably forming the Fano Plane minor. This state is non-graphic and represents purely algebraic linear dependence.

🚀 Getting Started
1. Installation
Bash

git clone <your-repo-url>
cd matroid-growth
pip install -r requirements.txt

2. Running a Phase Sweep

To visualize the 1D transition:
Bash

python3 -m experiments.phase_transition

3. Generating a 2D Heatmap

To map the boundary between R and β:
Bash

python3 -m experiments.heatmap

📊 Diagnostic Metrics

We use the following metrics in stats.py to debug "Phase Saturation" (where P[F7​] is stuck at 1.0):

    Collision Rate: 1.0−(Unique Columns/N). If this is high, the engine is "clumping" vectors too aggressively.

    Row Saturation: The percentage of the basis actually utilized. Low saturation means the simulation is effectively running in a much smaller subspace than intended.

🛠 Troubleshooting the "All 1.0" Probability

If your simulations result in P[F7​]=1.0 across all parameter ranges, consider the following physical limits:

    Preferential Collapse (β): At β>1.0, the "Rich-Get-Richer" effect is hyper-exponential. The engine will pick the same 4–10 rows regardless of how many you add. Try β=0.5 to restore sparsity.

    Density Overload (N): If N=3000 and R=50, you are placing 3000 vectors into a 50-dimensional space. Mathematically, the Fano Plane becomes almost certain. Reduce N to 1000 or increase R to 150.

    Discovery Rate (C): If C is too low, the basis does not grow fast enough to accommodate the new columns.

📜 Git Best Practices

    Avoid committing the results/ or __pycache__ directories.

    Use tags to mark stable mathematical "Golden States":
    git tag -a v1.2 -m "Modular engine with stats integration"
