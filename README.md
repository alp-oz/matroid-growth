Matroid Phase Transition Framework

A modular Python suite for simulating the growth of binary matroids and detecting the phase boundary between Graphic (network-representable) and Algebraic (vector-space-only) structures.
Getting Started
1. Installation

Clone the repo and install in editable mode:

git clone https://github.com/alp-oz/matroid-growth.git
cd matroid-growth
python3 -m venv venv
source venv/bin/activate
pip install -e .
2. Running Simulations

To run the primary research suite (the batch runner for N=3000):
run-matroid

To run specific experiment scripts manually:
python3 -m experiments.phase_transition
python3 -m experiments.heatmap
Project Architecture
matroid_core/

    engine.py: The primary growth simulator using preferential attachment and basis expansion logic.

    fields.py: Implementation of Galois Field arithmetic (GF(2) for binary matroids).

    minors.py: Definitions of fundamental forbidden minors (F7, W3, U2,4).

analysis/

    probe_minors.py: Detection of forbidden minors using bitset-based dependency checks.

    circuits.py: Algorithms for identifying minimal dependent sets (circuits).

    connectivity.py: Tools for evaluating the k-connectivity of the generated matroid.

    stats.py: Metric tracking for Zipf distribution and the Banal Effect.

experiments/

    batch_runner.py: High-performance execution of multiple simulations.

    phase_transition.py: 1D parameter sweeps for basis size R and attachment bias Beta.

    heatmap.py: 2D visualization of the algebraic-to-graphic phase boundary.

Scientific Background

This framework probes the Critical Density of random binary matroids.

    The Graphic Phase (M(W3)): At low densities or high discovery rates, the matroid mimics the cycle matroid of a graph. It is characterized by the presence of Wheel minors.

    The Algebraic Phase (F7): As the attachment bias Beta increases, local clusters become dense, inevitably forming the Fano Plane minor. This state is non-graphic and represents purely algebraic linear dependence.

Diagnostic Metrics

We use the following metrics in stats.py to debug Phase Saturation:

    Collision Rate: 1.0 - (Unique Columns/N). If this is high, the engine is clumping vectors too aggressively.

    Row Saturation: The percentage of the basis actually utilized.

    Banal Effect: Local nullity growth over time t, indicating structural dilution.

Troubleshooting the All 1.0 Probability

If your simulations result in P[F7]=1.0 across all parameter ranges, consider these physical limits:

    Preferential Collapse (Beta): At Beta > 1.0, the Rich-Get-Richer effect is hyper-exponential. Try Beta=0.5 to restore sparsity.

    Density Overload (N): If N=3000 and R=50, you are placing 3000 vectors into a 50D space; the Fano Plane becomes statistically certain.

    Discovery Rate (C): If C is too low, the basis does not grow fast enough to accommodate the new columns.