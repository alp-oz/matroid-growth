# Matroid Growth and the Emergence of Zipf's Law

This project explores how **Scale-Free properties (Zipf's Law)** emerge in matroid structures through iterative growth and basis exchanges.

## 🌀 The Core Concept
While traditional Network Science uses graphs to show the "Rich-Get-Richer" effect, this model uses **Matroid Theory**. By representing the system as a representation matrix $[I | A]$, we observe how complexity redistributes itself through algebraic pivots.

## 🛠 Model Parameters
* **$k$ (Circuit Size):** The number of basis elements a newcomer attaches to. Sampled from a **Truncated Power Law** ($1/k^\alpha$).
* **Basis Exchange:** The mechanism where a dependent element is "promoted" to the basis, triggering a row-XOR operation (Pivot).
* **$\alpha$ (Alpha):** The shape parameter (default 2.0) that regulates the density of the matroid.

## 📊 Key Results
- **Dynamic Basis:** Continuous basis exchange leads to a robust power-law degree distribution.
- **Circuit Explosion:** Even with a fixed input $k$, a single basis exchange can reveal "hidden" global circuits that span the entire rank.

## 🚀 How to Run
```bash
pip install numpy matplotlib
python core_simulation.py# matroid-growth
