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


Earlier models

Documentation: Phase 1 – The Greedy Baseline

Model: Edmonds’ Greedy Algorithm (1971)

Matroid Type: Binary Linear Matroid (GF(2))
1. Mathematical Objective

The goal is to find a Maximum Weight Independent Set (a Basis B) within a ground set E of n binary vectors. In matroid theory, the greedy algorithm is guaranteed to find the optimal solution for a static set of weighted elements.
2. The Process

The model follows a strict three-step lifecycle:

    Generation: A Ground Set E is created using n random bitstrings of a fixed Dimension. This dimension defines the maximum possible rank (r).

    Prioritization: Elements are sorted in descending order of their weights.

    Sequential Selection: The algorithm iterates through the sorted list exactly n times. For each element, it performs an Independence Test (XOR-based Gaussian Elimination). If the element is linearly independent of the current Basis B, it is permanently added.

3. Computational Complexity

While the algorithm processes n elements, the total running time is O(n⋅r2).

    n: The number of Oracle queries (iterations).

    r2: The cost of performing Gaussian elimination over the binary field for each query.

4. The "Greedy" Limitation

This model is static. Once an element is accepted into the Basis, it is never removed. If a high-weight element appears late in the sequence and the Basis is already at full rank, the element is rejected—even if it is more valuable than existing members. This limitation serves as the primary motivation for the Late Exchange Logic developed in Phase 3 of this project.

Phase 2: Lawler’s Circuit Mapping (1975)

File: lawler_circuit_map.py

Mathematical Concept: The Fundamental Circuit C(B,y)

While the Greedy model acts as a binary gatekeeper, the Circuit Map identifies the specific internal conflicts within the matroid. This phase transitions the project from simple filtering to structural analysis.
1. Identifying the "Blockage"

In a binary matroid, if an element y is dependent on the Basis B, there exists a unique minimal set of vectors in B that XOR-sum to y. This set is the Fundamental Circuit.

    The Logic: We solve the linear system B⋅c=y(mod2).

    The Result: The indices where c=1 are the "witnesses" that prove y is redundant.

2. The Exchange Property

According to the Strong Basis Exchange Property, any element x in the circuit C(B,y) is a valid candidate for eviction.

    Theorem: Replacing any x∈C(B,y)∖{y} with y maintains the rank r and results in a new valid Basis B′.

3. Transition to Late Exchange

This circuit mapping is the "engine" for our final model. It allows the algorithm to be selective:

    Greedy: "The basis is full, reject y."

    Lawler: "The basis is full because of elements {x1​,x2​,x3​}. If y is more valuable than any of them, we can swap."

The Late Exchange Algorithm (Dynamic)

File: late_exchange_model.py

Core Concept: Online Weight Maximization via Circuit Augmentation

This model represents the final transition from a static selection process to a Dynamic State-Machine. Unlike the previous models, this algorithm does not require a pre-sorted ground set and handles elements in a live "stream."
1. Mathematical Formalization

    Initial State: The Basis B0​=∅ with total weight W0​=0.

    The Leaderboard Effect: The Dimension (d) acts as a fixed capacity constraint. Once ∣B∣=d, the Basis enters a competitive state.

    Weight (w): Acts as the Selection Pressure. It is the metric used to resolve conflicts within a circuit.

2. Transition Rules

For every arriving element (et​,wt​), the algorithm applies one of two rules:

    Rule I: Augmentation If the element is linearly independent, it is immediately added to the Basis.
    Bt​=Bt−1​∪{et​}

    Rule II: Exchange If the element creates a dependency, the unique Fundamental Circuit C(Bt−1​,et​) is identified. We then find the "weakest link" x∗ in that circuit:
    x∗=argminx∈C∖{et​}​w(x)

    If w(et​)>w(x∗), we perform a swap. The Basis "evolves" by evicting x∗ to make room for the higher-weight et​.

3. Why this is "Dynamic"

This model is self-correcting. In the Edmonds (1971) model, a low-weight element accepted early could block a high-weight element later. In this model, the Late Exchange logic ensures that the Basis always contains the "optimal" set of vectors for the prefix of the stream seen so far.