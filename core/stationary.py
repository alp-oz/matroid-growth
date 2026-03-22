import numpy as np
import random
from collections import Counter
from core.engine import MatroidEngine
from core.markov_chain import fundamental_circuits, decompose_into_circuits
from core.circuits import all_circuits


# ─────────────────────────────────────────────────────────────────────────────
# Build exact transition matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_transition_matrix(M, r, circuits, mode):
    """
    Build the exact |circuits| x |circuits| row-stochastic transition matrix.

    Row i = distribution over next states when current state is circuits[i].

    global mode:
        Pick j uniformly from all non-basis elements.
        Decompose C ^ fc[j] into parts; pick one uniformly.
        P[i,k] = sum_j  1/|non_basis| * 1/|parts_j| * 1[k in parts_j]

    adjacent mode:
        Pick j uniformly from {non-basis j : fc[j] ∩ C ≠ ∅}.
        Same decompose-and-pick-uniform step.
        If C ^ fc[j] = ∅ (same circuit), stay at i.
    """
    fc = fundamental_circuits(M, r)
    non_basis = list(range(r, M.shape[1]))
    idx = {c: i for i, c in enumerate(circuits)}   # circuit -> row index
    N = len(circuits)

    P = np.zeros((N, N), dtype=np.float64)

    for i, C in enumerate(circuits):
        if mode == 'global':
            eligible = non_basis
        else:
            eligible = [j for j in non_basis if fc[j] & C]

        n_eligible = len(eligible)
        if n_eligible == 0:
            P[i, i] = 1.0
            continue

        for j in eligible:
            sym_diff = C ^ fc[j]
            if not sym_diff:
                # XOR with identical circuit → stay
                P[i, i] += 1.0 / n_eligible
                continue

            parts = decompose_into_circuits(M, sym_diff)
            if not parts:
                P[i, i] += 1.0 / n_eligible
                continue

            weight = 1.0 / (n_eligible * len(parts))
            for C_next in parts:
                k = idx.get(C_next)
                if k is not None:
                    P[i, k] += weight

    # Normalise rows (should already sum to 1 but guard against float drift)
    row_sums = P.sum(axis=1, keepdims=True)
    P /= row_sums
    return P


# ─────────────────────────────────────────────────────────────────────────────
# Stationary distribution
# ─────────────────────────────────────────────────────────────────────────────

def stationary_distribution(P):
    """
    Compute the stationary distribution π of a row-stochastic matrix P.

    Solves π = π P  (left eigenvector for eigenvalue 1) via:
      (P^T - I) augmented with normalisation constraint.

    Returns a 1-D array π with π[i] >= 0 and sum(π) = 1.
    """
    N = P.shape[0]
    # Solve (P^T - I) π = 0  with  sum(π) = 1
    # Replace last equation with the normalisation constraint
    A = (P.T - np.eye(N)).astype(np.float64)
    A[-1, :] = 1.0
    b = np.zeros(N)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0, None)
    pi /= pi.sum()
    return pi


# ─────────────────────────────────────────────────────────────────────────────
# Compare two distributions
# ─────────────────────────────────────────────────────────────────────────────

def compare_distributions(pi1, pi2, label1="global", label2="adjacent", top_k=20):
    l1   = np.sum(np.abs(pi1 - pi2))
    linf = np.max(np.abs(pi1 - pi2))
    kl   = np.sum(pi1 * np.log((pi1 + 1e-300) / (pi2 + 1e-300)))

    print(f"  L1 distance        : {l1:.6f}")
    print(f"  L∞ distance        : {linf:.6f}")
    print(f"  KL({label1}‖{label2})  : {kl:.6f}")
    print(f"  Same distribution  : {'YES' if l1 < 1e-6 else 'NO'}")

    # Top states by stationary probability
    order = np.argsort(-pi1)
    print(f"\n  {'Rank':>5}  {'π_global':>12}  {'π_adjacent':>12}  {'Ratio':>8}")
    print("  " + "-" * 45)
    for rank, i in enumerate(order[:top_k], 1):
        ratio = pi1[i] / pi2[i] if pi2[i] > 1e-12 else float('inf')
        print(f"  {rank:>5}  {pi1[i]:>12.6f}  {pi2[i]:>12.6f}  {ratio:>8.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Circuit features
# ─────────────────────────────────────────────────────────────────────────────

def circuit_features(M, r, circuits):
    """
    For each circuit compute structural features:

    size                 total number of elements |C|
    n_basis              |C ∩ basis|
    n_nonbasis           |C ∩ non-basis|  (always ≥ 1)
    n_eligible_global    number of fc[j] eligible in global mode (= all non-basis)
    n_eligible_adjacent  number of fc[j] sharing ≥1 element with C
    avg_decomp_global    average |decomp(C ^ fc[j])| over all j
    avg_decomp_adjacent  average |decomp(C ^ fc[j])| over eligible j
    degree_global        number of distinct circuits reachable in one global step
    degree_adjacent      number of distinct circuits reachable in one adjacent step

    Returns a dict feature_name -> np.array of length |circuits|.
    """
    fc = fundamental_circuits(M, r)
    non_basis = list(range(r, M.shape[1]))
    n_nb = len(non_basis)

    feats = {k: np.zeros(len(circuits)) for k in [
        "size", "n_basis", "n_nonbasis",
        "n_eligible_adjacent",
        "avg_decomp_global", "avg_decomp_adjacent",
        "degree_global", "degree_adjacent",
    ]}

    for i, C in enumerate(circuits):
        feats["size"][i]      = len(C)
        feats["n_basis"][i]   = sum(1 for e in C if e < r)
        feats["n_nonbasis"][i] = sum(1 for e in C if e >= r)

        eligible_adj = [j for j in non_basis if fc[j] & C]
        feats["n_eligible_adjacent"][i] = len(eligible_adj)

        reach_global   = set()
        reach_adjacent = set()
        decomp_sizes_global   = []
        decomp_sizes_adjacent = []

        for j in non_basis:
            sym_diff = C ^ fc[j]
            if not sym_diff:
                parts = [C]
            else:
                parts = decompose_into_circuits(M, sym_diff)

            decomp_sizes_global.append(len(parts))
            reach_global.update(parts)

            if j in eligible_adj:
                decomp_sizes_adjacent.append(len(parts))
                reach_adjacent.update(parts)

        feats["avg_decomp_global"][i]   = np.mean(decomp_sizes_global)
        feats["avg_decomp_adjacent"][i] = (np.mean(decomp_sizes_adjacent)
                                            if decomp_sizes_adjacent else 0)
        feats["degree_global"][i]   = len(reach_global)
        feats["degree_adjacent"][i] = len(reach_adjacent)

    return feats


def correlate_with_stationary(feats, pi, mode_label):
    """Pearson and Spearman correlations of each feature with π."""
    from scipy.stats import spearmanr, pearsonr

    print(f"\n  Correlations with π ({mode_label})")
    print(f"  {'Feature':<28}  {'Pearson r':>10}  {'Spearman ρ':>10}")
    print("  " + "-" * 52)
    for name, vals in feats.items():
        pr, _ = pearsonr(vals, pi)
        sr, _ = spearmanr(vals, pi)
        print(f"  {name:<28}  {pr:>10.4f}  {sr:>10.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    engine = MatroidEngine(n_steps=30, k_params=2, C=0.1, beta=0.8, start_r=2)
    result = engine.run()
    M, r, n = result["M"], result["r"], result["n"]
    print(f"Matroid: rank={r}, #elements={n}, #non-basis={n - r}")

    # Get full circuit set
    print("Finding all circuits...")
    all_c, _, _ = all_circuits(M, r, mode='global')
    circuits = sorted(all_c, key=sorted)
    print(f"Total circuits: {len(circuits)}\n")

    # Build transition matrices
    print("Building transition matrices...")
    P_global   = build_transition_matrix(M, r, circuits, mode='global')
    P_adjacent = build_transition_matrix(M, r, circuits, mode='adjacent')

    # Compute stationary distributions
    print("Computing stationary distributions...")
    pi_global   = stationary_distribution(P_global)
    pi_adjacent = stationary_distribution(P_adjacent)

    print(f"\n{'='*55}")
    print("  Stationary distribution comparison")
    print(f"{'='*55}")
    compare_distributions(pi_global, pi_adjacent, top_k=20)

    # Summary stats
    print(f"\n  --- Global mode ---")
    print(f"  Min π  : {pi_global.min():.6f}")
    print(f"  Max π  : {pi_global.max():.6f}")
    print(f"  Uniform: {1/len(circuits):.6f}")
    print(f"  Entropy: {-np.sum(pi_global * np.log(pi_global + 1e-300)):.4f}  "
          f"(max = {np.log(len(circuits)):.4f})")

    print(f"\n  --- Adjacent mode ---")
    print(f"  Min π  : {pi_adjacent.min():.6f}")
    print(f"  Max π  : {pi_adjacent.max():.6f}")
    print(f"  Uniform: {1/len(circuits):.6f}")
    print(f"  Entropy: {-np.sum(pi_adjacent * np.log(pi_adjacent + 1e-300)):.4f}  "
          f"(max = {np.log(len(circuits)):.4f})")

    # ── Feature correlations ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  Circuit feature correlations with stationary π")
    print(f"{'='*55}")
    print("  (computing features — may take ~30s)...")
    feats = circuit_features(M, r, circuits)
    correlate_with_stationary(feats, pi_global,   mode_label="global")
    correlate_with_stationary(feats, pi_adjacent, mode_label="adjacent")
