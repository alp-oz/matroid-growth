"""
Circuit tagging experiment.

For a small CSS code (default: Toric L=2, n=8, N=20 circuits):

1. Enumerate all circuits of M[H_X].
2. Tag each circuit correctly:
     stabilizer : 1_C ∈ rowspace(H_Z)               (Z-stabilizer support)
     logical    : 1_C ∈ ker(H_X) \ rowspace(H_Z)    (Z-logical operator support)
   Note: ALL circuits of M[H_X] are in ker(H_X) by definition, so every circuit
   is either a stabilizer or a logical — no "other" category exists.

3. Compute the EXACT stationary distribution of:
     (a) the original chain (no MH) — not uniform
     (b) the Metropolized chain (P_MH) — targets uniform exactly

4. Run the Metropolized chain and verify visit frequencies match uniform,
   with logicals and stabilizers both visited at their expected rates.

Key question: does the chain visit Z-logical circuits at the right proportion
for A(z) approximation?

Metropolization to uniform π:
   P_MH[i,j] = min(P[i,j], P[j,i])   for i ≠ j
   P_MH[i,i] = 1 - Σ_{j≠i} P_MH[i,j]
This is the standard "min" Metropolis rule: it preserves the support of P
and targets the uniform distribution exactly.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from scipy.stats import chisquare

from codes.toric_code import build_toric_hx, gf2_row_reduce
from core.circuits import all_circuits
from core.markov_chain import fundamental_circuits, decompose_into_circuits
from core.stationary import build_transition_matrix, stationary_distribution


# ─────────────────────────────────────────────────────────────────────────────
# GF(2) utilities
# ─────────────────────────────────────────────────────────────────────────────

def gf2_rank(H):
    H = H.copy().astype(np.uint8)
    m, n = H.shape
    rank = 0
    for col in range(n):
        pivot = next((r for r in range(rank, m) if H[r, col]), None)
        if pivot is None:
            continue
        H[[rank, pivot]] = H[[pivot, rank]]
        for r in range(m):
            if r != rank and H[r, col]:
                H[r] = (H[r] + H[rank]) % 2
        rank += 1
    return rank


def gf2_in_rowspace(H, v):
    """Test whether v ∈ rowspace(H) over GF(2)."""
    H = H.astype(np.uint8)
    v = np.array(v, dtype=np.uint8).reshape(1, -1)
    aug = np.vstack([H, v])
    return gf2_rank(aug) == gf2_rank(H)


# ─────────────────────────────────────────────────────────────────────────────
# Tag all circuits
# ─────────────────────────────────────────────────────────────────────────────

def tag_circuits(circuits, H_Z, col_order, n_q):
    """
    Tag each circuit as 'stabilizer' or 'logical'.

    Circuits of M[H_X] live in ker(H_X) by definition.
    Among them:
      stabilizer : 1_C ∈ rowspace(H_Z)   — Z-stabilizer
      logical    : 1_C ∉ rowspace(H_Z)   — Z-logical operator
    """
    tags = {}
    for C in circuits:
        c_vec = np.zeros(n_q, dtype=np.uint8)
        for j in C:
            c_vec[col_order[j]] = 1
        tags[C] = "stabilizer" if gf2_in_rowspace(H_Z, c_vec) else "logical"
    return tags


# ─────────────────────────────────────────────────────────────────────────────
# Metropolized chain targeting uniform π
# ─────────────────────────────────────────────────────────────────────────────

def metropolize(P):
    """
    Build P_MH from P targeting the uniform distribution.

    For uniform target π_i = 1/N:
      Detailed balance: π_i P_MH[i,j] = π_j P_MH[j,i]
      → P_MH[i,j] = P_MH[j,i]  (symmetric)

    Standard min rule:
      P_MH[i,j] = min(P[i,j], P[j,i])   for i ≠ j
      P_MH[i,i] = 1 - Σ_{j≠i} P_MH[i,j]

    This guarantees π_uniform is stationary and detailed balance holds.
    """
    N = P.shape[0]
    P_mh = np.minimum(P, P.T)
    np.fill_diagonal(P_mh, 0.0)
    np.fill_diagonal(P_mh, 1.0 - P_mh.sum(axis=1))
    return P_mh


def run_chain_from_matrix(P, circuits, T=200_000, burn_in=5_000, seed=42):
    """Run a chain using the given transition matrix. Returns visit Counter."""
    rng = np.random.default_rng(seed)
    N = len(circuits)
    state = rng.integers(N)
    counts = Counter()
    for t in range(T + burn_in):
        state = rng.choice(N, p=P[state])
        if t >= burn_in:
            counts[circuits[state]] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(L=2, T=200_000, seed=42):
    print(f"Toric L={L} — circuit tagging and uniformity experiment")
    print("=" * 60)

    # ── Build code ────────────────────────────────────────────────
    H_X = build_toric_hx(L)
    n_q = 2 * L * L

    H_Z = np.zeros((L * L, n_q), dtype=np.uint8)
    for i in range(L):
        for j in range(L):
            v = i * L + j
            H_Z[v, i * L + j]                     = 1
            H_Z[v, i * L + (j - 1) % L]           = 1
            H_Z[v, L*L + i * L + j]               = 1
            H_Z[v, L*L + ((i - 1) % L) * L + j]  = 1

    check = (H_X.astype(int) @ H_Z.T.astype(int)) % 2
    assert np.all(check == 0), "CSS condition failed!"

    H_rref, pivot_cols, r = gf2_row_reduce(H_X)
    free_cols = [j for j in range(n_q) if j not in set(pivot_cols)]
    col_order = pivot_cols + free_cols
    M = H_rref[:, col_order].astype(np.float64)

    print(f"  n={n_q}, rank(H_X)={r}, rank(H_Z)={gf2_rank(H_Z)}")
    print(f"  k = n - rank(H_X) - rank(H_Z) = {n_q - r - gf2_rank(H_Z)}")

    # ── Enumerate and tag circuits ─────────────────────────────────
    circuits_set, _, _ = all_circuits(M, r, mode="adjacent")
    circuits = list(circuits_set)
    N = len(circuits)
    print(f"  N={N} circuits")

    tags = tag_circuits(circuits, H_Z, col_order, n_q)
    tag_counts = Counter(tags.values())
    print(f"\nCircuit breakdown:")
    for tag in ["stabilizer", "logical"]:
        n_t = tag_counts[tag]
        print(f"  {tag:<12} : {n_t:>3}  ({100*n_t/N:.1f}%)")

    # ── Exact transition matrices ──────────────────────────────────
    print(f"\nBuilding exact transition matrix...", end=" ", flush=True)
    P_orig = build_transition_matrix(M, r, circuits, mode="adjacent")
    P_mh   = metropolize(P_orig)
    print("done")

    pi_orig = stationary_distribution(P_orig)
    pi_mh   = stationary_distribution(P_mh)
    uniform = np.full(N, 1.0 / N)

    print(f"\nStationary distributions:")
    print(f"  {'':20}  {'min':>8}  {'max':>8}  {'std':>8}  uniform?")
    print(f"  {'original chain':20}  {pi_orig.min():8.4f}  {pi_orig.max():8.4f}"
          f"  {pi_orig.std():8.5f}  {np.allclose(pi_orig, uniform, atol=1e-4)}")
    print(f"  {'metropolized':20}  {pi_mh.min():8.4f}  {pi_mh.max():8.4f}"
          f"  {pi_mh.std():8.5f}  {np.allclose(pi_mh, uniform, atol=1e-4)}")

    stab_idx = [i for i, C in enumerate(circuits) if tags[C] == "stabilizer"]
    log_idx  = [i for i, C in enumerate(circuits) if tags[C] == "logical"]

    print(f"\nMean π per class:")
    print(f"  {'':20}  {'stab':>8}  {'logical':>8}  ratio_log/stab")
    for label, pi in [("original", pi_orig), ("metropolized", pi_mh),
                       ("uniform", uniform)]:
        ms = pi[stab_idx].mean()
        ml = pi[log_idx].mean()
        print(f"  {label:<20}  {ms:8.5f}  {ml:8.5f}  {ml/ms:.4f}")

    # ── Run Metropolized chain ─────────────────────────────────────
    print(f"\nRunning metropolized chain  T={T:,}...", end=" ", flush=True)
    counts_mh = run_chain_from_matrix(P_mh, circuits, T=T, seed=seed)
    print("done")

    expected = T / N
    print(f"\nVisit analysis  (expected per circuit = {expected:.0f}):")
    print(f"  {'tag':<12}  {'N':>4}  {'mean_visits':>12}  {'ratio':>8}  chi2_p")
    for tag in ["stabilizer", "logical"]:
        circ_of_tag = [C for C, t in tags.items() if t == tag]
        visits = np.array([counts_mh.get(C, 0) for C in circ_of_tag], dtype=float)
        ratio = visits.mean() / expected
        exp_each = np.full(len(visits), visits.sum() / len(visits))
        _, p = chisquare(visits, f_exp=exp_each)
        uniform_tag = "uniform ✓" if p > 0.05 else "non-uniform ✗"
        print(f"  {tag:<12}  {len(circ_of_tag):>4}  {visits.mean():>12.1f}  "
              f"{ratio:>8.4f}  p={p:.3f} {uniform_tag}")

    # ── Figure ────────────────────────────────────────────────────
    tag_colors = {"stabilizer": "#2ca02c", "logical": "#e74c3c"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Toric L={L}: circuit visit frequencies  (T={T:,})\n"
        "Correct tagging: stabilizer = 1_C ∈ rowspace(H_Z);  logical = otherwise",
        fontsize=11, fontweight="bold"
    )

    # Panel A: original chain stationary dist
    ax = axes[0]
    xs = np.arange(N)
    sorted_idx = sorted(range(N), key=lambda i: tags[circuits[i]])
    sorted_colors = [tag_colors[tags[circuits[i]]] for i in sorted_idx]
    ax.bar(xs, pi_orig[sorted_idx] * N, color=sorted_colors, width=1, alpha=0.85)
    ax.axhline(1.0, color="black", lw=1.5, ls="--", label="uniform")
    patches = [mpatches.Patch(color=tag_colors[t],
               label=f"{t} (n={tag_counts[t]})") for t in ["stabilizer", "logical"]]
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel("Circuit index (sorted by tag)"); ax.set_ylabel("π × N  (1 = uniform)")
    ax.set_title("A.  Original chain\n(stationary dist, no MH)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Metropolized chain stationary dist (should be flat)
    ax = axes[1]
    ax.bar(xs, pi_mh[sorted_idx] * N, color=sorted_colors, width=1, alpha=0.85)
    ax.axhline(1.0, color="black", lw=1.5, ls="--", label="uniform")
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel("Circuit index (sorted by tag)"); ax.set_ylabel("π × N")
    ax.set_title("B.  Metropolized chain\n(should be exactly 1.0 everywhere)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: empirical visit ratio per tag
    ax = axes[2]
    tag_labels = ["stabilizer", "logical"]
    ratios, errors = [], []
    for tag in tag_labels:
        circ_of_tag = [C for C, t in tags.items() if t == tag]
        visits = np.array([counts_mh.get(C, 0) for C in circ_of_tag], dtype=float)
        ratios.append(visits.mean() / expected)
        errors.append(visits.std() / expected / np.sqrt(len(visits)))
    ax.bar(range(2), ratios, yerr=errors, color=[tag_colors[t] for t in tag_labels],
           width=0.5, edgecolor="white", alpha=0.88, capsize=6)
    ax.axhline(1.0, color="black", lw=1.5, ls="--", label="expected = 1")
    ax.set_xticks([0, 1]); ax.set_xticklabels(tag_labels)
    ax.set_ylabel("Mean visits / expected")
    ax.set_title("C.  Metropolized chain\nvisit ratio by class (should be ≈ 1)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(ratios) * 1.4)

    plt.tight_layout()
    out = "markov-circuits/tag_circuits.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out}")
    plt.close()

    return tags, pi_orig, pi_mh, counts_mh


if __name__ == "__main__":
    run_experiment(L=2, T=200_000)
