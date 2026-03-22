"""
chain1_basis_artifact.py

Tests whether Chain 1's non-uniform stationary distribution is a basis-choice
artifact.  For each basis B of M[H_X], builds Chain 1 with fc_B (fundamental
circuits of B) — a different transition kernel.  Computes pi_B per basis,
then aggregates.

Tasks
-----
  1. [[6,2,2]]  — enumerate all 12 bases, compute pi_B, pi_avg, var_B, P_avg
  2. Check: is pi_avg uniform?
  3. Toric L=2  — same analysis
  4. activity_score(C) = #{bases B : min(C) ∈ B}; correlate with pi_avg

All computation is exact (no simulation).
"""

import sys
import os
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr
from codes.toric_code import build_toric_hx


# ─────────────────────────────────────────────────────────────────────────────
# GF(2) primitives
# ─────────────────────────────────────────────────────────────────────────────

def gf2_rank(H, cols):
    """GF(2) rank of H restricted to column indices `cols`."""
    if not cols:
        return 0
    sub = H[:, list(cols)].astype(np.uint8).copy()
    m, nc = sub.shape
    row = 0
    for col in range(nc):
        piv = next((r for r in range(row, m) if sub[r, col]), None)
        if piv is None:
            continue
        sub[[row, piv]] = sub[[piv, row]]
        for r in range(m):
            if r != row and sub[r, col]:
                sub[r] = (sub[r] + sub[row]) % 2
        row += 1
    return row


def enumerate_bases(H_X):
    """Return (bases, r) where bases is a list of sorted column-index lists."""
    n = H_X.shape[1]
    r = gf2_rank(H_X, list(range(n)))
    bases = [list(cols) for cols in combinations(range(n), r)
             if gf2_rank(H_X, list(cols)) == r]
    return bases, r


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental circuits for an arbitrary basis
# ─────────────────────────────────────────────────────────────────────────────

def get_fc_for_basis(H_X, basis):
    """
    Compute fundamental circuits of M[H_X] w.r.t. given basis B.

    Returns (non_basis, fc) where:
      non_basis : sorted list of column indices not in basis
      fc        : dict  j -> frozenset of original column indices
                  fc[j] is the unique circuit contained in basis ∪ {j}
    """
    n = H_X.shape[1]
    r = len(basis)
    basis_set = set(basis)
    non_basis = sorted(j for j in range(n) if j not in basis_set)

    # Column-permuted matrix: [basis columns | non_basis columns]
    col_order = list(basis) + non_basis
    mat = H_X[:, col_order].astype(np.uint8).copy()

    # GF(2) RREF on the first r columns → they become I_r
    for col in range(r):
        piv = next((row for row in range(col, mat.shape[0])
                    if mat[row, col]), None)
        if piv is None:
            raise ValueError(f"Column {col} of basis has no pivot — "
                             "basis is not independent!")
        mat[[col, piv]] = mat[[piv, col]]
        for row in range(mat.shape[0]):
            if row != col and mat[row, col]:
                mat[row] = (mat[row] + mat[col]) % 2

    # After reduction: mat[:r, r+k] = A_B[:, k]
    # fc[non_basis[k]] = {non_basis[k]} ∪ {basis[i] : A_B[i,k] = 1}
    fc = {}
    for k, j in enumerate(non_basis):
        support = {j}
        for i in range(r):
            if mat[i, r + k]:
                support.add(basis[i])
        fc[j] = frozenset(support)

    return non_basis, fc


# ─────────────────────────────────────────────────────────────────────────────
# Circuit enumeration (brute force over all subsets)
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_all_circuits(H_X):
    """
    Enumerate ALL circuits of M[H_X] by brute force over all subsets.

    A subset S is a circuit iff it is dependent (gf2_rank(H_X,S) < |S|) and
    every proper subset is independent.  Feasible for n ≤ ~20.

    Returns a sorted list of frozensets (supports in original column indices).
    """
    n = H_X.shape[1]
    circuits = []
    for size in range(2, n + 1):
        for cols in combinations(range(n), size):
            cols_list = list(cols)
            if gf2_rank(H_X, cols_list) < size:          # dependent
                minimal = all(
                    gf2_rank(H_X, cols_list[:i] + cols_list[i+1:]) == size - 1
                    for i in range(size)
                )
                if minimal:
                    circuits.append(frozenset(cols))
    return sorted(circuits, key=lambda c: (len(c), sorted(c)))


# ─────────────────────────────────────────────────────────────────────────────
# Stationary distribution
# ─────────────────────────────────────────────────────────────────────────────

def stationary_distribution(P):
    """Exact stationary distribution via linear solve (π P = π, Σπ = 1)."""
    N = P.shape[0]
    A = (P.T - np.eye(N)).astype(np.float64)
    A[-1, :] = 1.0
    b = np.zeros(N); b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Chain 1 transition matrix for a given basis (fc_B-decomposition)
# ─────────────────────────────────────────────────────────────────────────────

def build_P_for_basis(circuits, idx, H_X, basis):
    """
    Build the Chain 1 transition matrix P_B using fundamental circuits of basis B.

    For circuit C, eligible j: fc_B[j] ∩ C ≠ ∅.
    sym = C △ fc_B[j].
    fc_B-decomposition of sym: {fc_B[j2] : j2 ∉ basis, j2 ∈ sym}.
    Pick one part uniformly → new state.

    This is correct because fc_B[j2] has exactly one non-basis element j2,
    so j2 ∈ sym iff fc_B[j2] appears in the fc_B-decomposition of sym.
    """
    non_basis, fc_B = get_fc_for_basis(H_X, basis)
    N = len(circuits)
    P = np.zeros((N, N), dtype=np.float64)

    for i, C in enumerate(circuits):
        eligible = [j for j in non_basis if fc_B[j] & C]
        n_elig = len(eligible)
        if n_elig == 0:
            P[i, i] = 1.0
            continue

        for j in eligible:
            sym = C ^ fc_B[j]
            if not sym:
                P[i, i] += 1.0 / n_elig
                continue

            # fc_B-decomposition: j2 ∉ basis, j2 ∈ sym
            part_keys = [j2 for j2 in non_basis if j2 in sym]
            if not part_keys:
                P[i, i] += 1.0 / n_elig
                continue

            parts = [fc_B[j2] for j2 in part_keys]
            w = 1.0 / (n_elig * len(parts))
            for part in parts:
                k = idx.get(part)
                if k is not None:
                    P[i, k] += w
                # If k is None: fc_B[j2] not in our circuit list — see assertion below

    row_sums = P.sum(axis=1, keepdims=True)
    P /= np.where(row_sums > 0, row_sums, 1.0)
    return P


# ─────────────────────────────────────────────────────────────────────────────
# Activity score: #{bases B : min(C) ∈ B}
# ─────────────────────────────────────────────────────────────────────────────

def activity_scores(circuits, bases):
    """
    activity_score(C) = #{B ∈ bases : min(C) ∈ B}.

    min(C) is the smallest original column index in C.
    """
    scores = np.zeros(len(circuits), dtype=int)
    for i, C in enumerate(circuits):
        m = min(C)
        for B in bases:
            if m in B:
                scores[i] += 1
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Full basis artifact analysis for one code
# ─────────────────────────────────────────────────────────────────────────────

def basis_artifact_analysis(H_X, label):
    print("=" * 70)
    print(f"BASIS ARTIFACT ANALYSIS — {label}")
    print("=" * 70)

    # ── Enumerate circuits ────────────────────────────────────────────────────
    circuits = enumerate_all_circuits(H_X)
    N = len(circuits)
    idx = {c: i for i, c in enumerate(circuits)}
    print(f"\n  Circuits: N = {N}")

    # ── Enumerate bases ───────────────────────────────────────────────────────
    bases, r = enumerate_bases(H_X)
    n = H_X.shape[1]
    B_count = len(bases)
    print(f"  Bases:    {B_count}  (rank r={r}, n={n})")

    # Check that all fc_B[j2] appearing in P_B are in circuits list
    for basis in bases:
        non_basis, fc_B = get_fc_for_basis(H_X, basis)
        for j, c in fc_B.items():
            assert c in idx, (f"fc_B[{j}] for basis {basis} = {sorted(c)} "
                              f"not in circuit list!")
    print(f"  All fc_B circuits in enumerated list ✓")

    # ── Compute pi_B for each basis ───────────────────────────────────────────
    all_pi = np.zeros((B_count, N), dtype=np.float64)
    all_P  = np.zeros((B_count, N, N), dtype=np.float64)

    for b_idx, basis in enumerate(bases):
        P_B  = build_P_for_basis(circuits, idx, H_X, basis)
        pi_B = stationary_distribution(P_B)
        all_pi[b_idx] = pi_B
        all_P[b_idx]  = P_B

    # ── Aggregate ─────────────────────────────────────────────────────────────
    pi_avg = all_pi.mean(axis=0)
    P_avg  = all_P.mean(axis=0)
    pi_uniform = np.full(N, 1.0 / N)

    # Variance across bases (per circuit, then averaged)
    pi_var_per_circuit = all_pi.var(axis=0)
    pi_var_mean = float(pi_var_per_circuit.mean())

    # L1 distances
    l1_avg_from_uniform = float(np.sum(np.abs(pi_avg - pi_uniform)))
    l1_per_basis = [float(np.sum(np.abs(all_pi[b] - pi_uniform)))
                    for b in range(B_count)]

    print(f"\n  Stationary distribution summary:")
    print(f"  {'':25}  {'mean':>10}  {'min':>10}  {'max':>10}")
    print(f"  {'L1(pi_B, uniform)':25}  "
          f"{np.mean(l1_per_basis):>10.6f}  "
          f"{np.min(l1_per_basis):>10.6f}  "
          f"{np.max(l1_per_basis):>10.6f}")
    print(f"  {'L1(pi_avg, uniform)':25}  {l1_avg_from_uniform:>10.6f}")
    print(f"  {'Var_B[pi_B(C)] averaged':25}  {pi_var_mean:>10.2e}")

    is_uniform = l1_avg_from_uniform < 1e-6
    print(f"\n  Is pi_avg uniform? {'YES ← artifact CONFIRMED' if is_uniform else 'NO ← artifact only partial'}")
    if not is_uniform:
        print(f"  (L1 from uniform = {l1_avg_from_uniform:.2e}, "
              f"threshold 1e-6)")

    # ── Per-circuit detail ────────────────────────────────────────────────────
    print(f"\n  Per-circuit: pi_avg vs pi_B range")
    print(f"  {'C':>3}  {'support':>28}  {'wt':>3}  {'pi_avg':>10}  "
          f"{'pi_min_B':>10}  {'pi_max_B':>10}  {'var_B':>10}")
    print("  " + "-" * 80)
    order = np.argsort(-pi_avg)
    for rank_i, i in enumerate(order):
        C = circuits[i]
        sup = sorted(C)
        wt  = len(C)
        pi_min = float(all_pi[:, i].min())
        pi_max = float(all_pi[:, i].max())
        print(f"  {i+1:>3}  {str(sup):>28}  {wt:>3}  "
              f"{pi_avg[i]:>10.6f}  {pi_min:>10.6f}  {pi_max:>10.6f}  "
              f"{pi_var_per_circuit[i]:>10.2e}")

    # ── P_avg stationarity ────────────────────────────────────────────────────
    pi_avg_P = pi_avg @ P_avg
    l1_P_stationary = float(np.sum(np.abs(pi_avg_P - pi_avg)))
    print(f"\n  Is pi_avg stationary for P_avg? "
          f"{'YES' if l1_P_stationary < 1e-9 else 'NO'}  "
          f"(L1 = {l1_P_stationary:.2e})")
    pi_unif_P = pi_uniform @ P_avg
    l1_unif_P = float(np.sum(np.abs(pi_unif_P - pi_uniform)))
    print(f"  Is uniform stationary for P_avg? "
          f"{'YES' if l1_unif_P < 1e-9 else 'NO'}  "
          f"(L1 = {l1_unif_P:.2e})")

    # ── Activity score vs pi_avg ──────────────────────────────────────────────
    act = activity_scores(circuits, bases).astype(float)
    wts = np.array([len(C) for C in circuits], dtype=float)

    print(f"\n  Correlation of pi_avg with circuit features:")
    print(f"  {'Feature':<22}  {'Pearson r':>10}  {'p-value':>10}")
    print("  " + "-" * 46)
    for fname, fvec in [("activity_score", act),
                         ("weight wt(C)",  wts)]:
        if np.std(fvec) < 1e-12:
            print(f"  {fname:<22}  (constant)")
            continue
        r_val, p_val = pearsonr(pi_avg, fvec)
        print(f"  {fname:<22}  {r_val:>10.4f}  {p_val:>10.4e}")

    # Normalised activity as probability candidate
    act_norm = act / act.sum()
    l1_act = float(np.sum(np.abs(pi_avg - act_norm)))
    print(f"\n  Testing pi_avg ∝ activity_score:")
    print(f"  L1(pi_avg, act/sum_act) = {l1_act:.6f}  "
          f"{'← MATCHES!' if l1_act < 1e-6 else '← does not match'}")

    # Uniform test on pi_avg
    print(f"\n  Testing pi_avg = uniform (1/{N}):")
    print(f"  L1(pi_avg, uniform) = {l1_avg_from_uniform:.6f}  "
          f"{'← MATCHES!' if l1_avg_from_uniform < 1e-6 else '← does not match'}")

    return {
        "label": label,
        "N": N,
        "B_count": B_count,
        "circuits": circuits,
        "pi_avg": pi_avg,
        "pi_var": pi_var_per_circuit,
        "P_avg": P_avg,
        "all_pi": all_pi,
        "l1_avg_uniform": l1_avg_from_uniform,
        "activity": act,
        "weights": wts,
        "is_uniform": is_uniform,
    }


# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 & 2 — [[6,2,2]]
# ═════════════════════════════════════════════════════════════════════════════

H6 = np.array([[1, 1, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 1]], dtype=np.uint8)

res6 = basis_artifact_analysis(H6, "[[6,2,2]]")


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3 — Toric L=2
# ═════════════════════════════════════════════════════════════════════════════

print()
H_t2 = build_toric_hx(2)
res_t2 = basis_artifact_analysis(H_t2, "Toric L=2")


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  {'Code':<16}  {'N':>4}  {'#bases':>7}  "
      f"{'L1(pi_avg, unif)':>18}  {'pi_avg uniform?'}")
print("  " + "-" * 60)
for res in [res6, res_t2]:
    print(f"  {res['label']:<16}  {res['N']:>4}  {res['B_count']:>7}  "
          f"{res['l1_avg_uniform']:>18.6f}  "
          f"{'YES' if res['is_uniform'] else 'NO'}")

print()
print("  Interpretation:")
print("  - If pi_avg is uniform: non-uniformity of pi_B is ENTIRELY a basis")
print("    artifact — the 'true' stationary measure (averaged over bases)")
print("    is uniform.")
print("  - If pi_avg is not uniform: there is an intrinsic non-uniformity")
print("    in the circuit space that persists across all basis choices.")
print()
print("  Connection to Tutte polynomial:")
print("  activity_score(C) = #{B : min(C) ∈ B} counts internal activity.")
print("  If pi_avg ∝ activity_score, then the stationary measure has a")
print("  direct interpretation via the Tutte expansion.")
