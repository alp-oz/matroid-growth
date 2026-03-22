"""
Chain 1 stationary distribution — exact computation and matroid theory connection.

Tasks:
  1. [[6,2,2]]: exact P matrix, stationary pi, reversibility check
  2. Toric L=2:  exact pi, correlation with circuit structure
  3. Tutte polynomial of M[H_X] for [[6,2,2]]
  4. Compare pi to matroid invariants
"""

import sys
import os
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr
from codes.toric_code import build_toric_hx, gf2_row_reduce
from core.markov_chain import fundamental_circuits, decompose_into_circuits
from core.stationary import build_transition_matrix, stationary_distribution
from core.circuits import all_circuits


# ─────────────────────────────────────────────────────────────────────────────
# GF(2) utilities
# ─────────────────────────────────────────────────────────────────────────────

def gf2_rank_subset(H, cols):
    """GF(2) rank of H restricted to column subset cols."""
    if not cols:
        return 0
    sub = H[:, list(cols)].astype(np.uint8).copy()
    m, n = sub.shape
    row = 0
    for col in range(n):
        piv = None
        for r in range(row, m):
            if sub[r, col]:
                piv = r; break
        if piv is None:
            continue
        sub[[row, piv]] = sub[[piv, row]]
        for r in range(m):
            if r != row and sub[r, col]:
                sub[r] = (sub[r] + sub[row]) % 2
        row += 1
    return row


def hx_to_matroid(H_X):
    """Convert H_X to RREF matroid [I|A] form. Returns M, r."""
    H_rref, pivot_cols, r = gf2_row_reduce(H_X)
    n = H_X.shape[1]
    free_cols = [j for j in range(n) if j not in set(pivot_cols)]
    col_order = pivot_cols + free_cols
    M = H_rref[:, col_order].astype(np.float64)
    return M, r


def circuit_support(C):
    """Return frozenset of element indices in circuit C (a frozenset or set)."""
    return frozenset(C)


def vec_to_support(v):
    """Binary vector → frozenset of nonzero positions."""
    return frozenset(int(i) for i in np.where(np.array(v) != 0)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Detailed analysis of one matroid
# ─────────────────────────────────────────────────────────────────────────────

def analyse_chain1(H_X, label, mode="adjacent", top_k=None):
    M, r = hx_to_matroid(H_X)
    n = H_X.shape[1]
    fc = fundamental_circuits(M, r)
    non_basis = list(range(r, n))

    # Enumerate all circuits (BFS)
    all_c, trunc, _ = all_circuits(M, r, mode=mode)
    circuits = sorted(all_c, key=lambda c: (len(c), sorted(c)))
    N = len(circuits)
    print(f"\n{label}: rank={r}, n={n}, N_circuits={N}, truncated={trunc}")

    # Build exact transition matrix
    P = build_transition_matrix(M, r, circuits, mode=mode)
    pi = stationary_distribution(P)

    # Per-circuit features
    circuit_list = list(circuits)
    weights = np.array([len(c) for c in circuit_list])

    eligible_sizes = np.zeros(N, dtype=int)
    degrees = np.zeros(N, dtype=int)

    idx = {c: i for i, c in enumerate(circuit_list)}

    for i, C in enumerate(circuit_list):
        elig = [j for j in non_basis if fc[j] & C]
        eligible_sizes[i] = len(elig)
        reachable = set()
        for j in elig:
            sym = C ^ fc[j]
            if sym:
                parts = decompose_into_circuits(M, sym)
                reachable.update(parts)
        degrees[i] = len(reachable)

    return {
        "label": label, "M": M, "r": r, "n": n, "N": N,
        "circuits": circuit_list,
        "P": P, "pi": pi,
        "weights": weights,
        "eligible_sizes": eligible_sizes,
        "degrees": degrees,
        "fc": fc, "non_basis": non_basis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reversibility check
# ─────────────────────────────────────────────────────────────────────────────

def check_reversibility(P, pi, circuits, label, tol=1e-9):
    N = len(pi)
    violations = []
    for i in range(N):
        for j in range(i+1, N):
            flow_ij = pi[i] * P[i, j]
            flow_ji = pi[j] * P[j, i]
            if abs(flow_ij - flow_ji) > tol:
                violations.append((i, j, flow_ij, flow_ji,
                                   abs(flow_ij - flow_ji)))

    print(f"\n{label} — Reversibility (detailed balance):")
    if not violations:
        print("  Chain IS reversible — detailed balance holds for all pairs.")
    else:
        print(f"  Chain is NOT reversible — {len(violations)} pairs violate detailed balance.")
        print(f"  {'i':>3} {'j':>3}  {'π(i)P(i,j)':>14}  {'π(j)P(j,i)':>14}  {'|diff|':>12}")
        for (i, j, fij, fji, diff) in sorted(violations, key=lambda x: -x[4])[:10]:
            ci = sorted(circuits[i])
            cj = sorted(circuits[j])
            print(f"  {i:>3} {j:>3}  {fij:>14.8f}  {fji:>14.8f}  {diff:>12.2e}"
                  f"   C{i+1}={ci} ↔ C{j+1}={cj}")
    return violations


# ─────────────────────────────────────────────────────────────────────────────
# Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────

def correlation_table(res):
    pi = res["pi"]
    label = res["label"]
    w   = res["weights"].astype(float)
    e   = res["eligible_sizes"].astype(float)
    d   = res["degrees"].astype(float)
    inv_e = 1.0 / np.where(e > 0, e, 1.0)

    print(f"\n{label} — Correlation with π:")
    print(f"  {'Feature':<22}  {'Pearson r':>10}  {'p-value':>10}")
    print("  " + "-" * 46)
    for name, x in [("weight wt(C)",      w),
                     ("|eligible(C)|",     e),
                     ("degree",            d),
                     ("1/|eligible(C)|",   inv_e)]:
        if np.std(x) < 1e-12:
            print(f"  {name:<22}  {'(constant)':>10}")
            continue
        r_val, p_val = pearsonr(pi, x)
        print(f"  {name:<22}  {r_val:>10.4f}  {p_val:>10.4e}")


# ─────────────────────────────────────────────────────────────────────────────
# Tutte polynomial  (brute force over all 2^n subsets)
# ─────────────────────────────────────────────────────────────────────────────

def tutte_polynomial(H_X):
    """
    Compute Tutte polynomial T(M; x, y) symbolically as coefficient table.

    T(M; x, y) = Σ_{A ⊆ E} (x-1)^{r(M)-r(A)} (y-1)^{|A|-r(A)}

    Returns dict {(i,j): coeff} for T = Σ c_{ij} (x-1)^i (y-1)^j,
    plus a function for evaluating at specific (x,y).
    """
    n = H_X.shape[1]
    r_M = gf2_rank_subset(H_X, list(range(n)))

    # Accumulate coefficients indexed by (r(M)-r(A), |A|-r(A))
    from collections import defaultdict
    coeffs = defaultdict(int)

    for mask in range(1 << n):
        A = [i for i in range(n) if mask & (1 << i)]
        r_A = gf2_rank_subset(H_X, A)
        i_exp = r_M - r_A       # power of (x-1)
        j_exp = len(A) - r_A   # power of (y-1)
        coeffs[(i_exp, j_exp)] += 1

    def evaluate(x, y):
        val = 0
        for (i, j), c in coeffs.items():
            val += c * ((x - 1) ** i) * ((y - 1) ** j)
        return val

    return dict(coeffs), evaluate, r_M


def rank_polynomial(H_X):
    """
    R(M; u, v) = Σ_{A ⊆ E} u^{r(M)-r(A)} v^{|A|-r(A)}
    Evaluates at (u,v).
    """
    n = H_X.shape[1]
    r_M = gf2_rank_subset(H_X, list(range(n)))
    from collections import defaultdict
    coeffs = defaultdict(int)
    for mask in range(1 << n):
        A = [i for i in range(n) if mask & (1 << i)]
        r_A = gf2_rank_subset(H_X, A)
        coeffs[(r_M - r_A, len(A) - r_A)] += 1

    def evaluate(u, v):
        return sum(c * (u ** i) * (v ** j) for (i, j), c in coeffs.items())

    return dict(coeffs), evaluate


def num_bases_containing(H_X, C_support):
    """Count bases of M[H_X] that contain all elements of C_support."""
    n = H_X.shape[1]
    r_M = gf2_rank_subset(H_X, list(range(n)))
    count = 0
    for cols in combinations(range(n), r_M):
        if all(c in cols for c in C_support):
            if gf2_rank_subset(H_X, list(cols)) == r_M:
                count += 1
    return count


# ═════════════════════════════════════════════════════════════════════════════
# TASK 1  —  [[6,2,2]] exact stationary distribution
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1 — [[6,2,2]] exact P matrix and stationary distribution")
print("=" * 70)

H6 = np.array([[1,1,1,1,0,0],
               [0,0,1,1,1,1]], dtype=np.uint8)

res6 = analyse_chain1(H6, "[[6,2,2]]", mode="adjacent")
circuits6 = res6["circuits"]
P6 = res6["P"]
pi6 = res6["pi"]
N6 = res6["N"]

print(f"\nCircuits ({N6} total):")
for i, C in enumerate(circuits6):
    sup = sorted(C)
    wt  = len(C)
    print(f"  C{i+1} = {sup}  wt={wt}  pi={pi6[i]:.8f}  "
          f"elig={res6['eligible_sizes'][i]}  deg={res6['degrees'][i]}")

print(f"\nTransition matrix P ({N6}×{N6}):")
np.set_printoptions(precision=6, suppress=True)
print(P6)

uniform = 1.0 / N6
l1_from_uniform = float(np.sum(np.abs(pi6 - uniform)))
print(f"\nStationary distribution π:")
for i, C in enumerate(circuits6):
    print(f"  π(C{i+1}) = {pi6[i]:.8f}  (uniform would be {uniform:.8f})")
print(f"\nUniform (1/N each)? {'YES' if l1_from_uniform < 1e-9 else 'NO'}")
print(f"L1 distance from uniform: {l1_from_uniform:.6e}")

check_reversibility(P6, pi6, circuits6, "[[6,2,2]]")


# ═════════════════════════════════════════════════════════════════════════════
# TASK 2  —  Toric L=2 exact stationary distribution
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 2 — Toric L=2 exact stationary distribution")
print("=" * 70)

H_toric2 = build_toric_hx(2)
res_t2 = analyse_chain1(H_toric2, "Toric L=2", mode="adjacent")
circuits_t2 = res_t2["circuits"]
P_t2 = res_t2["P"]
pi_t2 = res_t2["pi"]
N_t2 = res_t2["N"]

uniform_t2 = 1.0 / N_t2
l1_t2 = float(np.sum(np.abs(pi_t2 - uniform_t2)))

print(f"\nAll {N_t2} circuits sorted by π(C) descending:")
order = np.argsort(-pi_t2)
print(f"  {'#':>3}  {'support':>30}  {'wt':>4}  {'π(C)':>10}  "
      f"{'|elig|':>7}  {'deg':>5}")
print("  " + "-" * 65)
for rank, i in enumerate(order, 1):
    sup = sorted(circuits_t2[i])
    print(f"  {rank:>3}  {str(sup):>30}  {res_t2['weights'][i]:>4}  "
          f"{pi_t2[i]:>10.6f}  {res_t2['eligible_sizes'][i]:>7}  "
          f"{res_t2['degrees'][i]:>5}")

print(f"\nL1 distance from uniform (1/{N_t2} = {uniform_t2:.6f}): {l1_t2:.6f}")
check_reversibility(P_t2, pi_t2, circuits_t2, "Toric L=2", tol=1e-9)
correlation_table(res_t2)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3  —  Tutte polynomial of M[H_X] for [[6,2,2]]
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 3 — Tutte polynomial of M[H_X] for [[6,2,2]]")
print("=" * 70)

tutte_coeffs, tutte_eval, r_M = tutte_polynomial(H6)
rank_coeffs, rank_eval = rank_polynomial(H6)

print(f"\nGround set: {{0,1,2,3,4,5}},  rank r(M) = {r_M}")

print("\nTutte polynomial T(M; x, y) — coefficient table:")
print("  Coefficients of (x-1)^i (y-1)^j:")
all_i = sorted(set(i for i,j in tutte_coeffs))
all_j = sorted(set(j for i,j in tutte_coeffs))
print(f"  {'(i\\j)':>6}", end="")
for j in all_j:
    print(f"  j={j}", end="")
print()
for i in all_i:
    print(f"  i={i:>3}  ", end="")
    for j in all_j:
        print(f"  {tutte_coeffs.get((i,j), 0):>4}", end="")
    print()

print("\nSpecific evaluations:")
evals = [
    ("T(1,1) = #bases",              1, 1),
    ("T(2,1) = #independent sets",   2, 1),
    ("T(1,2) = #spanning sets",      1, 2),
    ("T(0,2)",                        0, 2),
    ("T(2,0)",                        2, 0),
    ("T(0,0)",                        0, 0),
    ("T(3,3)",                        3, 3),
]
for desc, x, y in evals:
    val = tutte_eval(x, y)
    print(f"  {desc:<30} = {int(round(val))}")

print("\nRank polynomial R(M; u, v) — coefficient table:")
print("  Coefficients of u^i v^j:")
for i in sorted(set(i for i,j in rank_coeffs)):
    for j in sorted(set(j for i,j in rank_coeffs)):
        c = rank_coeffs.get((i,j), 0)
        if c:
            print(f"  u^{i} v^{j}: {c}")


# ═════════════════════════════════════════════════════════════════════════════
# TASK 4  —  Compare π to matroid invariants
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 4 — Comparing π to matroid invariants")
print("=" * 70)

# --- [[6,2,2]] ---
print("\n[[6,2,2]] (if not uniform, investigate; if uniform, note by symmetry)")
print(f"  π uniform? {'YES — perfectly symmetric code' if l1_from_uniform < 1e-9 else 'NO'}")

if l1_from_uniform < 1e-9:
    print("  All circuits have identical structure → π = uniform by symmetry.")
    print("  Computing per-circuit matroid invariants anyway:")

for i, C in enumerate(circuits6):
    sup = frozenset(C)
    n_bases = num_bases_containing(H6, sup)
    wt = len(C)
    print(f"  C{i+1} {sorted(sup)} wt={wt}  #bases_containing={n_bases}  "
          f"π={pi6[i]:.6f}  |elig|={res6['eligible_sizes'][i]}")

# --- Toric L=2 (main analysis since non-uniform) ---
print(f"\nToric L=2 — π vs matroid invariants")

# Bases containing each circuit
bases_containing = np.zeros(N_t2, dtype=int)
for i, C in enumerate(circuits_t2):
    bases_containing[i] = num_bases_containing(H_toric2, frozenset(C))

print(f"\n  Per-circuit: π vs #bases_containing, wt, |elig|, degree")
print(f"  {'C':>3}  {'wt':>4}  {'π':>10}  {'|elig|':>7}  {'deg':>5}  "
      f"{'#bases':>7}  {'1/|elig|':>10}")
order2 = np.argsort(-pi_t2)
for i in order2[:20]:   # top 20 by pi
    print(f"  {i+1:>3}  {res_t2['weights'][i]:>4}  {pi_t2[i]:>10.6f}  "
          f"{res_t2['eligible_sizes'][i]:>7}  {res_t2['degrees'][i]:>5}  "
          f"{bases_containing[i]:>7}  "
          f"{1/res_t2['eligible_sizes'][i]:>10.6f}")

print(f"\n  Correlation of π with matroid invariants (Toric L=2):")
features = [
    ("weight wt(C)",     res_t2["weights"].astype(float)),
    ("|eligible(C)|",    res_t2["eligible_sizes"].astype(float)),
    ("degree",           res_t2["degrees"].astype(float)),
    ("1/|eligible|",     1.0/res_t2["eligible_sizes"].astype(float)),
    ("#bases_containing",bases_containing.astype(float)),
]
print(f"  {'Feature':<22}  {'Pearson r':>10}  {'p-value':>10}")
print("  " + "-" * 46)
for name, x in features:
    if np.std(x) < 1e-12:
        print(f"  {name:<22}  (constant)")
        continue
    r_val, p_val = pearsonr(pi_t2, x)
    print(f"  {name:<22}  {r_val:>10.4f}  {p_val:>10.4e}")

# Check if pi ∝ 1/|eligible| (closed-form candidate)
print("\n  Testing π ∝ 1/|eligible(C)|:")
inv_e = 1.0 / res_t2["eligible_sizes"].astype(float)
Z = inv_e.sum()
pi_candidate = inv_e / Z
l1_cand = float(np.sum(np.abs(pi_t2 - pi_candidate)))
print(f"  L1(π, 1/|elig| normalised) = {l1_cand:.6f}  "
      f"{'← MATCHES!' if l1_cand < 1e-6 else '← does not match'}")

print("\n  Testing π ∝ degree:")
deg = res_t2["degrees"].astype(float)
Z2 = deg.sum()
pi_cand_deg = deg / Z2
l1_deg = float(np.sum(np.abs(pi_t2 - pi_cand_deg)))
print(f"  L1(π, degree/sum_degree)   = {l1_deg:.6f}  "
      f"{'← MATCHES!' if l1_deg < 1e-6 else '← does not match'}")

# Check detailed balance violation pattern for Toric L=2
violations_t2 = check_reversibility(P_t2, pi_t2, circuits_t2, "Toric L=2", tol=1e-9)
if violations_t2:
    print(f"\n  Total probability flow imbalance per pair (top 5):")
    top5 = sorted(violations_t2, key=lambda x: -x[4])[:5]
    for (i, j, fij, fji, diff) in top5:
        print(f"  C{i+1}(wt={res_t2['weights'][i]}) ↔ "
              f"C{j+1}(wt={res_t2['weights'][j]}): "
              f"flow {fij:.6f} vs {fji:.6f}")
