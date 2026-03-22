"""
chain1_piavg_formula.py

Find the closed form of pi_avg(C) = f(wt(C), n, r).

Chain 1 basis-averaged stationary distribution:
  For each basis B, Chain 1 uses fc_B-decomposition (Method A):
    eligible j in non_B s.t. fc_B[j] ∩ C != ∅
    sym = C Δ fc_B[j]
    parts = {fc_B[j'] : j' in non_B, j' in sym}   ← fc_B-decomposition
    pick part uniformly → new state
  The chain visits only the n-r fundamental circuits of B.
  pi_B extended to all N circuits: pi_B(C)=0 if C not fundamental for B.

  pi_avg(C) = mean over all (sampled) bases B of pi_B(C).

Tasks:
  1. Fix/verify Toric L=2 full weight distribution
  2. Toric L=3 pi_avg analysis
  3. BB(3,3) pi_avg analysis
  4. Test 8 candidate closed forms
  5. Matroid-specific quantities

Codes: [[6,2,2]], Toric L=2, Toric L=3, BB(3,3)
"""

import sys, os
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr
from math import comb
import random as pyrandom
from codes.toric_code import build_toric_hx
from codes.bb_code import bb_code


# ─────────────────────────────────────────────────────────────────────────────
# GF(2) primitives (all in original column coords)
# ─────────────────────────────────────────────────────────────────────────────

def gf2_rank(H, cols):
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


def find_circuit_hx(H_X, S):
    """GF(2) Gaussian elimination on cols S (frozenset of original indices).
    Returns a circuit (frozenset) contained in S, or None if S is independent."""
    cols = sorted(S)
    nc = len(cols)
    if nc == 0:
        return None
    m = H_X.shape[0]
    reduced = {}  # pivot_row -> (col_vector, dep_vector)
    for j_idx in range(nc):
        v = H_X[:, cols[j_idx]].astype(int).copy()
        d = np.zeros(nc, dtype=int)
        d[j_idx] = 1
        for row in sorted(reduced):
            if v[row]:
                rv, rd = reduced[row]
                v = (v + rv) % 2
                d = (d + rd) % 2
        pivot = next((row for row in range(m) if v[row]), None)
        if pivot is None:
            return frozenset(cols[i] for i in range(nc) if d[i])
        reduced[pivot] = (v, d)
    return None


def decompose_hx(H_X, sym):
    """Decompose frozenset sym into circuits (Method B, arbitrary GF(2) order)."""
    circuits = []
    remaining = set(sym)
    while remaining:
        C = find_circuit_hx(H_X, frozenset(remaining))
        if C is None:
            break
        circuits.append(C)
        remaining ^= C
    return circuits


# ─────────────────────────────────────────────────────────────────────────────
# Basis utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_rank(H_X):
    return gf2_rank(H_X, list(range(H_X.shape[1])))


def get_fc_for_basis(H_X, basis):
    """Fundamental circuits of M[H_X] w.r.t. basis (in original column indices)."""
    n = H_X.shape[1]
    r = len(basis)
    basis_set = set(basis)
    non_basis = sorted(j for j in range(n) if j not in basis_set)
    col_order = list(basis) + non_basis
    mat = H_X[:, col_order].astype(np.uint8).copy()
    for col in range(r):
        piv = next((row for row in range(col, mat.shape[0]) if mat[row, col]), None)
        if piv is None:
            raise ValueError(f"Basis column {col} has no pivot")
        mat[[col, piv]] = mat[[piv, col]]
        for row in range(mat.shape[0]):
            if row != col and mat[row, col]:
                mat[row] = (mat[row] + mat[col]) % 2
    fc = {}
    for k, j in enumerate(non_basis):
        support = {j}
        for i in range(r):
            if mat[i, r + k]:
                support.add(basis[i])
        fc[j] = frozenset(support)
    return non_basis, fc


def canonical_basis(H_X):
    """Canonical basis from RREF of H_X."""
    n = H_X.shape[1]
    mat = H_X.astype(np.uint8).copy()
    m = mat.shape[0]
    pivot_cols = []
    cur_row = 0
    for col in range(n):
        piv = next((row for row in range(cur_row, m) if mat[row, col]), None)
        if piv is None:
            continue
        mat[[cur_row, piv]] = mat[[piv, cur_row]]
        for row in range(m):
            if row != cur_row and mat[row, col]:
                mat[row] = (mat[row] + mat[cur_row]) % 2
        pivot_cols.append(col)
        cur_row += 1
    return pivot_cols


def enumerate_bases(H_X):
    """All bases of M[H_X]. Returns (bases, r). Feasible for n <= ~18 with few bases."""
    n = H_X.shape[1]
    r = get_rank(H_X)
    bases = [list(cols) for cols in combinations(range(n), r)
             if gf2_rank(H_X, list(cols)) == r]
    return bases, r


def sample_bases(H_X, r, n_sample=200, seed=42):
    """Sample n_sample random bases uniformly (rejection sampling)."""
    rng = np.random.default_rng(seed)
    n = H_X.shape[1]
    bases = []
    seen = set()
    attempts = 0
    while len(bases) < n_sample and attempts < n_sample * 100:
        cols = tuple(sorted(rng.choice(n, r, replace=False).tolist()))
        attempts += 1
        if cols in seen:
            continue
        if gf2_rank(H_X, list(cols)) == r:
            bases.append(list(cols))
            seen.add(cols)
    return bases


# ─────────────────────────────────────────────────────────────────────────────
# Circuit enumeration (BFS, Method B, original coords)
# ─────────────────────────────────────────────────────────────────────────────

def bfs_circuits(H_X, max_circuits=20000):
    """Enumerate all circuits via BFS from canonical fc using Method B decomposition."""
    basis_c = canonical_basis(H_X)
    non_b, fc_c = get_fc_for_basis(H_X, basis_c)
    visited = {}
    queue = []
    for j in non_b:
        c = fc_c[j]
        if c not in visited:
            visited[c] = len(visited)
            queue.append(c)
    head = 0
    while head < len(queue) and len(visited) < max_circuits:
        C = queue[head]; head += 1
        for j in non_b:
            if not (fc_c[j] & C):
                continue
            sym = C ^ fc_c[j]
            if not sym:
                continue
            for part in decompose_hx(H_X, sym):
                if part not in visited:
                    visited[part] = len(visited)
                    queue.append(part)
    return list(visited.keys()), len(visited) >= max_circuits


def circuits_for_code(H_X, n_thresh=14):
    """Use brute force for small n, BFS for larger."""
    n = H_X.shape[1]
    if n <= n_thresh:
        r = get_rank(H_X)
        circuits = []
        for size in range(2, n + 1):
            for cols in combinations(range(n), size):
                cols_list = list(cols)
                if gf2_rank(H_X, cols_list) < size:
                    minimal = all(
                        gf2_rank(H_X, cols_list[:i] + cols_list[i+1:]) == size - 1
                        for i in range(size)
                    )
                    if minimal:
                        circuits.append(frozenset(cols))
        return sorted(circuits, key=lambda c: (len(c), sorted(c))), False
    else:
        circs, trunc = bfs_circuits(H_X)
        return sorted(circs, key=lambda c: (len(c), sorted(c))), trunc


# ─────────────────────────────────────────────────────────────────────────────
# Stationary distribution
# ─────────────────────────────────────────────────────────────────────────────

def stationary_dist(P):
    N = P.shape[0]
    A = (P.T - np.eye(N)).astype(np.float64)
    A[-1, :] = 1.0
    b = np.zeros(N); b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Method A: build Chain 1 transition matrix on n-r fundamental circuits of B
# ─────────────────────────────────────────────────────────────────────────────

def build_P_method_A(non_basis, fc_B):
    """
    Build P on state space {fc_B[j] : j in non_basis} (Method A, fc-decomposition).
    From state fc_B[j]: eligible j' s.t. fc_B[j'] ∩ fc_B[j] != ∅.
    sym = fc_B[j] Δ fc_B[j'].
    parts = {fc_B[j''] : j'' in non_basis, j'' in sym}  ← non-basis positions of sym.
    Pick part uniformly.
    Returns P (M x M, M = len(non_basis)) and state ordering (list of j).
    """
    M = len(non_basis)
    nb_idx = {j: i for i, j in enumerate(non_basis)}
    P = np.zeros((M, M), dtype=np.float64)

    for i, j in enumerate(non_basis):
        fc_j = fc_B[j]
        eligible = [j2 for j2 in non_basis if fc_B[j2] & fc_j]
        n_elig = len(eligible)
        if n_elig == 0:
            P[i, i] = 1.0
            continue
        for j2 in eligible:
            if j2 == j:
                # sym = empty → stay
                P[i, i] += 1.0 / n_elig
                continue
            sym = fc_j ^ fc_B[j2]
            # fc-decomposition: non-basis positions in sym
            part_keys = [j3 for j3 in non_basis if j3 in sym]
            if not part_keys:
                P[i, i] += 1.0 / n_elig
                continue
            w = 1.0 / (n_elig * len(part_keys))
            for j3 in part_keys:
                P[i, nb_idx[j3]] += w

    row_sums = P.sum(axis=1, keepdims=True)
    P /= np.where(row_sums > 0, row_sums, 1.0)
    return P


# ─────────────────────────────────────────────────────────────────────────────
# Method B: build Chain 1 on all N circuits using arbitrary GF(2) decomposition
# ─────────────────────────────────────────────────────────────────────────────

def build_P_method_B(circuits, idx, H_X, non_basis, fc_B):
    """Build N x N Chain 1 matrix using Method B (arbitrary GF(2) decomposition)."""
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
            parts = decompose_hx(H_X, sym)
            if not parts:
                P[i, i] += 1.0 / n_elig
                continue
            w = 1.0 / (n_elig * len(parts))
            for part in parts:
                k = idx.get(part)
                if k is not None:
                    P[i, k] += w
    row_sums = P.sum(axis=1, keepdims=True)
    P /= np.where(row_sums > 0, row_sums, 1.0)
    return P


# ─────────────────────────────────────────────────────────────────────────────
# Core: compute pi_avg over (sampled) bases for one code
# ─────────────────────────────────────────────────────────────────────────────

def compute_piavg(H_X, label, max_bases_exact=500, n_sample=200, seed=42,
                  also_method_B=False):
    """
    Compute pi_avg using Method A (fc-decomposition per basis).
    If also_method_B: also compute for Method B (slow, only for small N).
    Returns dict with results.
    """
    n = H_X.shape[1]
    r = get_rank(H_X)
    print(f"\n{'='*65}")
    print(f"  {label}   n={n}  r={r}  n-r={n-r}")
    print(f"{'='*65}")

    # Enumerate circuits
    print(f"  Enumerating circuits...", end=" ", flush=True)
    circuits, trunc = circuits_for_code(H_X)
    N = len(circuits)
    idx = {c: i for i, c in enumerate(circuits)}
    wts = np.array([len(c) for c in circuits])
    print(f"N={N}{'  (BFS, may be incomplete)' if trunc else ''}")

    # Weight distribution
    weight_classes = sorted(set(wts))
    A_w = {w: int((wts == w).sum()) for w in weight_classes}
    print(f"  Weight classes: { {w: A_w[w] for w in weight_classes} }")

    # Enumerate or sample bases
    print(f"  Getting bases...", end=" ", flush=True)
    all_bases, _ = enumerate_bases(H_X)
    n_total_bases = len(all_bases)
    if n_total_bases <= max_bases_exact:
        bases = all_bases
        sampled = False
    else:
        pyrandom.seed(seed)
        bases = sample_bases(H_X, r, n_sample, seed)
        sampled = True
    print(f"{'~' if sampled else ''}{len(bases)} bases "
          f"{'(sampled from ' + str(n_total_bases) + '+)' if sampled else '(all)'}")

    B_count = len(bases)

    # ── Method A ──────────────────────────────────────────────────────────────
    pi_matrix_A = np.zeros((B_count, N), dtype=np.float64)
    for b_idx, basis in enumerate(bases):
        non_basis, fc_B = get_fc_for_basis(H_X, basis)
        P_local = build_P_method_A(non_basis, fc_B)
        pi_local = stationary_dist(P_local)  # (n-r,) array
        for li, j in enumerate(non_basis):
            k = idx.get(fc_B[j])
            if k is not None:
                pi_matrix_A[b_idx, k] = pi_local[li]

    pi_avg_A = pi_matrix_A.mean(axis=0)

    # ── Method B (optional, small codes only) ─────────────────────────────────
    pi_avg_B = None
    if also_method_B:
        pi_matrix_B = np.zeros((B_count, N), dtype=np.float64)
        for b_idx, basis in enumerate(bases):
            non_basis, fc_B = get_fc_for_basis(H_X, basis)
            P_B = build_P_method_B(circuits, idx, H_X, non_basis, fc_B)
            pi_matrix_B[b_idx] = stationary_dist(P_B)
        pi_avg_B = pi_matrix_B.mean(axis=0)

    # ── Report ────────────────────────────────────────────────────────────────
    def report_piavg(pi_avg, method_label):
        print(f"\n  pi_avg [{method_label}]:")
        print(f"  {'wt':>4}  {'A_w':>6}  {'pi_avg(w)':>14}  {'A_w*pi_avg':>12}  {'as fraction'}")
        print("  " + "-" * 62)
        total_check = 0.0
        piavg_per_weight = {}
        for w in weight_classes:
            mask = (wts == w)
            pav = float(pi_avg[mask].mean()) if mask.any() else 0.0
            mass = float(A_w[w] * pav)
            total_check += mass
            # find simple fraction
            from fractions import Fraction
            frac = Fraction(pav).limit_denominator(10000)
            piavg_per_weight[w] = pav
            print(f"  {w:>4}  {A_w[w]:>6}  {pav:>14.8f}  {mass:>12.6f}  ≈ {frac}")
        print(f"  {'TOTAL':>4}  {N:>6}  {'':>14}  {total_check:>12.6f}")

        if np.std(pi_avg) > 1e-10:
            r_val, _ = pearsonr(pi_avg, wts.astype(float))
            print(f"  Pearson r(pi_avg, wt) = {r_val:.6f}")

        # Ratios between weight classes
        if len(weight_classes) >= 2:
            print(f"  Ratios pi_avg(w1)/pi_avg(w2):")
            for i, w1 in enumerate(weight_classes):
                for w2 in weight_classes[i+1:]:
                    ratio = piavg_per_weight[w1] / piavg_per_weight[w2] if piavg_per_weight[w2] > 1e-15 else float('inf')
                    from fractions import Fraction
                    frac = Fraction(ratio).limit_denominator(1000)
                    print(f"    pi({w1})/pi({w2}) = {ratio:.6f} ≈ {frac}")
        return piavg_per_weight

    pw_A = report_piavg(pi_avg_A, "Method A")
    pw_B = None
    if pi_avg_B is not None:
        pw_B = report_piavg(pi_avg_B, "Method B")
        print(f"  Method A vs B agree: {'YES' if np.allclose(pi_avg_A, pi_avg_B, atol=1e-6) else 'NO'}")

    return {
        "label": label, "n": n, "r": r, "N": N,
        "circuits": circuits, "wts": wts,
        "weight_classes": weight_classes, "A_w": A_w,
        "pi_avg_A": pi_avg_A, "pi_avg_B": pi_avg_B,
        "pw_A": pw_A, "pw_B": pw_B,
        "B_count": B_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Candidate formula testing
# ─────────────────────────────────────────────────────────────────────────────

def test_candidates(results):
    """Test 8 candidate closed forms across all codes."""
    print(f"\n{'='*75}")
    print("CANDIDATE FORMULA TESTING")
    print(f"{'='*75}")
    print("Testing pi_avg(C) ∝ f(w, n, r). Predicted ratio = f(w1)/f(w2).")
    print("Observed ratio = pi_avg(w1)/pi_avg(w2).")

    # Collect all (label, n, r, w1, w2, observed_ratio) tuples
    obs_data = []
    for res in results:
        n, r = res["n"], res["r"]
        pw = res["pw_A"]
        wcs = res["weight_classes"]
        for i, w1 in enumerate(wcs):
            for w2 in wcs[i+1:]:
                if pw[w2] > 1e-15:
                    obs_ratio = pw[w1] / pw[w2]
                    obs_data.append((res["label"], n, r, w1, w2, obs_ratio))

    # Define candidate functions f(w, n, r) → value
    candidates = [
        ("1/w",             lambda w, n, r: 1.0/w),
        ("1/(w-1)",         lambda w, n, r: 1.0/(w-1) if w>1 else float('inf')),
        ("1/(w*(w-1))",     lambda w, n, r: 1.0/(w*(w-1)) if w>1 else float('inf')),
        ("1/(w*(n-w))",     lambda w, n, r: 1.0/(w*(n-w)) if w<n else float('inf')),
        ("(n-w)/w",         lambda w, n, r: (n-w)/w),
        ("C(n-w,r-w+1)",    lambda w, n, r: float(comb(n-w, r-w+1)) if r>=w-1 and n-w>=r-w+1 else 0.0),
        ("1/C(n,w)",        lambda w, n, r: 1.0/comb(n, w)),
        ("C(n-w,r)/C(n,r)", lambda w, n, r: comb(n-w, r)/comb(n, r) if n-w>=r else 0.0),
    ]

    print()
    # Header
    col_names = [c[0] for c in candidates]
    print(f"  {'Code':<12} {'n':>3} {'r':>3} {'w1':>3} {'w2':>3} {'obs':>9}", end="")
    for name in col_names:
        print(f" {name[:9]:>9}", end="")
    print()
    print("  " + "-" * (40 + 10*len(candidates)))

    # Track which candidates survive all tests
    all_pass = {name: True for name, _ in candidates}
    tol = 0.02  # relative tolerance for "match"

    for (label, n, r, w1, w2, obs_ratio) in obs_data:
        print(f"  {label:<12} {n:>3} {r:>3} {w1:>3} {w2:>3} {obs_ratio:>9.4f}", end="")
        for name, f in candidates:
            fval1 = f(w1, n, r)
            fval2 = f(w2, n, r)
            if fval2 > 1e-15 and fval1 >= 0:
                pred = fval1 / fval2
                err = abs(pred - obs_ratio) / obs_ratio
                match = err < tol
                if not match:
                    all_pass[name] = False
                print(f" {'✓' if match else 'x'}{pred:>7.3f}", end="")
            else:
                print(f" {'?':>9}", end="")
                all_pass[name] = False
        print()

    print(f"\n  Survivors (match all observed ratios within {tol*100:.0f}%):")
    survivors = [name for name, ok in all_pass.items() if ok]
    if survivors:
        for name in survivors:
            print(f"    ✓  {name}")
    else:
        print("    None — check if all codes have been run")

    return survivors


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: matroid-specific quantities
# ─────────────────────────────────────────────────────────────────────────────

def matroid_quantities(res, n_sample_bases=None):
    """
    Q1: avg #{bases containing exactly w-1 elements of C} per weight w
        = avg #{bases B : C is fundamental for B} = avg "activity count"
    Q2: avg #{fc_B[j] of weight w} per basis
    """
    H_X = res.get("H_X")
    if H_X is None:
        return

    n, r = res["n"], res["r"]
    circuits = res["circuits"]
    wts = res["wts"]
    weight_classes = res["weight_classes"]

    print(f"\n  Matroid-specific quantities for {res['label']}:")

    # Sample some bases
    all_bases, _ = enumerate_bases(H_X)
    bases = all_bases[:min(200, len(all_bases))]
    B_count = len(bases)

    # Q1: for each circuit C, count #{B : C = fc_B[j] for some j}
    activity = np.zeros(len(circuits), dtype=float)
    q2_counts = {w: 0.0 for w in weight_classes}

    for basis in bases:
        non_basis, fc_B = get_fc_for_basis(H_X, basis)
        fc_set = set()
        for j in non_basis:
            c = fc_B[j]
            k = res["idx"].get(c)
            if k is not None:
                activity[k] += 1.0
                fc_set.add(len(c))
        for j in non_basis:
            w = len(fc_B[j])
            if w in q2_counts:
                q2_counts[w] += 1.0

    activity /= B_count

    print(f"  {'wt':>4}  {'A_w':>6}  {'Q1(avg_activity)':>18}  {'Q2(avg_fc_count)':>18}  {'pi_avg':>12}  {'pi_avg*Q1':>12}")
    print("  " + "-" * 75)
    for w in weight_classes:
        mask = (wts == w)
        q1 = float(activity[mask].mean()) if mask.any() else 0.0
        q2 = q2_counts.get(w, 0.0) / B_count
        pav = res["pw_A"].get(w, 0.0)
        print(f"  {w:>4}  {res['A_w'][w]:>6}  {q1:>18.6f}  {q2:>18.6f}  {pav:>12.8f}  {pav*q1:>12.8f}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

results = []

# [[6,2,2]]
H6 = np.array([[1,1,1,1,0,0],[0,0,1,1,1,1]], dtype=np.uint8)
res6 = compute_piavg(H6, "[[6,2,2]]", also_method_B=True)
res6["H_X"] = H6
res6["idx"] = {c: i for i, c in enumerate(res6["circuits"])}
results.append(res6)

# Toric L=2
H_t2 = build_toric_hx(2)
res_t2 = compute_piavg(H_t2, "Toric L=2", also_method_B=True)
res_t2["H_X"] = H_t2
res_t2["idx"] = {c: i for i, c in enumerate(res_t2["circuits"])}
results.append(res_t2)

# Toric L=3
H_t3 = build_toric_hx(3)
res_t3 = compute_piavg(H_t3, "Toric L=3", max_bases_exact=300, n_sample=200, seed=42)
res_t3["H_X"] = H_t3
res_t3["idx"] = {c: i for i, c in enumerate(res_t3["circuits"])}
results.append(res_t3)

# BB(3,3)
H_bb, _ = bb_code(3, 3, [(0,0),(1,0),(0,1)], [(0,0),(2,0),(0,2)])
res_bb = compute_piavg(H_bb, "BB(3,3)", max_bases_exact=300, n_sample=200, seed=42)
res_bb["H_X"] = H_bb
res_bb["idx"] = {c: i for i, c in enumerate(res_bb["circuits"])}
results.append(res_bb)

# ── Candidate formula testing ─────────────────────────────────────────────────
survivors = test_candidates(results)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print("SUMMARY TABLE")
print(f"{'Code':<14} {'n':>3} {'r':>3} {'w':>4} {'A_w':>6} {'pi_avg(w)':>14}  best fit")
print("-" * 85)
for res in results:
    for w in res["weight_classes"]:
        pav = res["pw_A"][w]
        from fractions import Fraction
        frac = str(Fraction(pav).limit_denominator(500))
        print(f"{res['label']:<14} {res['n']:>3} {res['r']:>3} {w:>4} "
              f"{res['A_w'][w]:>6} {pav:>14.8f}  {frac}")

# ── Matroid quantities (Tasks 5) ───────────────────────────────────────────────
print(f"\n{'='*75}")
print("TASK 5 — Matroid quantities Q1, Q2 vs pi_avg")
for res in results[:3]:   # skip BB(3,3) if slow
    matroid_quantities(res)
