"""
coset_descent_chain.py — Chain 4: Biased Codeword Walk for coset descent.

Chain 4  Biased Codeword Walk
  State:  any nonzero codeword c ∈ ker(H_X)
  Target: pi(c) ∝ q^{wt(c)}  — concentrates on low-weight codewords
  Output: d_Z estimate, A_logical(z) per logical coset

Usage
-----
  from coset_descent_chain import run_chain4
  result = run_chain4(H_X, H_Z, q=0.3)
"""

import numpy as np
import random


# ══════════════════════════════════════════════════════════════════════════════
# GF(2) preprocessing  (CSS codes)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(H_X):
    """GF(2) full RREF on H_X. Returns (basis, non_basis, r, fc)."""
    m, n = H_X.shape
    mat = H_X.astype(np.uint8).copy()
    pivot_cols = []
    cur_row = 0
    for col in range(n):
        piv = None
        for row in range(cur_row, m):
            if mat[row, col] == 1:
                piv = row; break
        if piv is None:
            continue
        if piv != cur_row:
            mat[[cur_row, piv]] = mat[[piv, cur_row]]
        for row in range(m):
            if row != cur_row and mat[row, col] == 1:
                mat[row] = (mat[row] + mat[cur_row]) % 2
        pivot_cols.append(col)
        cur_row += 1
    r = len(pivot_cols)
    basis = pivot_cols
    non_basis = [j for j in range(n) if j not in set(basis)]
    fc = {}
    for j in non_basis:
        vec = np.zeros(n, dtype=np.uint8)
        vec[j] = 1
        for i, b in enumerate(basis):
            if mat[i, j] == 1:
                vec[b] = 1
        fc[j] = vec
    return basis, non_basis, r, fc


# ══════════════════════════════════════════════════════════════════════════════
# Coset / logical labeling
# ══════════════════════════════════════════════════════════════════════════════

def compute_logical_basis(H_X, H_Z):
    """
    Find a basis {Z_1,...,Z_{k_logical}} for logical Z-operators.
    Pass H_Z for Z-stabilizers to find X-logical operators.
    (Swap args to find Z-logicals from X-stabilizers.)
    """
    _, non_basis, _, fc = preprocess(H_X)
    k = len(non_basis)

    stab_mat = np.array([[int(row[j]) for j in non_basis]
                         for row in H_Z], dtype=np.uint8)
    m_stab = stab_mat.shape[0]
    pivot_row = 0
    for col in range(k):
        piv = None
        for r in range(pivot_row, m_stab):
            if stab_mat[r, col]:
                piv = r; break
        if piv is None:
            continue
        stab_mat[[pivot_row, piv]] = stab_mat[[piv, pivot_row]]
        for r in range(m_stab):
            if r != pivot_row and stab_mat[r, col]:
                stab_mat[r] = (stab_mat[r] + stab_mat[pivot_row]) % 2
        pivot_row += 1
    stab_rref = list(stab_mat[:pivot_row])

    echelon = list(stab_rref)
    logicals = []
    for i, j in enumerate(non_basis):
        e_i = np.zeros(k, dtype=np.uint8)
        e_i[i] = 1
        v = e_i.copy()
        for b in echelon:
            lead = int(np.where(b)[0][0])
            if v[lead]:
                v = (v + b) % 2
        if not v.any():
            continue
        logicals.append(fc[j].copy())
        new_lead = int(np.where(v)[0][0])
        for idx in range(len(echelon)):
            if echelon[idx][new_lead]:
                echelon[idx] = (echelon[idx] + v) % 2
        echelon.append(v.copy())
    return logicals


def coset_label(c, logicals):
    """Tuple of (dot(c, Z_i) % 2) for each logical Z_i."""
    return tuple(int(np.dot(c, z)) % 2 for z in logicals)


# ══════════════════════════════════════════════════════════════════════════════
# Chain 4 step
# ══════════════════════════════════════════════════════════════════════════════

def chain4_step(c, fc, non_basis, q):
    """
    One step of Chain 4. Proposal: pick j, XOR with fc[j].
    Accept if weight decreases or stays; accept with prob q^{delta_w} otherwise.
    Stationary: pi(c) ∝ q^{wt(c)}.
    """
    j = random.choice(non_basis)
    c_new = (c.astype(np.int32) + fc[j].astype(np.int32)) % 2
    c_new = c_new.astype(np.uint8)
    if not c_new.any():
        return c.copy()
    delta_w = int(c_new.sum()) - int(c.sum())
    if delta_w <= 0:
        return c_new
    if random.random() < q ** delta_w:
        return c_new
    return c.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Top-level runner
# ══════════════════════════════════════════════════════════════════════════════

def run_chain4(H_X, H_Z, q=0.3, T_max=200000,
               stability_window=10000, seed=42):
    """
    Run Chain 4 (Biased Codeword Walk) to estimate d_Z and A_logical(z).

    Tracks:
      T_cover   — first step at which all non-zero cosets have been visited
      T_converge — first step >= T_cover at which no min-weight decreased
                   for stability_window steps

    Returns dict with: k_logical, n_cosets, d_Z, A_logical_z, coverage,
                       n_visited, T_cover, T_converge, q.
    """
    random.seed(seed)
    basis, non_basis, r, fc = preprocess(H_X)
    n = H_X.shape[1]
    logicals = compute_logical_basis(H_Z, H_X)   # X-logicals for Z-coset labeling
    k_logical = len(logicals)
    n_nonzero_cosets = 2 ** k_logical - 1
    zero_label = tuple(0 for _ in range(k_logical))

    if k_logical == 0:
        return {"k_logical": 0, "n_cosets": 0, "d_Z": None,
                "A_logical_z": {}, "coverage": 0.0, "n_visited": 0,
                "T_cover": -1, "T_converge": -1, "q": q}

    c = fc[random.choice(non_basis)].copy()
    min_wt = {}
    first_visit = {}
    T_cover = -1
    T_converge = -1
    last_decrease_step = 0

    for t in range(1, T_max + 1):
        c = chain4_step(c, fc, non_basis, q)

        lab = coset_label(c, logicals)
        wt = int(c.sum())
        if lab != zero_label:
            if lab not in first_visit:
                first_visit[lab] = t
            if lab not in min_wt or wt < min_wt[lab]:
                min_wt[lab] = wt
                last_decrease_step = t

        if T_cover == -1 and len(first_visit) == n_nonzero_cosets:
            T_cover = t
        if (T_cover != -1 and T_converge == -1
                and t - last_decrease_step >= stability_window):
            T_converge = t
            break

    A_logical_z = {}
    for wt in min_wt.values():
        A_logical_z[wt] = A_logical_z.get(wt, 0) + 1

    return {
        "k_logical":    k_logical,
        "n_cosets":     n_nonzero_cosets,
        "d_Z":          min(min_wt.values()) if min_wt else None,
        "A_logical_z":  A_logical_z,
        "coverage":     len(first_visit) / n_nonzero_cosets,
        "n_visited":    len(first_visit),
        "T_cover":      T_cover,
        "T_converge":   T_converge,
        "q":            q,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    from toric_code import build_toric_hx, build_toric_hz
    H_X = build_toric_hx(2)
    H_Z = build_toric_hz(2)
    res = run_chain4(H_X, H_Z, q=0.3, T_max=50000)
    assert res["d_Z"] == 2, f"Expected d_Z=2, got {res['d_Z']}"
    assert res["A_logical_z"] == {2: 2, 4: 1}, f"Wrong A_logical: {res['A_logical_z']}"
    print(f"Chain 4 self-test passed: d_Z={res['d_Z']}, "
          f"A_logical={res['A_logical_z']}, coverage={res['coverage']:.0%}")
