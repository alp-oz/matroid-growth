"""
codeword_walk.py — Chain 3: Codeword Walk on the full codeword space.

Chain 3  Codeword Walk
  State:  any nonzero codeword c ∈ ker(H_X)
  Target: exactly uniform over ker(H_X) \\ {0}
  Output: A(z) weight enumerator estimate

Key difference from Chain 1: do NOT decompose the XOR result into circuits.
Keep the full vector (c + fc[j]) % 2 as the new state.

Usage
-----
  from codeword_walk import preprocess, run_chain3
  results = run_chain3(H_X)
"""

import numpy as np
import random
from scipy.optimize import curve_fit


# ══════════════════════════════════════════════════════════════════════════════
# GF(2) preprocessing  (CSS codes)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(H_X):
    """
    GF(2) full RREF on H_X. Returns (basis, non_basis, r, fc).
    fc[j] : fundamental circuit of non-basis element j.
    """
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
    for j, fvec in fc.items():
        chk = (H_X.astype(np.int32) @ fvec.astype(np.int32)) % 2
        assert np.all(chk == 0), f"fc[{j}] not in ker(H_X)"
    return basis, non_basis, r, fc


# ══════════════════════════════════════════════════════════════════════════════
# Chain 3 step
# ══════════════════════════════════════════════════════════════════════════════

def chain3_step(c, fc, non_basis):
    """
    One step of Chain 3 (Codeword Walk). Stationary: uniform over ker(H_X)\\{0}.
    Pick j uniformly, XOR with fc[j]; stay if zero.
    """
    j = random.choice(non_basis)
    c_new = (c.astype(np.int32) + fc[j].astype(np.int32)) % 2
    c_new = c_new.astype(np.uint8)
    return c.copy() if not c_new.any() else c_new


# ══════════════════════════════════════════════════════════════════════════════
# Exact analysis (small codes)
# ══════════════════════════════════════════════════════════════════════════════

def enumerate_all_codewords(fc, non_basis, n):
    """All 2^k - 1 nonzero codewords."""
    generators = [fc[j] for j in non_basis]
    k = len(generators)
    codewords = []
    for mask in range(1, 2 ** k):
        c = np.zeros(n, dtype=np.uint8)
        for i in range(k):
            if mask & (1 << i):
                c = (c + generators[i]) % 2
        codewords.append(c.copy())
    return codewords


def build_chain3_matrix(codewords, fc, non_basis):
    """Build exact N×N transition matrix of Chain 3."""
    N = len(codewords)
    idx = {tuple(c): i for i, c in enumerate(codewords)}
    P = np.zeros((N, N), dtype=np.float64)
    p0 = 1.0 / len(non_basis)
    for i, c in enumerate(codewords):
        for j in non_basis:
            c_new = (c.astype(np.int32) + fc[j].astype(np.int32)) % 2
            c_new = c_new.astype(np.uint8)
            if not c_new.any():
                P[i, i] += p0
            else:
                k_idx = idx.get(tuple(c_new))
                if k_idx is not None:
                    P[i, k_idx] += p0
    row_sums = P.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, P / row_sums, P)


def spectral_gap(P):
    eigs = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    return float(1.0 - eigs[1]) if len(eigs) > 1 else 1.0


def stationary_dist(P):
    N = P.shape[0]
    A = (P.T - np.eye(N)).astype(np.float64)
    A[-1, :] = 1.0
    b = np.zeros(N); b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


def tv_distance_curve(P, pi, max_t=300, epsilon=0.25):
    N = P.shape[0]
    Pt = np.eye(N, dtype=np.float64)
    pi_row = pi[np.newaxis, :]
    tv_max = np.zeros(max_t)
    t_mix = None
    for t in range(max_t):
        Pt = Pt @ P
        tv = 0.5 * np.abs(Pt - pi_row).sum(axis=1)
        tv_max[t] = float(tv.max())
        if t_mix is None and tv_max[t] < epsilon:
            t_mix = t + 1
    return tv_max, t_mix


def weight_enumerator(weights, k_dim, T):
    """Estimate A(z) from Chain 3 trajectory (uniform stationary)."""
    n_codewords = 2 ** k_dim - 1
    unique, counts = np.unique(weights, return_counts=True)
    return {int(w): float(n_codewords * cnt / T) for w, cnt in zip(unique, counts)}


# ══════════════════════════════════════════════════════════════════════════════
# Empirical gap estimator
# ══════════════════════════════════════════════════════════════════════════════

def autocorr_gap(step_fn, fc, non_basis, T=60000, burn_in=6000, seed=42):
    """Estimate spectral gap from autocorrelation time of element indicators."""
    random.seed(seed)
    rng = np.random.default_rng(seed)
    n = len(list(fc.values())[0])
    test_elems = rng.choice(n, size=min(16, n), replace=False).tolist()
    c = fc[random.choice(non_basis)].copy()
    for _ in range(burn_in):
        c = step_fn(c, fc, non_basis)
    traces = np.zeros((len(test_elems), T))
    for t in range(T):
        c = step_fn(c, fc, non_basis)
        for i, e in enumerate(test_elems):
            traces[i, t] = float(c[e])
    taus = []
    max_lag = min(600, T // 10)
    for row in traces:
        x = row - row.mean()
        var = np.var(x)
        if var < 1e-10:
            continue
        rho = np.array([np.mean(x[:T - lag] * x[lag:]) / var
                        for lag in range(max_lag)])
        pos = rho > 0.05
        if pos.sum() < 3:
            taus.append(1.0); continue
        lags_fit = np.where(pos)[0]
        try:
            popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau),
                                lags_fit, rho[pos], p0=[5.0],
                                bounds=(0.1, 4000))
            taus.append(float(popt[0]))
        except Exception:
            taus.append(float(0.5 + np.sum(rho[1:])))
    if not taus:
        return np.inf, 0.0
    tau = float(np.mean(taus))
    return tau, 1.0 / tau


# ══════════════════════════════════════════════════════════════════════════════
# Top-level runner
# ══════════════════════════════════════════════════════════════════════════════

def run_chain3(H_X, T_empirical=80000, burn_in=8000,
               max_t_tv=300, max_codewords=50000, seed=42, label=""):
    """Run Chain 3 (Codeword Walk) on the CSS code defined by H_X."""
    random.seed(seed); np.random.seed(seed)
    tag = f"[{label}] " if label else ""
    basis, non_basis, r, fc = preprocess(H_X)
    n = H_X.shape[1]
    k_dim = len(non_basis)
    N_cw = 2 ** k_dim - 1
    print(f"{tag}n={n}  r={r}  k={k_dim}  #codewords={N_cw}")
    result = {"n": n, "r": r, "k": k_dim, "N_codewords": N_cw}

    if N_cw <= max_codewords:
        codewords = enumerate_all_codewords(fc, non_basis, n)
        P3 = build_chain3_matrix(codewords, fc, non_basis)
        pi3 = stationary_dist(P3)
        result["gap_exact"] = spectral_gap(P3)
        result["t_mix"] = tv_distance_curve(P3, pi3, max_t=max_t_tv)[1]
        print(f"{tag}  gap (exact) = {result['gap_exact']:.5f}")

    tau3, d3 = autocorr_gap(chain3_step, fc, non_basis,
                             T=T_empirical, burn_in=burn_in, seed=seed)
    result["tau_empirical"] = tau3
    result["gap_empirical"] = d3
    print(f"{tag}  gap (empirical) = {d3:.5f}  (tau={tau3:.1f})")

    random.seed(seed + 1)
    c = fc[random.choice(non_basis)].copy()
    for _ in range(burn_in):
        c = chain3_step(c, fc, non_basis)
    weights = np.zeros(T_empirical, dtype=int)
    for t in range(T_empirical):
        c = chain3_step(c, fc, non_basis)
        weights[t] = int(c.sum())
    result["A_z"] = weight_enumerator(weights, k_dim, T_empirical)
    print(f"{tag}  A(z): " +
          "  ".join(f"A_{w}~{v:.1f}" for w, v in sorted(result["A_z"].items())))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from collections import Counter
    H_X = np.array([[1,1,1,1,0,0],[0,0,1,1,1,1]], dtype=np.uint8)
    _, non_basis, r, fc = preprocess(H_X)
    codewords = enumerate_all_codewords(fc, non_basis, 6)
    A = dict(Counter(int(c.sum()) for c in codewords))
    assert A == {2: 3, 3: 8, 4: 3, 6: 1}, f"Wrong A(z): {A}"
    print("Chain 3 self-test passed: A(z) exact enumeration correct")
