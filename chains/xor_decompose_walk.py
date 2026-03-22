"""
xor_decompose_walk.py — Chain 1: XOR-Decompose Walk on the circuit space.

Works on ANY binary matroid:
  CSS codes   — pass H_X (m × n parity check matrix over GF(2))
  PA matroids — pass M already in [I | A] form (r × n); preprocessing is a no-op

Chain 1  XOR-Decompose Walk
  State:  a circuit (minimum-weight nonzero codeword)
  Target: non-uniform over circuits
  Step:   pick eligible j (fc[j] ∩ C ≠ ∅), sym-diff S = C ⊕ fc[j],
          decompose S into fundamental circuits, pick one uniformly.

Usage
-----
  from xor_decompose_walk import preprocess, run_chain1
  results = run_chain1(H_X)
  results = run_chain1(M_rref, rank=r)   # PA matroid
"""

import numpy as np
import random
from scipy.optimize import curve_fit


# ══════════════════════════════════════════════════════════════════════════════
# GF(2) preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(H, rank=None):
    """
    GF(2) full RREF on binary matrix H.

    For CSS codes:    pass H_X, leave rank=None.
    For PA matroids:  pass M in [I|A] form and set rank=r.

    Returns (basis, non_basis, r, fc).
    fc[j] : np.array(n, uint8) — fundamental circuit of non-basis element j.
    """
    m, n = H.shape
    mat = H.astype(np.uint8).copy()
    pivot_cols = []
    cur_row = 0

    for col in range(n):
        piv = None
        for row in range(cur_row, m):
            if mat[row, col] == 1:
                piv = row
                break
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
        chk = (H.astype(np.int32) @ fvec.astype(np.int32)) % 2
        assert np.all(chk == 0), f"fc[{j}] not in ker(H)"

    return basis, non_basis, r, fc


# ══════════════════════════════════════════════════════════════════════════════
# Circuit decomposition
# ══════════════════════════════════════════════════════════════════════════════

def decompose_into_fundamental(s, fc, non_basis):
    """Decompose s ∈ ker(H) as GF(2) sum of fundamental circuits."""
    return [fc[j].copy() for j in non_basis if s[j] == 1]


# ══════════════════════════════════════════════════════════════════════════════
# Chain 1 step and circuit enumeration
# ══════════════════════════════════════════════════════════════════════════════

def chain1_step(c, fc, non_basis):
    """One step of Chain 1 (XOR-Decompose Walk)."""
    eligible = [j for j in non_basis if int(np.dot(fc[j], c)) % 2 == 1]
    if not eligible:
        return c.copy()
    j = random.choice(eligible)
    s = (c.astype(np.int32) + fc[j].astype(np.int32)) % 2
    s = s.astype(np.uint8)
    if not s.any():
        return c.copy()
    parts = decompose_into_fundamental(s, fc, non_basis)
    return random.choice(parts) if parts else c.copy()


def enumerate_circuits(fc, non_basis, n, max_circuits=10000):
    """BFS over the circuit graph starting from all fundamental circuits."""
    visited = {}
    queue = []
    for j in non_basis:
        t = tuple(fc[j])
        if t not in visited:
            visited[t] = len(visited)
            queue.append(fc[j].copy())
    head = 0
    while head < len(queue):
        if len(visited) > max_circuits:
            raise RuntimeError(
                f"Circuit count exceeded {max_circuits}.")
        c = queue[head]; head += 1
        eligible = [j for j in non_basis if int(np.dot(fc[j], c)) % 2 == 1]
        for j in eligible:
            s = (c.astype(np.int32) + fc[j].astype(np.int32)) % 2
            s = s.astype(np.uint8)
            if not s.any():
                continue
            for part in decompose_into_fundamental(s, fc, non_basis):
                t = tuple(part)
                if t not in visited:
                    visited[t] = len(visited)
                    queue.append(part.copy())
    return queue


# ══════════════════════════════════════════════════════════════════════════════
# Transition matrix (Chain 1) — also used by mh_walk.py
# ══════════════════════════════════════════════════════════════════════════════

def build_transition_matrix(circuits, fc, non_basis):
    """Build N×N row-stochastic transition matrix P of Chain 1."""
    N = len(circuits)
    idx = {tuple(c): i for i, c in enumerate(circuits)}
    P = np.zeros((N, N), dtype=np.float64)
    for i, c in enumerate(circuits):
        eligible = [j for j in non_basis if int(np.dot(fc[j], c)) % 2 == 1]
        if not eligible:
            P[i, i] = 1.0; continue
        n_el = len(eligible)
        for j in eligible:
            s = (c.astype(np.int32) + fc[j].astype(np.int32)) % 2
            s = s.astype(np.uint8)
            if not s.any():
                P[i, i] += 1.0 / n_el; continue
            parts = decompose_into_fundamental(s, fc, non_basis)
            if not parts:
                P[i, i] += 1.0 / n_el; continue
            w = 1.0 / (n_el * len(parts))
            for part in parts:
                k = idx.get(tuple(part))
                if k is not None:
                    P[i, k] += w
    for i in range(N):
        rs = P[i].sum()
        if rs > 1e-12:
            P[i] /= rs
        else:
            P[i, i] = 1.0
    return P


# ══════════════════════════════════════════════════════════════════════════════
# Analysis utilities
# ══════════════════════════════════════════════════════════════════════════════

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


def autocorr_gap(step_fn, fc, non_basis, T=60000, burn_in=6000, seed=42):
    """Empirical spectral gap via autocorrelation time of element indicators."""
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
    return float(np.mean(taus)), 1.0 / float(np.mean(taus))


# ══════════════════════════════════════════════════════════════════════════════
# Top-level runner
# ══════════════════════════════════════════════════════════════════════════════

def run_chain1(H, rank=None, T_empirical=60000, burn_in=6000,
               max_t_tv=300, max_circuits=2000, seed=42, label=""):
    """Run Chain 1 on the binary matroid defined by H."""
    random.seed(seed); np.random.seed(seed)
    tag = f"[{label}] " if label else ""
    basis, non_basis, r, fc = preprocess(H, rank=rank)
    n = H.shape[1]
    print(f"{tag}n={n}  r={r}  |non_basis|={len(non_basis)}")
    result = {"n": n, "r": r}
    try:
        circuits = enumerate_circuits(fc, non_basis, n, max_circuits=max_circuits)
        N = len(circuits)
        result["N"] = N
        print(f"{tag}  N = {N} circuits")
        P1 = build_transition_matrix(circuits, fc, non_basis)
        pi1 = stationary_dist(P1)
        result["gap_exact"] = spectral_gap(P1)
        result["pi_uniform"] = bool(np.max(np.abs(pi1 - 1.0/N)) < 1e-3)
        result["tv_max"], result["t_mix"] = tv_distance_curve(P1, pi1, max_t=max_t_tv)
        print(f"{tag}  gap (exact) = {result['gap_exact']:.5f}")
        result["_P"] = P1   # passed to mh_walk if needed
        result["_circuits"] = circuits
        result["_fc"] = fc
        result["_non_basis"] = non_basis
    except RuntimeError as e:
        print(f"{tag}  {e}")
        result["N"] = f">{max_circuits}"
    tau, d = autocorr_gap(chain1_step, fc, non_basis,
                          T=T_empirical, burn_in=burn_in, seed=seed)
    result["tau_empirical"] = tau
    result["gap_empirical"] = d
    print(f"{tag}  gap (empirical) = {d:.5f}  (tau={tau:.1f})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    H_X = np.array([[1,1,1,1,0,0],[0,0,1,1,1,1]], dtype=np.uint8)
    basis, non_basis, r, fc = preprocess(H_X)
    assert r == 2 and basis == [0,2] and non_basis == [1,3,4,5]
    circuits = enumerate_circuits(fc, non_basis, 6)
    assert len(circuits) == 4, f"expected 4 circuits, got {len(circuits)}"
    print(f"Chain 1 self-test passed: N={len(circuits)} circuits")
