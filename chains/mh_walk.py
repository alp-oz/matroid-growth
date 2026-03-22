"""
mh_walk.py — Chain 2: MH-Corrected Walk on the circuit space.

Chain 2  Min-Metropolis Walk
  State:  a circuit (minimum-weight nonzero codeword)
  Target: exactly uniform over all circuits
  Note:   requires building the full N×N matrix of Chain 1 — only feasible
          for small codes (N ≲ 2000).

Usage
-----
  from mh_walk import run_chain2
  results = run_chain2(H_X)
"""

import numpy as np
import random

from chains.xor_decompose_walk import (
    preprocess, enumerate_circuits, build_transition_matrix,
    spectral_gap, stationary_dist, tv_distance_curve, autocorr_gap,
)


# ══════════════════════════════════════════════════════════════════════════════
# Chain 2 — Min-Metropolis correction
# ══════════════════════════════════════════════════════════════════════════════

def apply_min_metropolis(P):
    """
    Min-Metropolis rule: P_MH[i,j] = min(P[i,j], P[j,i]) for i≠j.
    Targets exactly uniform stationary distribution.
    """
    P_mh = np.minimum(P, P.T)
    np.fill_diagonal(P_mh, 0.0)
    np.fill_diagonal(P_mh, np.clip(1.0 - P_mh.sum(axis=1), 0.0, None))
    return P_mh


def make_chain2_step(P_mh, circuits):
    """Return a step function sampling from row i of P_MH."""
    idx = {tuple(c): i for i, c in enumerate(circuits)}

    def step(c, *_):
        i = idx[tuple(c)]
        j = np.random.choice(len(circuits), p=P_mh[i])
        return circuits[j].copy()

    return step


# ══════════════════════════════════════════════════════════════════════════════
# Top-level runner
# ══════════════════════════════════════════════════════════════════════════════

def run_chain2(H, rank=None, max_t_tv=300, max_circuits=2000, seed=42, label=""):
    """
    Run Chain 2 (MH-corrected) on the binary matroid defined by H.
    Returns dict with gap_chain1, gap_MH, ratio, N, n.
    """
    random.seed(seed); np.random.seed(seed)
    tag = f"[{label}] " if label else ""

    basis, non_basis, r, fc = preprocess(H, rank=rank)
    n = H.shape[1]
    print(f"{tag}n={n}  r={r}")

    circuits = enumerate_circuits(fc, non_basis, n, max_circuits=max_circuits)
    N = len(circuits)
    print(f"{tag}N={N} circuits")

    P1 = build_transition_matrix(circuits, fc, non_basis)
    P2 = apply_min_metropolis(P1)

    pi1 = stationary_dist(P1)
    pi2 = stationary_dist(P2)

    gap1 = spectral_gap(P1)
    gap2 = spectral_gap(P2)
    ratio = gap1 / gap2 if gap2 > 0 else float("inf")

    tv1, tmix1 = tv_distance_curve(P1, pi1, max_t=max_t_tv)
    tv2, tmix2 = tv_distance_curve(P2, pi2, max_t=max_t_tv)

    print(f"{tag}  gap Chain 1 = {gap1:.5f}  (t_mix={tmix1})")
    print(f"{tag}  gap Chain 2 = {gap2:.5f}  (t_mix={tmix2})")
    print(f"{tag}  ratio       = {ratio:.2f}x")
    print(f"{tag}  pi2 uniform = {bool(np.max(np.abs(pi2 - 1.0/N)) < 1e-6)}")

    return {
        "n": n, "N": N,
        "gap_chain1": gap1,
        "gap_MH":     gap2,
        "ratio":      ratio,
        "t_mix_chain1": tmix1,
        "t_mix_MH":     tmix2,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    H_X = np.array([[1,1,1,1,0,0],[0,0,1,1,1,1]], dtype=np.uint8)
    res = run_chain2(H_X, label="[[6,2,2]]")
    assert res["N"] == 4
    print(f"Chain 2 self-test passed: gap_MH={res['gap_MH']:.4f}")
