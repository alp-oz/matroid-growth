"""
Chain 3-large gap scaling experiment.

Compares spectral gap of Chain 3 vs Chain 3-large (s=2 and s=3 steps)
on Toric codes L=2 and L=3.

Chain 3-large (s steps): pick j_1,...,j_s WITHOUT replacement from non-basis;
  c_new = (c + fc[j_1] + ... + fc[j_s]) % 2
  If c_new = 0: stay. Else: move.

Stationary: still exactly uniform over ker(H_X) \\ {0}.
Proof: P[c -> c'] = #{size-s subsets S : XOR_{j in S} fc[j] = c XOR c'} / C(k,s)
       which is symmetric in c and c', so detailed balance holds.

Key question: does gap(Chain 3-large) > gap(Chain 3)?
  YES -> larger steps help, step size bottleneck
  NO  -> geometry bottleneck, step size irrelevant
"""

import sys
import os
import numpy as np
from chains.codeword_walk import (
    preprocess, enumerate_all_codewords, spectral_gap, stationary_dist,
)
from codes.toric_code import build_toric_hx


# ─────────────────────────────────────────────────────────────────────────────
# Chain 3-large transition matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_chain3large_matrix(codewords, fc, non_basis, s):
    """
    Exact N×N transition matrix of Chain 3-large with step size s.

    From state c: pick s distinct elements j_1,...,j_s uniformly at random
    (without replacement) from non_basis; move to c XOR fc[j_1] XOR ... XOR fc[j_s].
    If result is zero: stay (lazy).

    P[c -> c'] = #{size-s subsets S of non_basis : XOR_{j in S} fc[j] = c XOR c'} / C(k,s)
    """
    from itertools import combinations

    N = len(codewords)
    k = len(non_basis)
    idx = {tuple(c): i for i, c in enumerate(codewords)}

    # Precompute all size-s subsets and their XOR
    subsets = list(combinations(range(k), s))
    n_subsets = len(subsets)  # = C(k, s)

    P = np.zeros((N, N), dtype=np.float64)
    weight = 1.0 / n_subsets

    fcs = [fc[non_basis[i]] for i in range(k)]

    for i, c in enumerate(codewords):
        for sub in subsets:
            # XOR of fc[j] for j in this subset
            delta = np.zeros(len(c), dtype=np.int32)
            for idx_nb in sub:
                delta += fcs[idx_nb].astype(np.int32)
            c_new = (c.astype(np.int32) + delta) % 2
            c_new = c_new.astype(np.uint8)
            if not c_new.any():
                P[i, i] += weight
            else:
                j_idx = idx.get(tuple(c_new))
                if j_idx is not None:
                    P[i, j_idx] += weight

    row_sums = P.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, P / row_sums, P)


# ─────────────────────────────────────────────────────────────────────────────
# Verify uniform stationary
# ─────────────────────────────────────────────────────────────────────────────

def check_uniform(P, N, label=""):
    pi = stationary_dist(P)
    max_dev = float(np.max(np.abs(pi - 1.0 / N)))
    ok = max_dev < 1e-6
    tag = f"[{label}] " if label else ""
    print(f"{tag}  π uniform? {'YES' if ok else 'NO'}  (max dev = {max_dev:.2e})")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(L, s_values=(1, 2, 3)):
    H_X = build_toric_hx(L)
    basis, non_basis, r, fc = preprocess(H_X)
    n = H_X.shape[1]
    k = len(non_basis)
    N = 2 ** k - 1

    print(f"\nToric L={L}  n={n}  k={k}  N_codewords={N}")

    codewords = enumerate_all_codewords(fc, non_basis, n)
    print(f"  Enumerated {len(codewords)} codewords.")

    results = {}
    for s in s_values:
        if s > k:
            print(f"  s={s}: skipped (k={k} < s)")
            continue
        from math import comb
        n_subsets = comb(k, s)
        print(f"  Building matrix for s={s} ({n_subsets} subsets × {N} states)...",
              end=" ", flush=True)
        P = build_chain3large_matrix(codewords, fc, non_basis, s)
        check_uniform(P, N)
        gap = spectral_gap(P)
        results[s] = gap
        print(f"  s={s}  gap = {gap:.6f}")

    return results


if __name__ == "__main__":
    print("Chain 3-large gap scaling")
    print("=" * 60)
    print("Chain 3-large: pick s non-basis elements WITHOUT replacement,")
    print("XOR all their fundamental circuits.  Stationary = uniform.")
    print()

    all_results = {}
    for L in [2, 3]:
        res = run_experiment(L, s_values=[1, 2, 3])
        all_results[L] = res

    # Summary table
    print()
    print("=" * 60)
    print("Summary — spectral gap by L and step size s")
    print(f"{'':>10}  {'s=1 (Chain 3)':>15}  {'s=2':>10}  {'s=3':>10}  "
          f"{'ratio s=2/s=1':>15}  {'ratio s=3/s=1':>15}")
    print("-" * 80)
    for L, res in sorted(all_results.items()):
        g1 = res.get(1)
        g2 = res.get(2)
        g3 = res.get(3)
        r2 = f"{g2/g1:.2f}x" if (g1 and g2) else "—"
        r3 = f"{g3/g1:.2f}x" if (g1 and g3) else "—"
        g1s = f"{g1:.6f}" if g1 else "—"
        g2s = f"{g2:.6f}" if g2 else "—"
        g3s = f"{g3:.6f}" if g3 else "—"
        print(f"Toric L={L}  {g1s:>15}  {g2s:>10}  {g3s:>10}  {r2:>15}  {r3:>15}")

    print()
    print("Interpretation:")
    print("  ratio >> 1 -> larger steps significantly improve mixing")
    print("  ratio ≈ 1  -> bottleneck is geometric, not step size")
