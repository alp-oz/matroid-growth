"""
Chain 4 q sweep — biased codeword walk for A_logical(z) and d_Z estimation.

Chain 4: biased codeword walk on ker(H_X) \\ {0}, targeting π(c) ∝ q^{wt(c)}.
Records minimum-weight representative per logical coset → A_logical(z), d_Z.

Background
----------
A CSS code has H_X (m_X × n) and H_Z (m_Z × n) over GF(2) with H_X H_Z^T = 0.
  V = ker(H_X)        — Z-type undetectable errors
  W = rowspan(H_Z)    — Z-stabilizers (trivial errors)
  W ⊆ V always.

The nonzero cosets of V/W are logical error classes.
  k_logical = n − rank(H_X) − rank(H_Z)
  d_Z = min weight over all nonzero cosets
  A_logical(z) = Σ_{nonzero cosets L} z^{min_wt(L)}

Chain 4 step from state c:
  1. Pick j uniformly from ALL non-basis elements.
  2. c_new = (c + fc[j]) % 2.
  3. If c_new = 0: stay.
  4. Else: delta_w = wt(c_new) − wt(c).
           Accept if delta_w ≤ 0; accept with prob q^{delta_w} if delta_w > 0.
  Recording (EVERY step, accepted or not):
     label = tuple(dot(c, Z_i) % 2 for Z_i in logical_basis)
     min_wt[label] = min(min_wt.get(label, inf), wt(c))

Stationary: π(c) ∝ q^{wt(c)}.  Smaller q → stronger bias toward low weight.

Experiment
----------
Test on [[6,2,2]] code (verified exact answers).
Then sweep q ∈ {0.01, 0.1, 0.3, 0.5, 0.7, 1.0} on Toric L=2 and L=3.

Metrics per q:
  T_cover   : steps until all nonzero cosets first visited
  T_converge: steps until min_wt last decreased (+ stability window)
  d_Z       : final estimated minimum code distance
  A_logical : final estimated logical weight enumerator
"""

import sys
import os
import random
import math
import numpy as np
from chains.codeword_walk import preprocess
from chains.coset_descent_chain import compute_logical_basis, coset_label
from codes.toric_code import build_toric_hx


# ─────────────────────────────────────────────────────────────────────────────
# Chain 4 runner with convergence tracking
# ─────────────────────────────────────────────────────────────────────────────

def run_chain4(H_X, H_Z, q, T_max=200000, stability_window=5000, seed=42):
    """
    Run Chain 4 and track convergence of A_logical(z).

    Returns dict with:
      T_cover     : step when last new coset was first visited (-1 if incomplete)
      T_converge  : step when min_wt last decreased across any coset (-1 if incomplete)
      d_Z         : minimum distance estimate
      A_logical   : dict {weight: count} — logical weight enumerator
      min_wt      : dict {label: min weight seen}
      coverage    : fraction of nonzero cosets visited
    """
    random.seed(seed)

    basis, non_basis, r, fc = preprocess(H_X)
    n = H_X.shape[1]
    # X-type logicals (ker(H_Z) / rowspan(H_X)) are needed for coset labeling:
    # dot(c, X_i) % 2 vanishes on Z-stabilizers and distinguishes Z-cosets.
    # compute_logical_basis(A, B) returns ker(A) \ rowspan(B), so swap args.
    logicals = compute_logical_basis(H_Z, H_X)
    k_logical = len(logicals)
    n_nonzero_cosets = 2 ** k_logical - 1
    zero_label = tuple(0 for _ in range(k_logical))

    # Initialise state at a random fundamental circuit
    c = fc[random.choice(non_basis)].copy()

    min_wt = {}          # label -> min weight seen
    first_visit = {}     # label -> step of first visit
    T_cover = -1
    T_converge = -1
    last_decrease_step = 0

    for t in range(1, T_max + 1):
        # ── Proposal ──────────────────────────────────────────────────────────
        j = random.choice(non_basis)
        c_new = (c.astype(np.int32) + fc[j].astype(np.int32)) % 2
        c_new = c_new.astype(np.uint8)

        if not c_new.any():
            pass  # stay
        else:
            delta_w = int(c_new.sum()) - int(c.sum())
            if delta_w <= 0 or random.random() < q ** delta_w:
                c = c_new

        # ── Recording (every step) ────────────────────────────────────────────
        lab = coset_label(c, logicals)
        wt = int(c.sum())

        if lab != zero_label:
            if lab not in first_visit:
                first_visit[lab] = t

            if lab not in min_wt or wt < min_wt[lab]:
                min_wt[lab] = wt
                last_decrease_step = t

        # ── Convergence checks ────────────────────────────────────────────────
        if T_cover == -1 and len(first_visit) == n_nonzero_cosets:
            T_cover = t

        if (T_cover != -1 and T_converge == -1
                and t - last_decrease_step >= stability_window):
            T_converge = t

        if T_converge != -1:
            break  # done

    # Build A_logical(z)
    A_logical = {}
    for wt in min_wt.values():
        A_logical[wt] = A_logical.get(wt, 0) + 1

    d_Z = min(min_wt.values()) if min_wt else None
    coverage = len(first_visit) / n_nonzero_cosets if n_nonzero_cosets > 0 else 0.0

    return {
        "T_cover":    T_cover,
        "T_converge": T_converge,
        "d_Z":        d_Z,
        "A_logical":  A_logical,
        "min_wt":     min_wt,
        "coverage":   coverage,
        "k_logical":  k_logical,
        "n_cosets":   n_nonzero_cosets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Verification — [[6,2,2]] test case
# ─────────────────────────────────────────────────────────────────────────────

def verify_test_case():
    """
    [[6,2,2]] code: k_logical=2, d_Z=2, A_logical(z) = 3z^2.
    3 nonzero cosets, each with minimum weight 2.
    """
    H_X = np.array([[1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1]], dtype=np.uint8)
    H_Z = np.array([[1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1]], dtype=np.uint8)

    print("=" * 60)
    print("VERIFICATION — [[6,2,2]] code")
    print("Expected: k_logical=2, d_Z=2, A_logical = 3z^2")
    print("=" * 60)

    # Check preprocessing
    basis, non_basis, r, fc = preprocess(H_X)
    print(f"  rank(H_X) = {r}  (expected 2)")
    print(f"  non_basis = {non_basis}  (expected [1,3,4,5])")
    for j in non_basis:
        print(f"  fc[{j}] = {fc[j].tolist()}  wt={int(fc[j].sum())}")

    logicals = compute_logical_basis(H_Z, H_X)   # X-logicals for coset labeling
    k_log = len(logicals)
    print(f"  k_logical = {k_log}  (expected 2)")
    n_cosets = 2 ** k_log - 1
    print(f"  nonzero cosets = {n_cosets}  (expected 3)")

    # Run with two q values
    passed = True
    for q in [0.1, 1.0]:
        res = run_chain4(H_X, H_Z, q=q, T_max=50000,
                         stability_window=2000, seed=42)
        d_Z = res["d_Z"]
        A = res["A_logical"]
        cov = res["coverage"]
        ok_d = (d_Z == 2)
        ok_A = (A == {2: 3})
        ok_c = (cov == 1.0)
        status = "PASS" if (ok_d and ok_A and ok_c) else "FAIL"
        print(f"\n  q={q}: T_cover={res['T_cover']}  "
              f"T_converge={res['T_converge']}  "
              f"d_Z={d_Z}  A_logical={A}  coverage={cov:.0%}  [{status}]")
        if status == "FAIL":
            passed = False
            print(f"    EXPECTED: d_Z=2, A_logical={{2:3}}, coverage=100%")

    print()
    if passed:
        print("Verification PASSED ✓")
    else:
        print("Verification FAILED ✗ — check implementation before continuing")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Beta sweep on one code instance
# ─────────────────────────────────────────────────────────────────────────────

BETAS = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]

def q_sweep(H_X, H_Z, label, T_max=300000, stability_window=10000, seed=42):
    print(f"\n{'─'*65}")
    print(f"Beta sweep — {label}")
    print(f"{'─'*65}")
    print(f"{'q':>6}  {'T_cover':>10}  {'T_converge':>12}  "
          f"{'d_Z':>5}  {'A_logical':>25}  {'coverage':>9}")
    print("-" * 75)

    results = []
    for q in BETAS:
        res = run_chain4(H_X, H_Z, q=q,
                         T_max=T_max, stability_window=stability_window,
                         seed=seed)
        tc  = str(res["T_cover"])    if res["T_cover"]    != -1 else f">T_max"
        tcv = str(res["T_converge"]) if res["T_converge"] != -1 else f">T_max"
        dz  = str(res["d_Z"]) if res["d_Z"] is not None else "?"
        A_str = "  ".join(f"z^{w}×{c}" for w, c in sorted(res["A_logical"].items()))
        cov = f"{res['coverage']:.0%}"
        print(f"{q:>6.2f}  {tc:>10}  {tcv:>12}  {dz:>5}  {A_str:>25}  {cov:>9}")
        results.append((q, res))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: verify on [[6,2,2]] ───────────────────────────────────────────
    ok = verify_test_case()
    if not ok:
        sys.exit(1)

    # ── Step 2: Toric L=2 ────────────────────────────────────────────────────
    from toric_code import build_toric_hx, build_toric_hz

    L = 2
    H_X2 = build_toric_hx(L)
    H_Z2 = build_toric_hz(L)
    q_sweep(H_X2, H_Z2, label=f"Toric L={L}  n={2*L*L}",
               T_max=500000, stability_window=20000)

    # ── Step 3: Toric L=3 ────────────────────────────────────────────────────
    L = 3
    H_X3 = build_toric_hx(L)
    H_Z3 = build_toric_hz(L)
    q_sweep(H_X3, H_Z3, label=f"Toric L={L}  n={2*L*L}",
               T_max=2000000, stability_window=50000)
