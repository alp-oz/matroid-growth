"""
Metropolized chain gap scaling — Toric codes L=2..5

Two estimators compared side-by-side:
  (A) Original adjacent chain  (no MH correction)
  (B) Metropolized chain       (min-Metropolis rule targeting uniform)

For L=2,3: exact spectral gap via full N×N eigenvalue decomposition.
For L=4,5: empirical autocorrelation-time estimator (exact infeasible).

Empirical MH step: single-link acceptance ratio.
  From state C, propose C' via j*. Accept with probability
    min(1, [|elig(C)| · |fwd_parts(j,C)|] / [|elig(C')| · |bwd_parts(j,C')|])
  where bwd_parts(j,C') = decompose(C' △ fc[j]).
  C is rejected if it is not reachable from C' via j*.
  Stationary distribution is approximately uniform (not exactly, but much
  more uniform than original chain — and identical for L=2,3 when checked
  against exact min-Metropolis).

Goal: fit δ̂ ~ n^{-α} for Metropolized chain; compare α to original (0.36).
  α < 1 → poly(n) mixing (evidence preserved)
  α ≥ 1 → mixing time ≥ linear in n (poly evidence weakened/broken)
"""

import numpy as np
import random
import json
import sys
import os
OUT_JSON = os.path.join(os.path.dirname(__file__), "chain2_results.json")

from codes.toric_code import build_toric_hx, gf2_row_reduce
from core.circuits import all_circuits
from core.markov_chain import fundamental_circuits, decompose_into_circuits
from core.stationary import build_transition_matrix, stationary_distribution
from scipy.optimize import curve_fit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def metropolize(P):
    """Min-Metropolis rule: targets uniform exactly."""
    P_mh = np.minimum(P, P.T)
    np.fill_diagonal(P_mh, 0.0)
    np.fill_diagonal(P_mh, 1.0 - P_mh.sum(axis=1))
    return P_mh


def spectral_gap(P):
    eigs = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    return float(1.0 - eigs[1])


# ─────────────────────────────────────────────────────────────────────────────
# Chain steps
# ─────────────────────────────────────────────────────────────────────────────

def original_step(C, fc, M):
    """One step of original adjacent chain (no MH)."""
    eligible = [j for j, fj in fc.items() if fj & C]
    if not eligible:
        return C
    j = random.choice(eligible)
    sym = C.symmetric_difference(fc[j])
    if not sym:
        return C
    parts = decompose_into_circuits(M, sym)
    return random.choice(parts) if parts else C


def mh_step(C, fc, M):
    """
    One step of the Metropolized adjacent chain.

    Proposal: same as original (pick j uniformly from eligible(C),
              sym-diff, pick component uniformly).
    Acceptance: single-link ratio using j* only:
      alpha = min(1, [|elig(C)| · |fwd|] / [|elig(C')| · |bwd|])
    where fwd = decompose(C △ fc[j*]) and bwd = decompose(C' △ fc[j*]).
    Reject if C ∉ bwd_parts (C not reachable from C' via j*).
    """
    eligible_C = [j for j, fj in fc.items() if fj & C]
    if not eligible_C:
        return C
    j = random.choice(eligible_C)
    sym_fwd = C.symmetric_difference(fc[j])
    if not sym_fwd:
        return C
    fwd_parts = decompose_into_circuits(M, sym_fwd)
    if not fwd_parts:
        return C
    C_prop = random.choice(fwd_parts)

    # Compute backward
    eligible_prop = [j2 for j2, fj2 in fc.items() if fj2 & C_prop]
    if not eligible_prop:
        return C
    if not (fc[j] & C_prop):
        # j not eligible from C_prop → can't reverse via j → reject
        return C
    sym_rev = C_prop.symmetric_difference(fc[j])
    if not sym_rev:
        return C
    bwd_parts = decompose_into_circuits(M, sym_rev)
    if not bwd_parts or C not in bwd_parts:
        return C  # C not reachable from C' via j → reject

    ratio = (len(eligible_C) * len(fwd_parts)) / (len(eligible_prop) * len(bwd_parts))
    if random.random() < min(1.0, ratio):
        return C_prop
    return C


# ─────────────────────────────────────────────────────────────────────────────
# Autocorrelation time estimator (element-indicator)
# ─────────────────────────────────────────────────────────────────────────────

def autocorr_tau(step_fn, fc, M, T=60000, burn_in=6000, seed=42, label=""):
    """
    Estimate autocorrelation time τ and gap δ̂ = 1/τ via element indicators.
    Works for any step function without knowing circuit space.
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)

    all_elems = sorted(set().union(*fc.values()))
    k = min(16, len(all_elems))
    test_elems = rng.choice(all_elems, size=k, replace=False).tolist()

    C = random.choice(list(fc.values()))
    for _ in range(burn_in):
        C = step_fn(C, fc, M)

    traces = np.zeros((k, T), dtype=np.float64)
    for t in range(T):
        C = step_fn(C, fc, M)
        for i, e in enumerate(test_elems):
            traces[i, t] = float(e in C)

    taus = []
    max_lag = min(600, T // 10)
    rho_last = None
    for i in range(k):
        x = traces[i] - traces[i].mean()
        var = np.var(x)
        if var < 1e-10:
            continue
        rho = np.array([np.mean(x[:T - lag] * x[lag:]) / var
                        for lag in range(max_lag)])
        rho_last = rho
        pos_mask = rho > 0.05
        if pos_mask.sum() < 3:
            taus.append(1.0)
            continue
        lags_fit = np.where(pos_mask)[0]
        try:
            popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau),
                                lags_fit, rho[pos_mask],
                                p0=[5.0], bounds=(0.1, 4000))
            taus.append(float(popt[0]))
        except Exception:
            taus.append(float(0.5 + np.sum(rho[1:])))

    if not taus:
        return np.inf, 0.0
    tau = float(np.mean(taus))
    return tau, 1.0 / tau


# ─────────────────────────────────────────────────────────────────────────────
# Build Toric matroid
# ─────────────────────────────────────────────────────────────────────────────

def toric_matroid(L):
    """Returns M (RREF), r, fc (fundamental circuits), n_q."""
    H_X = build_toric_hx(L)
    n_q = 2 * L * L
    H_rref, pivot_cols, r = gf2_row_reduce(H_X)
    free_cols = [j for j in range(n_q) if j not in set(pivot_cols)]
    col_order = pivot_cols + free_cols
    M = H_rref[:, col_order].astype(np.float64)
    fc = fundamental_circuits(M, r)
    return M, r, fc, n_q


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

EXACT_MAX_N = 800   # build full matrix only if N ≤ this

# ─────────────────────────────────────────────────────────────────────────────
# Load cached results
# ─────────────────────────────────────────────────────────────────────────────

if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as f:
        results = json.load(f)
    done_Ls = {r["L"] for r in results}
    print(f"Loaded {len(results)} cached results from {OUT_JSON}")
else:
    results = []
    done_Ls = set()

print("Metropolized chain gap scaling — Toric codes")
print("=" * 65)
print(f"{'L':>3}  {'n':>4}  {'N':>6}  "
      f"{'gap_orig':>10}  {'gap_MH':>10}  {'ratio':>7}  "
      f"{'τ_orig':>7}  {'τ_MH':>7}  {'δ_orig':>8}  {'δ_MH':>8}  method")
print("-" * 95)

for L in [2, 3, 4, 5]:
    if L in done_Ls:
        r = next(x for x in results if x["L"] == L)
        gap_o = f"{r['gap_orig']:.4f}" if r["gap_orig"] else "—"
        gap_m = f"{r['gap_mh']:.4f}"   if r["gap_mh"]   else "—"
        ratio = (f"{r['gap_orig']/r['gap_mh']:.1f}x"
                 if r["gap_orig"] and r["gap_mh"] else "—")
        print(f"{L:>3}  {r['n']:>4}  {str(r['N']):>6}  "
              f"{gap_o:>10}  {gap_m:>10}  {ratio:>7}  "
              f"{r['tau_orig']:>7.1f}  {r['tau_mh']:>7.1f}  "
              f"{r['delta_orig']:>8.5f}  {r['delta_mh']:>8.5f}  [cached]")
        continue
    n_q = 2 * L * L
    M, r, fc, _ = toric_matroid(L)

    # Count circuits (BFS) only if we need exact gaps
    # For large L, skip enumeration and go straight to empirical
    gap_orig_exact = None
    gap_mh_exact   = None
    N = "?"

    if L <= 3:
        circuits_set, _, _ = all_circuits(M, r, mode="adjacent")
        circuits = list(circuits_set)
        N = len(circuits)
        if N <= EXACT_MAX_N:
            P_orig = build_transition_matrix(M, r, circuits, mode="adjacent")
            P_mh   = metropolize(P_orig)
            gap_orig_exact = spectral_gap(P_orig)
            gap_mh_exact   = spectral_gap(P_mh)

    # Empirical estimators (always run — gives τ and δ̂)
    tau_orig, delta_orig = autocorr_tau(original_step, fc, M,
                                        T=60000, burn_in=6000, seed=42)
    tau_mh,   delta_mh   = autocorr_tau(mh_step,       fc, M,
                                        T=60000, burn_in=6000, seed=42)

    # Report
    method = "exact+empirical" if gap_orig_exact is not None else "empirical"
    gap_o_str = f"{gap_orig_exact:.4f}" if gap_orig_exact is not None else "—"
    gap_m_str = f"{gap_mh_exact:.4f}"   if gap_mh_exact   is not None else "—"
    ratio_str = (f"{gap_orig_exact/gap_mh_exact:.1f}×"
                 if (gap_orig_exact and gap_mh_exact) else "—")

    print(f"{L:>3}  {n_q:>4}  {str(N):>6}  "
          f"{gap_o_str:>10}  {gap_m_str:>10}  {ratio_str:>7}  "
          f"{tau_orig:>7.1f}  {tau_mh:>7.1f}  "
          f"{delta_orig:>8.5f}  {delta_mh:>8.5f}  {method}")

    row = {
        "label":      f"Toric L={L}",
        "L":          L,
        "n":          n_q,
        "N":          N,
        "gap_chain1": gap_orig_exact,
        "gap_MH":     gap_mh_exact,
        "ratio":      round(gap_orig_exact / gap_mh_exact, 3)
                      if (gap_orig_exact and gap_mh_exact) else None,
        "gap_orig":   gap_orig_exact,
        "gap_mh":     gap_mh_exact,
        "tau_orig":   tau_orig,
        "tau_mh":     tau_mh,
        "delta_orig": delta_orig,
        "delta_mh":   delta_mh,
    }
    results.append(row)
    results_sorted = sorted(results, key=lambda x: x["L"])
    with open(OUT_JSON, "w") as f:
        json.dump(results_sorted, f, indent=2)

print("-" * 95)
print(f"\nResults saved to {OUT_JSON}")

# ─────────────────────────────────────────────────────────────────────────────
# Power-law fit   δ ~ C · n^{-α}
# ─────────────────────────────────────────────────────────────────────────────

ns = np.array([r["n"] for r in results], dtype=float)

for label, key in [("Original chain", "delta_orig"), ("MH chain", "delta_mh")]:
    deltas = np.array([r[key] for r in results], dtype=float)
    valid  = deltas > 1e-6
    if valid.sum() < 3:
        print(f"\n{label}: not enough data points for fit")
        continue
    log_n = np.log(ns[valid])
    log_d = np.log(deltas[valid])
    try:
        def plaw(ln, lc, a): return lc - a * ln
        popt, _ = curve_fit(plaw, log_n, log_d, p0=[0.0, 0.5])
        log_c, alpha = popt
        res = log_d - plaw(log_n, log_c, alpha)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((log_d - log_d.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        print(f"\n{label}:  α = {alpha:.3f}  (R² = {r2:.3f})")
        verdict = "POLY MIXING SUPPORTED (α<1)" if alpha < 1 else "α≥1 — POLY MIXING NOT SUPPORTED"
        print(f"  → {verdict}")
    except Exception as e:
        print(f"\n{label}: fit failed ({e})")

# Local slope L=2→3 for MH chain (most reliable, exact gaps)
d2 = next((r for r in results if r["L"] == 2), None)
d3 = next((r for r in results if r["L"] == 3), None)
if d2 and d3 and d2["gap_mh"] and d3["gap_mh"]:
    alpha_local = -np.log(d3["gap_mh"] / d2["gap_mh"]) / np.log(d3["n"] / d2["n"])
    print(f"\nLocal α (L=2→3, exact gaps):  α = {alpha_local:.3f}")
elif d2 and d3:
    alpha_local = -np.log(d3["delta_mh"] / d2["delta_mh"]) / np.log(d3["n"] / d2["n"])
    print(f"\nLocal α (L=2→3, empirical δ̂): α = {alpha_local:.3f}")
