"""
Chain 3 gap scaling — does the codeword walk mix in poly(n) time?

For each code family and size, runs Chain 3 (Codeword Walk) and estimates
the spectral gap δ via autocorrelation time.  Fits δ ~ n^{-α} per family.

Families
--------
  Toric      L = 2, 3, 4, 5        n = 8, 18, 32, 50
  Bicycle    r = 6, 9, 12, 15, 18  n = 12, 18, 24, 30, 36
  BB         (3,3),(4,3),(6,3)      n = 18, 24, 36

Small codes (2^k ≤ threshold): also compute exact gap from N×N matrix.
All codes: empirical gap from autocorrelation time of element indicators.

Cross-reference: Chain 1 α values from gap_scaling.py results
  Toric α₁ = 0.36,  Bicycle α₁ = 0.46,  BB α₁ = 0.50

Key question: is α₃ < 1 for Chain 3?
"""

import sys
import os
import json
import numpy as np
import random
from chains.codeword_walk import (
    preprocess, chain3_step, autocorr_gap,
    enumerate_all_codewords, build_chain3_matrix,
    spectral_gap, stationary_dist, tv_distance_curve,
)
from codes.toric_code import build_toric_hx
from codes.qecc_comparison import bicycle_code, h_to_matroid
from codes.bb_code import bb_code
from scipy.optimize import curve_fit


# ──────────────────────────────────────────────────────────────────────────────
# Code instances
# ──────────────────────────────────────────────────────────────────────────────

def get_instances():
    """
    Returns list of dicts:
      label, family, n, H_X
    """
    instances = []

    # ── Toric ─────────────────────────────────────────────────────────────────
    for L in [2, 3, 4, 5]:
        H_X = build_toric_hx(L)
        instances.append({"label": f"Toric L={L}", "family": "Toric",
                          "n": 2 * L * L, "H_X": H_X})

    # ── Bicycle ───────────────────────────────────────────────────────────────
    for r in [6, 9, 12, 15, 18]:
        try:
            H_X = bicycle_code(r, [0, 1, 2], [0, 2, 3])
            n = H_X.shape[1]
            instances.append({"label": f"Bicycle r={r}", "family": "Bicycle",
                               "n": n, "H_X": H_X.astype(np.uint8)})
        except Exception as e:
            print(f"  bicycle_code r={r} failed: {e}")

    # ── BB (bivariate bicycle) ─────────────────────────────────────────────────
    bb_params = [
        (3, 3, [(0,0),(1,0),(0,1)], [(0,0),(2,0),(0,2)]),   # n=18
        (6, 3, [(0,0),(2,0),(0,1)], [(0,0),(4,0),(0,2)]),   # n=36
        (6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)]),   # n=144 (IBM)
    ]
    for l, m, a_sh, b_sh in bb_params:
        try:
            H_X, _ = bb_code(l, m, a_sh, b_sh)
            n = H_X.shape[1]
            instances.append({"label": f"BB({l},{m})", "family": "BB",
                               "n": n, "H_X": H_X.astype(np.uint8)})
        except Exception as e:
            print(f"  bb_code({l},{m}) failed: {e}")

    return instances


# ──────────────────────────────────────────────────────────────────────────────
# Per-instance analysis
# ──────────────────────────────────────────────────────────────────────────────

EXACT_CW_THRESHOLD = 8000   # enumerate codewords exactly if 2^k ≤ this

def analyse_instance(inst, T=60000, burn_in=6000, seed=42):
    """
    Run Chain 3 on one instance. Returns result dict.
    """
    label   = inst["label"]
    family  = inst["family"]
    n       = inst["n"]
    H_X     = inst["H_X"]

    basis, non_basis, r, fc = preprocess(H_X)
    k_dim  = len(non_basis)     # dim ker(H_X) = n - rank
    N_cw   = 2 ** k_dim - 1    # number of nonzero codewords

    res = {"label": label, "family": family, "n": n,
           "r": r, "k_dim": k_dim, "N_cw": N_cw}

    # Exact gap if state space is small enough
    if N_cw <= EXACT_CW_THRESHOLD:
        try:
            codewords = enumerate_all_codewords(fc, non_basis, n)
            P3 = build_chain3_matrix(codewords, fc, non_basis)
            pi3 = stationary_dist(P3)
            res["gap_exact"]   = spectral_gap(P3)
            res["pi_uniform"]  = bool(np.max(np.abs(pi3 - 1.0 / N_cw)) < 1e-6)
            tv_max, t_mix = tv_distance_curve(P3, pi3, max_t=200)
            res["t_mix"]  = t_mix
        except Exception as e:
            res["gap_exact"] = None
            res["exact_error"] = str(e)
    else:
        res["gap_exact"] = None

    # Empirical gap (always)
    try:
        tau, delta = autocorr_gap(chain3_step, fc, non_basis,
                                  T=T, burn_in=burn_in, seed=seed)
        res["tau"]         = round(tau, 2)
        res["gap_empirical"] = round(delta, 6)
    except Exception as e:
        res["tau"]         = None
        res["gap_empirical"] = None
        res["empirical_error"] = str(e)

    # Use exact gap for power-law fit when available, else empirical
    res["gap_fit"] = res["gap_exact"] if res["gap_exact"] else res["gap_empirical"]

    return res


# ──────────────────────────────────────────────────────────────────────────────
# Power-law fit  δ ~ C · n^{-α}
# ──────────────────────────────────────────────────────────────────────────────

def fit_powerlaw(results, family):
    pts = [(r["n"], r["gap_fit"])
           for r in results
           if r["family"] == family and r.get("gap_fit") and r["gap_fit"] > 1e-6]
    if len(pts) < 3:
        return None, None, None
    ns    = np.array([p[0] for p in pts], dtype=float)
    deltas = np.array([p[1] for p in pts], dtype=float)
    try:
        popt, _ = curve_fit(lambda ln, lc, a: lc - a * ln,
                            np.log(ns), np.log(deltas), p0=[0.0, 0.5])
        log_c, alpha = popt
        pred  = log_c - alpha * np.log(ns)
        ss_res = np.sum((np.log(deltas) - pred) ** 2)
        ss_tot = np.sum((np.log(deltas) - np.log(deltas).mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        return float(alpha), float(np.exp(log_c)), r2
    except Exception:
        return None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Chain 1 α values for comparison (from gap_scaling_results.json)
CHAIN1_ALPHA = {"Toric": 0.36, "Bicycle": 0.46, "BB": 0.50}

if __name__ == "__main__":

    print("Chain 3 gap scaling")
    print("=" * 70)

    instances = get_instances()
    print(f"Running {len(instances)} instances...\n")

    results = []
    for inst in instances:
        label = inst["label"]
        sys.stdout.write(f"  {label:<20} n={inst['n']:<4} ... ")
        sys.stdout.flush()
        r = analyse_instance(inst)
        results.append(r)

        gap_str = (f"gap_exact={r['gap_exact']:.5f}" if r["gap_exact"]
                   else f"gap_emp={r['gap_empirical']}")
        print(f"k={r['k_dim']}  N_cw={r['N_cw']:<8}  τ={r['tau']}  {gap_str}")

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'label':<22} {'n':>4} {'k':>3} {'τ':>7} {'δ_exact':>10} "
          f"{'δ_emp':>10} {'method'}")
    print("-" * 70)
    for r in results:
        method = "exact" if r.get("gap_exact") else "empirical"
        g_ex = f"{r['gap_exact']:.5f}" if r.get("gap_exact") else "—"
        g_em = f"{r['gap_empirical']:.5f}" if r.get("gap_empirical") else "—"
        print(f"{r['label']:<22} {r['n']:>4} {r['k_dim']:>3} "
              f"{str(r['tau']):>7} {g_ex:>10} {g_em:>10}  {method}")

    # ── Power-law fits ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Power-law fits  δ ~ n^{-α}")
    print(f"{'Family':<12} {'α_chain3':>10} {'R²':>6}  vs  {'α_chain1':>10}  verdict")
    print("-" * 65)

    for family in ["Toric", "Bicycle", "BB"]:
        alpha, _, r2 = fit_powerlaw(results, family)
        alpha1 = CHAIN1_ALPHA.get(family, "?")
        if alpha is not None:
            verdict = ("POLY MIXING (α<1)" if alpha < 1.0
                       else "α≥1 — not sub-linear")
            print(f"{family:<12} {alpha:>10.3f} {r2:>6.3f}     {alpha1:>10}  {verdict}")
        else:
            print(f"{family:<12} {'—':>10} {'—':>6}     {alpha1:>10}  (not enough data)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "chain3_gap_scaling.json")
    compact = []
    for r in results:
        compact.append({
            "label":        r["label"],
            "family":       r["family"],
            "n":            r["n"],
            "k_dim":        r["k_dim"],
            "N_cw":         r["N_cw"],
            "tau":          r["tau"],
            "gap_exact":    round(r["gap_exact"], 6) if r.get("gap_exact") else None,
            "gap_empirical":round(r["gap_empirical"], 6) if r.get("gap_empirical") else None,
        })
    with open(out_path, "w") as f:
        json.dump(compact, f, indent=2)
    print(f"\nSaved → {out_path}")
