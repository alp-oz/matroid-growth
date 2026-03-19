"""
Spectral gap scaling study — evidence for/against poly(n) mixing.

Two empirical estimators that work WITHOUT full circuit enumeration,
so they scale to large n where exact diagonalisation is impossible:

1. AUTOCORRELATION TIME (τ)
   Run one long chain, record circuit size at each step.
   Fit the autocorrelation ρ(t) ~ A·exp(-t/τ).
   Gap estimate: δ̂ = 1/τ.

2. COUPLING TIME (T_couple)
   Run two independent chains from different starting circuits.
   Record first time T they visit the same circuit.
   By the coupling lemma: t_mix ≤ E[T_couple].
   Gap lower bound: δ ≥ 1/(2·E[T_couple]).

Cross-validation: for small codes where exact gap is computable,
compare all three estimates to validate the empirical methods.

Scale-up: bicycle codes r = 6, 9, 12, 15, 18, 21, 24
          HGP(C523, rep(L)) for L = 3, 4, 5

Fit log(δ) vs log(n): slope α.
  α finite  → gap ~ 1/n^α → poly(n) mixing (supportive evidence)
  slope steepening → exponential decay → counter-evidence
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

from markov_chain import fundamental_circuits, decompose_into_circuits
from qecc_comparison import bicycle_code, h_to_matroid
from hgp_code import hgp, c523, rep
from toric_code import toric_to_matroid


# ─────────────────────────────────────────────────────────────────────────────
# Standalone chain step  (no full circuit list needed)
# ─────────────────────────────────────────────────────────────────────────────

def adjacent_step(C, fc, M):
    """
    One step of the adjacent chain.  Uses the correct GF(2) decomposition
    from markov_chain.py — no full circuit list required, only M and fc.
    """
    eligible = [j for j, fj in fc.items() if fj & C]
    if not eligible:
        return C
    j   = random.choice(eligible)
    sym = C.symmetric_difference(fc[j])
    if not sym:
        return C
    parts = decompose_into_circuits(M, sym)
    return random.choice(parts) if parts else C


# ─────────────────────────────────────────────────────────────────────────────
# Autocorrelation time estimator — element indicator version
# ─────────────────────────────────────────────────────────────────────────────

def autocorr_indicator(fc, M, n_elements=8, T=30000, burn_in=3000, seed=42):
    """
    Better autocorrelation estimate using element-presence indicators.

    For K random elements e, record f_e(C_t) = 1[e ∈ C_t] along one
    long trajectory.  Fit ρ_e(t) ~ exp(-t/τ_e) and return mean τ.

    This captures all eigenmodes of P (not just size-related ones), giving
    a better proxy for the true spectral gap δ ≈ 1/τ.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    all_elems = sorted(set().union(*fc.values()))
    k = min(n_elements, len(all_elems))
    test_elems = rng.choice(all_elems, size=k, replace=False).tolist()

    # Burn-in
    C = random.choice(list(fc.values()))
    for _ in range(burn_in):
        C = adjacent_step(C, fc, M)

    # Record indicators for all test elements in one pass
    traces = np.zeros((k, T), dtype=np.float64)
    for t in range(T):
        C = adjacent_step(C, fc, M)
        for i, e in enumerate(test_elems):
            traces[i, t] = float(e in C)

    # Fit autocorrelation for each element
    taus = []
    max_lag = min(300, T // 10)
    for i in range(k):
        x = traces[i] - traces[i].mean()
        var = np.var(x)
        if var < 1e-10:
            continue
        rho = np.array([np.mean(x[:T-lag] * x[lag:]) / var
                        for lag in range(max_lag)])
        pos = rho > 0.05
        if pos.sum() < 3:
            taus.append(1.0); continue
        lags_fit = np.where(pos)[0]
        try:
            popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau),
                                lags_fit, rho[pos], p0=[5.0],
                                bounds=(0.1, 2000))
            taus.append(float(popt[0]))
        except Exception:
            taus.append(float(0.5 + np.sum(rho[1:])))

    if not taus:
        return {"tau": np.inf, "delta": 0.0, "rho": np.zeros(50)}
    tau = float(np.mean(taus))
    return {"tau": tau, "delta": 1.0 / tau, "rho": rho}


def autocorrelation_time(fc, M, T=20000, burn_in=2000, seed=42):
    """
    Run one long chain, record |C| at each step.
    Fit ρ(t) ~ exp(-t/τ) to get τ ≈ 1/δ.

    Returns dict with τ_exp, δ_est, and the raw ρ array.
    """
    random.seed(seed)

    # Start from a random fundamental circuit
    starts = list(fc.values())
    C = random.choice(starts)

    # Burn-in
    for _ in range(burn_in):
        C = adjacent_step(C, fc, M)

    # Record sizes
    sizes = np.empty(T, dtype=np.float64)
    for t in range(T):
        C = adjacent_step(C, fc, M)
        sizes[t] = len(C)

    # Autocorrelation function
    sizes -= sizes.mean()
    var = np.var(sizes)
    if var < 1e-12:
        return {"tau": np.inf, "delta": 0.0, "rho": np.zeros(50)}

    max_lag = min(200, T // 10)
    rho = np.array([
        np.mean(sizes[:T-lag] * sizes[lag:]) / var
        for lag in range(max_lag)
    ])

    # Find where ρ first drops below 1/e — exponential fit
    rho_pos = rho[rho > 0.05]  # truncate at noise floor
    lags_pos = np.arange(len(rho_pos))

    if len(rho_pos) < 3:
        tau = 1.0
    else:
        try:
            def exp_decay(t, tau): return np.exp(-t / tau)
            popt, _ = curve_fit(exp_decay, lags_pos, rho_pos,
                                p0=[5.0], bounds=(0.1, 1000))
            tau = float(popt[0])
        except Exception:
            # Fallback: integrated autocorrelation time
            tau = 0.5 + float(np.sum(rho[1:]))

    return {"tau": tau, "delta": 1.0 / tau if tau > 0 else 0.0, "rho": rho}


# ─────────────────────────────────────────────────────────────────────────────
# Coupling time estimator
# ─────────────────────────────────────────────────────────────────────────────

def coupling_time_crn(fc, M, start1, start2, max_t=5000, seed=None):
    """
    Common-random-numbers (CRN) coupling — tighter than independent chains.

    At each step, both chains try to use the SAME non-basis index j.
    - If j is eligible for both C1 and C2: apply same step to both
      (same random decomposition choice → chains merge if they land together).
    - If j is eligible for only one: that chain moves, the other stays.

    This maximises shared transitions and reduces coupling time significantly
    vs fully independent chains, giving a tighter upper bound on t_mix.
    """
    if seed is not None:
        random.seed(seed)
    C1, C2 = start1, start2
    all_j = list(fc.keys())

    for t in range(1, max_t + 1):
        if C1 == C2:
            return t - 1
        # Pick one j for both chains
        j  = random.choice(all_j)
        fj = fc[j]
        e1 = bool(fj & C1)
        e2 = bool(fj & C2)

        if e1:
            sym1  = C1.symmetric_difference(fj)
            p1    = decompose_into_circuits(M, sym1) if sym1 else []
        if e2:
            sym2  = C2.symmetric_difference(fj)
            p2    = decompose_into_circuits(M, sym2) if sym2 else []

        if e1 and e2 and p1 and p2:
            # Same random index into whichever list is shorter
            idx = random.randrange(max(len(p1), len(p2)))
            C1  = p1[idx % len(p1)]
            C2  = p2[idx % len(p2)]
        else:
            if e1 and p1: C1 = random.choice(p1)
            if e2 and p2: C2 = random.choice(p2)

    return max_t


def coupling_time_single(fc, M, start1, start2, max_t=5000, seed=None):
    """
    Run two independent adjacent chains from start1, start2.
    Return first meeting time, or max_t if they don't meet.
    """
    if seed is not None:
        random.seed(seed)
    C1, C2 = start1, start2
    for t in range(1, max_t + 1):
        C1 = adjacent_step(C1, fc, M)
        C2 = adjacent_step(C2, fc, M)
        if C1 == C2:
            return t
    return max_t


def estimate_gap_coupling(fc, M, n_pairs=30, max_t=2000, seed=42):
    """
    Estimate t_mix (and hence gap) via coupling.
    Samples n_pairs of starting fundamental circuits, runs coupling,
    returns mean and max coupling time.
    """
    random.seed(seed)
    starts = list(fc.values())
    if len(starts) < 2:
        return {"t_couple_mean": np.inf, "t_couple_max": np.inf,
                "delta_lb": 0.0}

    times = []
    for _ in range(n_pairs):
        s1, s2 = random.sample(starts, 2)
        t = coupling_time_single(fc, M, s1, s2, max_t=max_t,
                                  seed=random.randint(0, 10**6))
        times.append(t)

    t_mean = float(np.mean(times))
    t_max  = float(np.max(times))
    # coupling lemma: t_mix ≤ t_couple; gap ≥ 1/(2·t_mix)
    delta_lb = 1.0 / (2.0 * t_max) if t_max > 0 else 0.0
    return {"t_couple_mean": t_mean, "t_couple_max": t_max,
            "delta_lb": delta_lb, "times": times}


# ─────────────────────────────────────────────────────────────────────────────
# Exact gap (small codes only)
# ─────────────────────────────────────────────────────────────────────────────

def exact_gap(M, r, max_circuits=4000):
    from analysis import all_circuits
    from stationary import build_transition_matrix
    from mixing import spectral_analysis

    all_c, trunc, _ = all_circuits(M, r, mode='global',
                                   max_circuits=max_circuits)
    if trunc or len(all_c) < 2:
        return None
    circuits = sorted(all_c, key=sorted)
    P  = build_transition_matrix(M, r, circuits, mode='adjacent')
    sp = spectral_analysis(P)
    return {"gap": sp["gap"], "N": len(circuits)}


# ─────────────────────────────────────────────────────────────────────────────
# Run one code instance: exact + empirical
# ─────────────────────────────────────────────────────────────────────────────

def analyse_instance(label, M, r, n,
                     T_auto=20000, n_pairs=40, max_t_couple=3000,
                     seed=42):
    fc = fundamental_circuits(M, r)

    # Exact gap (if feasible)
    eg = exact_gap(M, r)

    # Autocorrelation — element indicator (better than size)
    ac = autocorr_indicator(fc, M, n_elements=8, T=T_auto,
                            burn_in=2000, seed=seed)

    # CRN coupling (tighter than independent)
    starts = list(fc.values())
    crn_times = []
    rng_s = seed + 1
    for _ in range(n_pairs):
        s1, s2 = random.sample(starts, 2)
        t = coupling_time_crn(fc, M, s1, s2, max_t=max_t_couple,
                               seed=rng_s)
        crn_times.append(t)
        rng_s += 1
    t_crn_mean = float(np.mean(crn_times))
    t_crn_max  = float(np.max(crn_times))

    # Independent coupling (for comparison)
    cp = estimate_gap_coupling(fc, M, n_pairs=n_pairs,
                                max_t=max_t_couple, seed=seed + 100)

    return {
        "label": label, "n": n, "r": r,
        "N":          eg["N"]   if eg else None,
        "gap_exact":  eg["gap"] if eg else None,
        "tau":        ac["tau"],
        "delta_ind":  ac["delta"],
        "rho":        ac["rho"],
        "t_crn_mean": t_crn_mean,
        "t_crn_max":  t_crn_max,
        "delta_crn":  1.0 / (2.0 * t_crn_max) if t_crn_max > 0 else 0.0,
        "t_couple_mean": cp["t_couple_mean"],
        "t_couple_max":  cp["t_couple_max"],
        "delta_coup":    cp["delta_lb"],
        "couple_times":  cp["times"],
        "crn_times":     crn_times,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Code families to sweep
# ─────────────────────────────────────────────────────────────────────────────

def build_sweep():
    instances = []

    # Bicycle w3: r = 6, 9, 12, 15, 18, 21, 24
    for r_val in [6, 9, 12, 15, 18, 21, 24]:
        H = bicycle_code(r_val, [0, 1, 2], [0, 2, 3])
        M, r, n = h_to_matroid(H)
        instances.append((f"Bicycle-w3 r={r_val}", M, r, n, "Bicycle"))

    # Toric: L = 2, 3, 4, 5
    for L in [2, 3, 4, 5]:
        M, r, n, _, _ = toric_to_matroid(L)
        instances.append((f"Toric L={L}", M, r, n, "Toric"))

    # HGP(C523, rep(L)): L = 3, 4, 5
    for L in [3, 4, 5]:
        H_X, _ = hgp(c523(), rep(L))
        M, r, n = h_to_matroid(H_X)
        instances.append((f"HGP(C523,rep{L})", M, r, n, "HGP"))

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

FAM_COL = {"Bicycle": "#2ca02c", "Toric": "#1f77b4", "HGP": "#9467bd"}
FAM_MKR = {"Bicycle": "s", "Toric": "o", "HGP": "^"}

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)

    instances = build_sweep()

    print("Gap scaling study — exact + empirical estimators")
    print("=" * 90)
    print(f"{'label':<24}  {'n':>4}  {'N':>7}  {'gap_exact':>10}  "
          f"{'δ_auto':>8}  {'τ':>7}  {'δ_coup_lb':>10}  {'t_couple':>9}")
    print("=" * 90)

    results = []
    for label, M, r, n, family in instances:
        print(f"  {label:<24}  n={n}...", end=" ", flush=True)
        res = analyse_instance(label, M, r, n)
        res["family"] = family
        results.append(res)

        N_s      = str(res["N"]) if res["N"] else "—"
        gap_s    = f"{res['gap_exact']:.4f}" if res["gap_exact"] is not None else "—"
        tau_s    = f"{res['tau']:.1f}"
        dauto_s  = f"{res['delta_ind']:.4f}"
        dcoup_s  = f"{res['delta_coup']:.4f}"
        tcouple_s = f"{res['t_couple_mean']:.1f}"
        print(f"\r  {label:<24}  {n:>4}  {N_s:>7}  {gap_s:>10}  "
              f"{dauto_s:>8}  {tau_s:>7}  {dcoup_s:>10}  {tcouple_s:>9}")

    # ── Fit gap vs n ──────────────────────────────────────────────────────────
    print("\nPower-law fits  δ ~ n^(-α)  per family:")
    print(f"  {'family':<10}  {'estimator':<12}  α (slope)   R²")
    print("  " + "-" * 45)

    def powerlaw(log_n, log_a, alpha): return log_a - alpha * log_n

    for family in ["Bicycle", "Toric", "HGP"]:
        fam_res = [r for r in results if r["family"] == family]
        if len(fam_res) < 3: continue
        ns = np.array([r["n"] for r in fam_res], dtype=float)

        for key, label in [("gap_exact",  "exact    "),
                            ("delta_ind",  "autocorr "),
                            ("delta_coup", "coupling ")]:
            vals = np.array([r[key] for r in fam_res], dtype=float)
            mask = (vals > 1e-6) & np.isfinite(vals)
            if mask.sum() < 3: continue
            log_n = np.log(ns[mask])
            log_v = np.log(vals[mask])
            try:
                popt, _ = curve_fit(powerlaw, log_n, log_v, p0=[0.0, 1.0])
                residuals = log_v - powerlaw(log_n, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((log_v - log_v.mean())**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                print(f"  {family:<10}  {label}  α={popt[1]:>6.3f}    R²={r2:.3f}")
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════════════
    # Figure
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.38)

    # ── Panel 1: Gap (exact) vs n ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    seen = set()
    for res in results:
        if res["gap_exact"] is None: continue
        fam = res["family"]
        lbl = fam if fam not in seen else None
        ax1.scatter(res["n"], res["gap_exact"],
                    color=FAM_COL[fam], marker=FAM_MKR[fam], s=80, zorder=3,
                    label=lbl)
        seen.add(fam)
    ax1.set_xlabel("n", fontsize=11); ax1.set_ylabel("Exact spectral gap δ", fontsize=11)
    ax1.set_title("Exact gap vs n", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # ── Panel 2: log-log exact gap ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for family in ["Bicycle", "Toric", "HGP"]:
        fam_res = [r for r in results if r["family"] == family
                   and r["gap_exact"] is not None and r["gap_exact"] > 1e-6]
        if not fam_res: continue
        ns  = np.array([r["n"] for r in fam_res])
        gs_ = np.array([r["gap_exact"] for r in fam_res])
        order = np.argsort(ns)
        ax2.loglog(ns[order], gs_[order], FAM_MKR[family] + "-",
                   color=FAM_COL[family], lw=2, ms=7, label=family)
    # Reference lines
    n_ref = np.array([8, 60], dtype=float)
    ax2.loglog(n_ref, 0.5 / n_ref,    "k:", lw=1, alpha=0.6, label="~1/n")
    ax2.loglog(n_ref, 2.0 / n_ref**2, "k--", lw=1, alpha=0.4, label="~1/n²")
    ax2.set_xlabel("n  (log scale)", fontsize=11)
    ax2.set_ylabel("Exact gap δ  (log scale)", fontsize=11)
    ax2.set_title("Log-log: gap vs n\n(slope = −α)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")

    # ── Panel 3: δ_auto vs n (log-log) ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for family in ["Bicycle", "Toric", "HGP"]:
        fam_res = [r for r in results if r["family"] == family
                   and r["delta_ind"] > 1e-6]
        if not fam_res: continue
        ns  = np.array([r["n"] for r in fam_res])
        ds  = np.array([r["delta_ind"] for r in fam_res])
        order = np.argsort(ns)
        ax3.loglog(ns[order], ds[order], FAM_MKR[family] + "-",
                   color=FAM_COL[family], lw=2, ms=7, label=family)
    ax3.loglog(n_ref, 0.5 / n_ref,    "k:", lw=1, alpha=0.6, label="~1/n")
    ax3.loglog(n_ref, 2.0 / n_ref**2, "k--", lw=1, alpha=0.4, label="~1/n²")
    ax3.set_xlabel("n  (log scale)", fontsize=11)
    ax3.set_ylabel("δ̂ = 1/τ  (log scale)", fontsize=11)
    ax3.set_title("Autocorrelation gap estimate\n(log-log)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3, which="both")

    # ── Panel 4: Coupling time vs n ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    seen = set()
    for res in results:
        fam = res["family"]
        lbl = fam if fam not in seen else None
        ax4.scatter(res["n"], res["t_couple_mean"],
                    color=FAM_COL[fam], marker=FAM_MKR[fam], s=60, zorder=3,
                    label=lbl, alpha=0.8)
        seen.add(fam)
    ax4.plot(n_ref, n_ref, "k:", lw=1.2, alpha=0.6, label="t = n")
    ax4.set_xlabel("n", fontsize=11)
    ax4.set_ylabel("Mean coupling time", fontsize=11)
    ax4.set_title("Coupling time vs n\n(below t=n line → O(n) mixing)",
                  fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    # ── Panel 5: Autocorrelation curves ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    # One per family: pick largest available instance
    plotted = set()
    for res in sorted(results, key=lambda x: -x["n"]):
        fam = res["family"]
        if fam in plotted: continue
        rho = res["rho"]
        lags = np.arange(len(rho))
        tau_s = f"{res['tau']:.1f}"
        ax5.plot(lags, rho, color=FAM_COL[fam], lw=2,
                 label=f"{fam} n={res['n']}  τ={tau_s}")
        plotted.add(fam)
    ax5.axhline(0, color="gray", lw=0.8, ls=":")
    ax5.axhline(1/np.e, color="gray", lw=1, ls="--", alpha=0.5, label="1/e")
    ax5.set_xlim(0, 60); ax5.set_ylim(-0.1, 1.05)
    ax5.set_xlabel("Lag t", fontsize=11)
    ax5.set_ylabel("ρ(t)  (circuit size autocorrelation)", fontsize=11)
    ax5.set_title("Autocorrelation functions", fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

    # ── Panel 6: Exact vs auto gap comparison (cross-validation) ─────────────
    ax6 = fig.add_subplot(gs[1, 2])
    exact_vals, auto_vals = [], []
    for res in results:
        if res["gap_exact"] is not None and res["delta_ind"] > 1e-6:
            exact_vals.append(res["gap_exact"])
            auto_vals.append(res["delta_ind"])
    if exact_vals:
        ax6.scatter(exact_vals, auto_vals, s=80, color="#333333", zorder=3)
        lim = max(max(exact_vals), max(auto_vals)) * 1.1
        ax6.plot([0, lim], [0, lim], "r--", lw=1.5, alpha=0.7, label="y=x (perfect)")
        ax6.set_xlabel("Exact spectral gap δ", fontsize=11)
        ax6.set_ylabel("Autocorrelation estimate 1/τ", fontsize=11)
        ax6.set_title("Cross-validation:\nexact gap vs autocorrelation estimate",
                      fontsize=11, fontweight="bold")
        ax6.legend(fontsize=9); ax6.grid(alpha=0.3)

    fig.suptitle("Spectral gap scaling — evidence for/against poly(n) mixing",
                 fontsize=14, fontweight="bold")
    out = "markov-circuits/gap_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()
