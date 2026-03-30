"""
Chain 4 spectral gap scaling — δ vs n (log-log), six CSS families.

Chain 4: biased codeword walk on ker(H_X)\\{0} with MH acceptance rule.
  Step:  pick j uniformly from non_basis; propose c' = (c + fc[j]) mod 2.
         If c' = 0: stay. Else accept with prob min(1, q^{wt(c')−wt(c)}).
  Stationary: π(c) ∝ q^{wt(c)}.  Fixed q = 0.3.

For each instance:
  Exact gap (2^k_dim ≤ EXACT_THRESHOLD): build full N×N matrix, compute eigvals.
  Empirical gap (larger codes): autocorrelation time of wt(c_t).

Families and sizes
------------------
  Toric:   L ∈ {2,3,4,5}                     n ∈ {8,18,32,50}
  Bicycle: r ∈ {6,9,12,15,18}                n ∈ {12,18,24,30,36}
  BB:      (3,3), (6,3)                       n ∈ {18,36}
  HGP:     (C523,rep3), (rep4), (rep5)        n ∈ {21,29,37}
  FB:      uniform(3,3), alt(4,4), alt(6,6)   n ∈ {18,32,72}
  QT:      PSL(2,3)|A|=2, PSL(2,3)|A|=3      n ∈ {48,72}

Saves: figures/chain4_gap_scaling.png
"""
import sys
import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chains.codeword_walk import preprocess, enumerate_all_codewords
from codes.toric_code import build_toric_hx
from codes.qecc_comparison import bicycle_code
from codes.bb_code import bb_code
from codes.hgp_code import hgp, c523, rep
from codes.fb_code import fb_code
from codes.qt_code import build_psl2, qt_code, find_gen_index, mat_inv_mod

Q          = 0.3          # MH bias parameter
EXACT_THRESHOLD = 4096    # build exact matrix when 2^k_dim − 1 ≤ this
T_AUTO     = 80000        # chain length for empirical autocorrelation
BURN_IN    = 8000
OUT_FIGURE = os.path.join(os.path.dirname(__file__), "..", "..", "figures",
                          "chain4_gap_scaling.png")
OUT_JSON   = os.path.join(os.path.dirname(__file__), "chain4_gap_scaling.json")

FAM_COL = {"Toric": "#1f77b4", "Bicycle": "#2ca02c", "BB": "#e74c3c",
           "HGP": "#9467bd",   "FB": "#ff7f0e",      "QT": "#8c564b"}
FAM_MKR = {"Toric": "o", "Bicycle": "s", "BB": "D",
           "HGP": "^",   "FB": "P",      "QT": "*"}


# ── Chain 4 utilities ──────────────────────────────────────────────────────────

def chain4_step(c, fc, non_basis, q):
    j     = random.choice(non_basis)
    c_new = (c.astype(np.int32) + fc[j].astype(np.int32)) % 2
    c_new = c_new.astype(np.uint8)
    if not c_new.any():
        return c.copy()
    delta_w = int(c_new.sum()) - int(c.sum())
    if delta_w <= 0 or random.random() < q ** delta_w:
        return c_new
    return c.copy()


def build_chain4_matrix(codewords, fc, non_basis, q):
    """N × N transition matrix for Chain 4 (not symmetric)."""
    N   = len(codewords)
    idx = {bytes(c): i for i, c in enumerate(codewords)}
    P   = np.zeros((N, N), dtype=np.float64)
    p0  = 1.0 / len(non_basis)
    for i, c in enumerate(codewords):
        wt_c = int(c.sum())
        for j in non_basis:
            c_new = (c.astype(np.int32) + fc[j].astype(np.int32)) % 2
            c_new = c_new.astype(np.uint8)
            if not c_new.any():
                P[i, i] += p0
            else:
                delta_w = int(c_new.sum()) - wt_c
                alpha   = 1.0 if delta_w <= 0 else q ** delta_w
                k_idx   = idx.get(bytes(c_new))
                if k_idx is not None:
                    P[i, k_idx] += p0 * alpha
                    P[i, i]     += p0 * (1.0 - alpha)
    return P


def exact_gap(codewords, fc, non_basis, q):
    P    = build_chain4_matrix(codewords, fc, non_basis, q)
    eigs = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    return float(1.0 - eigs[1]) if len(eigs) > 1 else 1.0


def empirical_gap(fc, non_basis, q, T=T_AUTO, burn_in=BURN_IN, seed=42):
    """Gap estimate from autocorrelation of wt(c_t) under stationary π."""
    random.seed(seed)
    c = fc[random.choice(non_basis)].copy()
    for _ in range(burn_in):
        c = chain4_step(c, fc, non_basis, q)
    weights = np.empty(T, dtype=np.float64)
    for t in range(T):
        c        = chain4_step(c, fc, non_basis, q)
        weights[t] = float(c.sum())
    x   = weights - weights.mean()
    var = np.var(x)
    if var < 1e-10:
        return np.inf, 0.0
    max_lag = min(600, T // 10)
    rho = np.array([np.mean(x[:T - lag] * x[lag:]) / var
                    for lag in range(max_lag)])
    pos = rho > 0.05
    if pos.sum() < 3:
        return 1.0, 1.0
    try:
        popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau),
                            np.where(pos)[0], rho[pos],
                            p0=[10.0], bounds=(0.1, 5000))
        tau = float(popt[0])
    except Exception:
        tau = float(0.5 + np.sum(rho[1:]))
    return tau, 1.0 / tau if tau > 0 else 0.0


def analyse(H_X, q=Q, seed=42):
    basis, non_basis, r, fc = preprocess(H_X)
    n       = H_X.shape[1]
    k_dim   = len(non_basis)
    N_cw    = 2 ** k_dim - 1

    if N_cw <= EXACT_THRESHOLD:
        codewords = enumerate_all_codewords(fc, non_basis, n)
        delta     = exact_gap(codewords, fc, non_basis, q)
        tau       = None
        method    = "exact"
    else:
        tau, delta = empirical_gap(fc, non_basis, q, seed=seed)
        method = "empirical"

    return {"n": n, "k_dim": k_dim, "N_cw": N_cw,
            "delta": delta, "tau": tau, "method": method}


# ── Code instances ─────────────────────────────────────────────────────────────

def get_instances():
    insts = []

    for L in [2, 3, 4, 5]:
        insts.append(("Toric", f"Toric L={L}", build_toric_hx(L)))

    for rr in [6, 9, 12, 15, 18]:
        insts.append(("Bicycle", f"Bicycle r={rr}",
                      bicycle_code(rr, [0, 1, 2], [0, 2, 3])))

    for l, m, a_sh, b_sh in [
        (3, 3, [(0,0),(1,0),(0,1)], [(0,0),(2,0),(0,2)]),
        (6, 3, [(0,0),(2,0),(0,1)], [(0,0),(4,0),(0,2)]),
    ]:
        try:
            H_X, _ = bb_code(l, m, a_sh, b_sh)
            insts.append(("BB", f"BB({l},{m})", H_X))
        except AssertionError as e:
            print(f"  BB({l},{m}): CSS check failed — skipping ({e})")

    for L in [3, 4, 5]:
        H_X, _ = hgp(c523(), rep(L))
        insts.append(("HGP", f"HGP(C523,rep{L})", H_X))

    rng_fb = np.random.default_rng(42)
    for label, r_fb, s_fb, shifts in [
        ("FB-uniform(3,3)", 3, 3, [1]*3),
        ("FB-alt(4,4)",     4, 4, [i % 2 for i in range(4)]),
        ("FB-alt(6,6)",     6, 6, [i % 2 for i in range(6)]),
    ]:
        try:
            H_X, _ = fb_code(r_fb, s_fb, shifts)
            insts.append(("FB", label, H_X))
        except AssertionError as e:
            print(f"  {label}: CSS failed — skipping ({e})")

    elems, elem_index, mul_table = build_psl2(3)
    s_mat = np.array([[1, 1], [0, 1]]); t_mat = np.array([[0, 2], [1, 0]])
    s_inv = mat_inv_mod(s_mat, 3)
    s_idx  = find_gen_index(s_mat, 3, elem_index)
    si_idx = find_gen_index(s_inv, 3, elem_index)
    t_idx  = find_gen_index(t_mat, 3, elem_index)
    if None not in (s_idx, si_idx, t_idx):
        for A_idx, B_idx, glab in [
            ([s_idx, si_idx], [s_idx, si_idx], "|A|=2"),
            ([s_idx, si_idx, t_idx], [s_idx, si_idx, t_idx], "|A|=3"),
        ]:
            try:
                H_X, _ = qt_code(mul_table, A_idx, B_idx)
                insts.append(("QT", f"QT-PSL(2,3) {glab}", H_X))
            except AssertionError as e:
                print(f"  QT-PSL(2,3) {glab}: CSS failed — skipping ({e})")

    return insts


# ── Load or compute ────────────────────────────────────────────────────────────

print(f"Chain 4 gap scaling  (q={Q})")

if os.path.exists(OUT_JSON):
    print(f"Loading cached results from {OUT_JSON}")
    with open(OUT_JSON) as f:
        results = json.load(f)
    # N_cw may have been stored as a large int; ensure it stays int-compatible
    for r in results:
        r.setdefault("tau", None)
else:
    print("=" * 70)
    print(f"{'label':<24}  {'n':>4}  {'k_dim':>6}  {'N_cw':>8}  "
          f"{'δ':>10}  {'method'}")
    print("=" * 70)

    instances = get_instances()
    results   = []

    for seed_idx, (family, label, H_X) in enumerate(instances):
        print(f"  {label:<24} n={H_X.shape[1]:<4}...", end="  ", flush=True)
        res = analyse(H_X, q=Q, seed=100 + seed_idx * 13)
        res["family"] = family
        res["label"]  = label
        results.append(res)
        tau_s = f"τ={res['tau']:.1f}" if res["tau"] is not None else "τ=—"
        print(f"k={res['k_dim']:3d}  N_cw={res['N_cw']:<8d}  "
              f"δ={res['delta']:.5f}  {tau_s}  [{res['method']}]")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCached → {OUT_JSON}")


# ── Table ──────────────────────────────────────────────────────────────────────

print()
print(f"{'family':<10} {'label':<24} {'n':>4} {'k_dim':>6} "
      f"{'delta':>10} {'method'}")
print("-" * 62)
for r in results:
    print(f"{r['family']:<10} {r['label']:<24} {r['n']:>4} "
          f"{r['k_dim']:>6} {r['delta']:>10.5f} {r['method']}")


# ── Power-law fits δ ~ C · n^{-γ} ────────────────────────────────────────────

print()
print("Power-law fits  δ ~ n^{-γ}:")
print(f"  {'family':<10}  {'γ':>7}  {'R²':>6}  {'pts':>4}")
print("  " + "-" * 35)

gamma_fits = {}
for family in ["Toric", "Bicycle", "BB", "HGP", "FB", "QT"]:
    pts = [(r["n"], r["delta"]) for r in results
           if r["family"] == family and r["delta"] > 1e-8
           and np.isfinite(r["delta"])]
    if len(pts) < 2:
        gamma_fits[family] = None
        continue
    ns  = np.array([p[0] for p in pts], dtype=float)
    ds  = np.array([p[1] for p in pts], dtype=float)
    try:
        popt, _ = curve_fit(lambda ln, lc, g: lc - g * ln,
                            np.log(ns), np.log(ds), p0=[0.0, 1.0])
        log_c, gamma = popt
        pred  = log_c - gamma * np.log(ns)
        ss_res = np.sum((np.log(ds) - pred) ** 2)
        ss_tot = np.sum((np.log(ds) - np.log(ds).mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        gamma_fits[family] = (float(gamma), r2, float(np.exp(log_c)))
        n_pts = len(pts)
        marker = "✓" if r2 > 0.85 else "~"
        print(f"  {family:<10}  {gamma:>7.3f}  {r2:>6.3f}  {n_pts:>4}  {marker}")
    except Exception as e:
        gamma_fits[family] = None
        print(f"  {family:<10}  {'—':>7}  {'—':>6}  {len(pts):>4}  (fit failed: {e})")


# ── Save JSON ──────────────────────────────────────────────────────────────────

compact = [{"label": r["label"], "family": r["family"], "n": r["n"],
            "k_dim": r["k_dim"], "N_cw": r["N_cw"],
            "delta": round(r["delta"], 8), "method": r["method"]}
           for r in results]
with open(OUT_JSON, "w") as f:
    json.dump(compact, f, indent=2)
print(f"\nJSON  → {OUT_JSON}")


# ── Figure ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 6))

seen_fam = set()
for r in results:
    fam = r["family"]
    lbl = fam if fam not in seen_fam else None
    kw  = dict(color=FAM_COL[fam], marker=FAM_MKR[fam], s=90,
               zorder=4, label=lbl)
    if r["method"] == "exact":
        ax.scatter(r["n"], r["delta"], **kw, edgecolors="k", linewidths=0.8)
    else:
        ax.scatter(r["n"], r["delta"], **kw, alpha=0.7, edgecolors="none")
    seen_fam.add(fam)

# Power-law fit lines per family
n_range = np.linspace(6, 80, 200)
for family, fit in gamma_fits.items():
    if fit is None:
        continue
    gamma, r2, C = fit
    if r2 < 0.5:
        continue
    y_fit = C * n_range ** (-gamma)
    ax.loglog(n_range, y_fit, "-", color=FAM_COL[family],
              lw=1.5, alpha=0.55)

# Annotate γ on the plot (for families with good fits)
for family, fit in gamma_fits.items():
    if fit is None:
        continue
    gamma, r2, C = fit
    if r2 < 0.5:
        continue
    # Place label at the right end of the fit line
    n_label = 70.0
    if n_label > max(r["n"] for r in results if r["family"] == family) * 1.5:
        n_label = max(r["n"] for r in results if r["family"] == family) * 1.2
    y_label = C * n_label ** (-gamma)
    ax.text(n_label * 1.03, y_label,
            fr"$\gamma={gamma:.2f}$", color=FAM_COL[family],
            fontsize=8.5, va="center")

# Reference lines
n_ref = np.array([8, 80], dtype=float)
ax.loglog(n_ref, 0.4 / n_ref,       "k:", lw=1, alpha=0.5, label=r"$\sim n^{-1}$")
ax.loglog(n_ref, 1.5 / n_ref ** 2,  "k--", lw=1, alpha=0.4, label=r"$\sim n^{-2}$")

# Legend distinguishing exact vs empirical
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles, labels = ax.get_legend_handles_labels()
handles += [
    Line2D([0], [0], marker="o", color="gray", mfc="gray", mec="k",
           linewidth=0, ms=8, label="exact"),
    Line2D([0], [0], marker="o", color="gray", mfc="gray", mec="none",
           linewidth=0, ms=8, alpha=0.7, label="empirical"),
]
labels += ["exact (filled)", "empirical (open)"]
ax.legend(handles, labels, fontsize=8.5, ncol=2, loc="upper right",
          framealpha=0.9)

ax.set_xlabel(r"Code length $n$", fontsize=13)
ax.set_ylabel(r"Spectral gap  $\delta$", fontsize=13)
ax.set_title(
    fr"Chain 4 (biased codeword walk, $q={Q}$): $\delta \sim n^{{-\gamma}}$",
    fontsize=12)
ax.grid(True, which="both", alpha=0.2, lw=0.7)
ax.tick_params(labelsize=11)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_FIGURE), exist_ok=True)
fig.savefig(OUT_FIGURE, dpi=180, bbox_inches="tight")
print(f"Figure → {OUT_FIGURE}")
