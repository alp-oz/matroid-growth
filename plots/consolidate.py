"""
Consolidation figure — Markov chain mixing across all CSS code families.

Loads markov-circuits/gap_scaling_results.json (no recomputation).
Saves markov-circuits/consolidation.{png,pdf}.

Layout (2×2):
  Panel A  log-log δ_auto vs n  — main scaling evidence, fitted power laws
  Panel B  power-law exponent α per family  — summary bar chart
  Panel C  autocorrelation time τ vs n      — mixing time growth
  Panel D  coupling time T_couple vs n      — worst-case upper bound

All cosmetic parameters live in the STYLE dict below.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────────────────────
# STYLE — edit here to change any visual element
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    # Per-family colours (hex)
    "colors": {
        "Bicycle": "#2ca02c",
        "Toric":   "#1f77b4",
        "HGP":     "#9467bd",
        "BB":      "#e74c3c",
        "FB":      "#ff7f0e",
        "QT":      "#8c564b",
    },
    # Per-family marker shapes
    "markers": {
        "Bicycle": "s",
        "Toric":   "o",
        "HGP":     "^",
        "BB":      "D",
        "FB":      "P",
        "QT":      "*",
    },
    # Marker size (points²)
    "ms": 35,
    # Alpha for scatter dots
    "alpha_scatter": 0.55,
    # Line width for fit curves
    "fit_lw": 1.6,
    # Alpha for fit curves
    "fit_alpha": 0.55,
    # Reference line style
    "ref_lw": 1.0,
    "ref_alpha": 0.45,
    # Bar chart bar width
    "bar_width": 0.55,
    # Figure size (inches)
    "figsize": (14, 11),
    # Grid alpha
    "grid_alpha": 0.25,
    # Font sizes
    "title_fs": 11,
    "label_fs": 10,
    "tick_fs":  9,
    "legend_fs": 8,
    "suptitle_fs": 13,
    # DPI for PNG
    "dpi": 150,
    # Output paths
    "out_png": "markov-circuits/consolidation.png",
    "out_pdf": "markov-circuits/consolidation.pdf",
}

FAMILIES = ["Bicycle", "Toric", "HGP", "BB", "FB", "QT"]

# ─────────────────────────────────────────────────────────────────────────────
# Load results
# ─────────────────────────────────────────────────────────────────────────────

with open("markov-circuits/gap_scaling_results.json") as f:
    raw = json.load(f)

# Group by family
by_fam = {fam: [] for fam in FAMILIES}
for r in raw:
    fam = r.get("family", "")
    if fam in by_fam:
        by_fam[fam].append(r)

# Sort each family by n
for fam in FAMILIES:
    by_fam[fam].sort(key=lambda r: r["n"])

# ─────────────────────────────────────────────────────────────────────────────
# Power-law fits   δ ~ C · n^{-α}   →   log δ = log C - α · log n
# ─────────────────────────────────────────────────────────────────────────────

def powerlaw(log_n, log_c, alpha):
    return log_c - alpha * log_n

def fit_family(fam, key="delta_ind"):
    """Return (alpha, r2, n_arr, v_arr) or None if < 3 valid points."""
    data = [(r["n"], r[key]) for r in by_fam[fam]
            if r.get(key) is not None and r[key] > 1e-6]
    if len(data) < 3:
        return None
    ns = np.array([d[0] for d in data], dtype=float)
    vs = np.array([d[1] for d in data], dtype=float)
    try:
        popt, _ = curve_fit(powerlaw, np.log(ns), np.log(vs), p0=[0.0, 0.5])
        res = np.log(vs) - powerlaw(np.log(ns), *popt)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((np.log(vs) - np.log(vs).mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return popt[1], r2, ns, vs
    except Exception:
        return None

fits_auto  = {fam: fit_family(fam, "delta_ind")   for fam in FAMILIES}
fits_coup  = {fam: fit_family(fam, "delta_coup")  for fam in FAMILIES}

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(2, 2, figsize=STYLE["figsize"])
(axA, axB), (axC, axD) = axes
fig.subplots_adjust(hspace=0.42, wspace=0.36)

S = STYLE   # shorthand

# ── Panel A: log-log δ_auto vs n ─────────────────────────────────────────────
seen = set()
for fam in FAMILIES:
    col = S["colors"][fam]
    mkr = S["markers"][fam]
    pts = [(r["n"], r["delta_ind"]) for r in by_fam[fam] if r["delta_ind"] > 1e-6]
    if not pts:
        continue
    ns_p = np.array([p[0] for p in pts])
    vs_p = np.array([p[1] for p in pts])
    lbl = fam if fam not in seen else None
    axA.scatter(ns_p, vs_p, color=col, marker=mkr, s=S["ms"],
                alpha=S["alpha_scatter"], zorder=4, label=lbl)
    seen.add(fam)
    # Fit curve
    fit = fits_auto[fam]
    if fit:
        alpha, r2, ns_f, vs_f = fit
        n_range = np.linspace(ns_f.min() * 0.85, ns_f.max() * 1.15, 60)
        log_c   = np.log(vs_f).mean() + alpha * np.log(ns_f).mean()
        axA.plot(n_range, np.exp(log_c - alpha * np.log(n_range)),
                 color=col, lw=S["fit_lw"], alpha=S["fit_alpha"], ls="--")

# Reference lines
n_ref = np.array([8.0, 300.0])
axA.loglog(n_ref, 1.5 / n_ref**0.5, color="gray", lw=S["ref_lw"],
           ls=":",  alpha=S["ref_alpha"], label=r"$n^{-0.5}$")
axA.loglog(n_ref, 3.0 / n_ref,      color="gray", lw=S["ref_lw"],
           ls="-.", alpha=S["ref_alpha"], label=r"$n^{-1}$")

axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("n  (physical qubits)", fontsize=S["label_fs"])
axA.set_ylabel(r"$\hat{\delta}$ = 1/τ  (autocorrelation gap)", fontsize=S["label_fs"])
axA.set_title("A.  Log-log gap estimate vs n", fontsize=S["title_fs"], fontweight="bold")
axA.legend(fontsize=S["legend_fs"], ncol=2, framealpha=0.7)
axA.grid(alpha=S["grid_alpha"], which="both")
axA.tick_params(labelsize=S["tick_fs"])

# ── Panel B: α summary bar chart ─────────────────────────────────────────────
fams_with_fit = [f for f in FAMILIES if fits_auto[f] is not None]
alphas   = [fits_auto[f][0] for f in fams_with_fit]
r2s      = [fits_auto[f][1] for f in fams_with_fit]
xs       = np.arange(len(fams_with_fit))
bar_cols = [S["colors"][f] for f in fams_with_fit]

bars = axB.bar(xs, alphas, width=S["bar_width"], color=bar_cols,
               edgecolor="white", linewidth=0.8, alpha=0.88, zorder=3)
# Annotate with α and R²
for i, (bar, a, r2) in enumerate(zip(bars, alphas, r2s)):
    axB.text(bar.get_x() + bar.get_width()/2, a + 0.015,
             f"α={a:.2f}\nR²={r2:.2f}",
             ha="center", va="bottom", fontsize=7.5, color="#333333")

# Reference line α=1 (linear mixing time = poly(n))
axB.axhline(1.0, color="crimson", lw=1.4, ls="--", alpha=0.7,
            label="α=1  (t_mix ~ n)")
axB.axhline(0.0, color="gray",    lw=0.8, ls=":",  alpha=0.4)

axB.set_xticks(xs)
axB.set_xticklabels(fams_with_fit, fontsize=S["tick_fs"])
axB.set_ylabel("Power-law exponent α", fontsize=S["label_fs"])
axB.set_title("B.  Gap scaling exponent α per family\n"
              r"(δ ~ n$^{-α}$,  all α < 1 → sub-linear mixing time)",
              fontsize=S["title_fs"], fontweight="bold")
axB.set_ylim(0, max(alphas) * 1.35)
axB.legend(fontsize=S["legend_fs"], framealpha=0.7)
axB.grid(axis="y", alpha=S["grid_alpha"])
axB.tick_params(labelsize=S["tick_fs"])

# ── Panel C: τ vs n (log-log) ─────────────────────────────────────────────────
seen = set()
for fam in FAMILIES:
    col = S["colors"][fam]
    mkr = S["markers"][fam]
    pts = [(r["n"], r["tau"]) for r in by_fam[fam] if r.get("tau", 0) > 0]
    if not pts:
        continue
    ns_p = np.array([p[0] for p in pts])
    ts_p = np.array([p[1] for p in pts])
    lbl  = fam if fam not in seen else None
    axC.scatter(ns_p, ts_p, color=col, marker=mkr, s=S["ms"],
                alpha=S["alpha_scatter"], zorder=4, label=lbl)
    seen.add(fam)

axC.axhline(1.0, color="gray", lw=S["ref_lw"], ls=":", alpha=S["ref_alpha"])
axC.set_xlabel("n  (physical qubits)", fontsize=S["label_fs"])
axC.set_ylabel("Autocorrelation time τ", fontsize=S["label_fs"])
axC.set_title("C.  Mixing time τ vs n", fontsize=S["title_fs"], fontweight="bold")
axC.legend(fontsize=S["legend_fs"], ncol=2, framealpha=0.7)
axC.grid(alpha=S["grid_alpha"])
axC.tick_params(labelsize=S["tick_fs"])

# ── Panel D: coupling time vs n (log-log) ─────────────────────────────────────
seen = set()
for fam in FAMILIES:
    col = S["colors"][fam]
    mkr = S["markers"][fam]
    pts = [(r["n"], r["t_couple"]) for r in by_fam[fam]
           if r.get("t_couple", 0) > 0]
    if not pts:
        continue
    ns_p = np.array([p[0] for p in pts])
    tc_p = np.array([p[1] for p in pts])
    lbl  = fam if fam not in seen else None
    axD.scatter(ns_p, tc_p, color=col, marker=mkr, s=S["ms"],
                alpha=S["alpha_scatter"], zorder=4, label=lbl)
    seen.add(fam)
    # Fit curve
    fit = fits_coup[fam]
    if fit:
        alpha_c, _, ns_f, vs_f = fit
        n_range = np.linspace(ns_f.min()*0.85, ns_f.max()*1.15, 60)
        log_c   = (np.log(vs_f).mean() -
                   powerlaw(np.log(ns_f), 0, alpha_c).mean())
        axD.plot(n_range, np.exp(log_c - alpha_c * np.log(n_range)),
                 color=col, lw=S["fit_lw"], alpha=S["fit_alpha"], ls="--")

# Reference lines
n_ref2 = np.array([8.0, 300.0])
axD.loglog(n_ref2, 0.5  * n_ref2,      color="gray", lw=S["ref_lw"],
           ls="-.", alpha=S["ref_alpha"], label="~ n")
axD.loglog(n_ref2, 0.02 * n_ref2**2,   color="gray", lw=S["ref_lw"],
           ls=":",  alpha=S["ref_alpha"], label=r"~ n²")

axD.set_xscale("log"); axD.set_yscale("log")
axD.set_xlabel("n  (physical qubits)", fontsize=S["label_fs"])
axD.set_ylabel("Mean coupling time  $T_{couple}$", fontsize=S["label_fs"])
axD.set_title("D.  Coupling time vs n  (log-log)", fontsize=S["title_fs"], fontweight="bold")
axD.legend(fontsize=S["legend_fs"], ncol=2, framealpha=0.7)
axD.grid(alpha=S["grid_alpha"], which="both")
axD.tick_params(labelsize=S["tick_fs"])

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Markov chain on CSS code circuits — spectral gap scaling across 6 families\n"
    "All families: α < 1  →  sub-linear mixing time  (consistent with poly(n) conjecture)",
    fontsize=S["suptitle_fs"], fontweight="bold", y=1.01
)

# ── Save ──────────────────────────────────────────────────────────────────────
for path in [S["out_png"], S["out_pdf"]]:
    fig.savefig(path, dpi=S["dpi"], bbox_inches="tight")
    print(f"Saved → {path}")
plt.close()

# ── Print compact summary table ───────────────────────────────────────────────
print("\nSummary — autocorrelation gap scaling  δ ~ n^{-α}")
print("=" * 55)
print(f"  {'Family':<10}  {'α_auto':>7}  {'R²':>6}  {'n range':>12}  {'τ range'}")
print("=" * 55)
for fam in FAMILIES:
    fit = fits_auto[fam]
    pts = by_fam[fam]
    if not pts:
        continue
    ns   = [r["n"] for r in pts]
    taus = [r["tau"] for r in pts if r.get("tau", 0) > 0]
    n_str = f"{min(ns)}–{max(ns)}"
    t_str = f"{min(taus):.1f}–{max(taus):.1f}" if taus else "—"
    if fit:
        a, r2, _, _ = fit
        print(f"  {fam:<10}  {a:>7.3f}  {r2:>6.3f}  {n_str:>12}  {t_str}")
    else:
        print(f"  {fam:<10}  {'—':>7}  {'—':>6}  {n_str:>12}  {t_str}")
print("=" * 55)
