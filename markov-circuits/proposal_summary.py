"""
Research-proposal summary figure.

Three panels:
  1. Log-log: CRN coupling time vs n — polynomial fits, reference lines
  2. Exact spectral gap δ vs n — stays bounded away from 0
  3. Autocorrelation time τ vs n — roughly flat (O(1) or O(log n))

Plus a printed evidence table.

Runs the same sweep as gap_scaling.py but caches results, so a second
call reuses them.  Set RERUN=True below to force fresh computation.
"""
import numpy as np
import random
import pickle, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

from gap_scaling import build_sweep, analyse_instance

CACHE   = "markov-circuits/proposal_results.pkl"
RERUN   = False          # set True to recompute everything
OUT_FIG = "markov-circuits/proposal_summary.png"

FAM_COL = {"Bicycle": "#2ca02c", "Toric": "#1f77b4", "HGP": "#9467bd"}
FAM_MKR = {"Bicycle": "s",       "Toric": "o",       "HGP": "^"}
FAM_LABEL = {"Bicycle": "Bicycle (CSS quasi-cyclic)",
             "Toric":   "Toric code",
             "HGP":     "Hypergraph product"}


def run_sweep():
    instances = build_sweep()
    results   = []
    for label, M, r, n, family in instances:
        print(f"  {label}  (n={n})...", flush=True)
        res = analyse_instance(label, M, r, n,
                               T_auto=30000, n_pairs=40,
                               max_t_couple=4000, seed=42)
        res["family"] = family
        results.append(res)
    return results


# ── Load or compute ────────────────────────────────────────────────────────────
if (not RERUN) and os.path.exists(CACHE):
    print("Loading cached results …")
    with open(CACHE, "rb") as f:
        results = pickle.load(f)
else:
    print("Running sweep (this takes a few minutes) …")
    random.seed(42); np.random.seed(42)
    results = run_sweep()
    with open(CACHE, "wb") as f:
        pickle.dump(results, f)
    print("Saved to", CACHE)


# ── Power-law fit helper ───────────────────────────────────────────────────────
def fit_powerlaw_grow(ns, vals):
    """Fit log(val) = log(a) + alpha*log(n) for GROWING quantities (t_couple).
    Returns (alpha, R2) where positive alpha means val ~ n^alpha."""
    mask = (np.array(vals) > 1e-6) & np.isfinite(vals)
    if mask.sum() < 3:
        return None, None
    ln = np.log(np.array(ns)[mask])
    lv = np.log(np.array(vals)[mask])
    def f(x, loga, alpha): return loga + alpha * x
    try:
        popt, _ = curve_fit(f, ln, lv, p0=[0.0, 1.5])
        resid   = lv - f(ln, *popt)
        r2      = 1 - resid.var() / lv.var() if lv.var() > 0 else 0
        return float(popt[1]), float(r2)
    except Exception:
        return None, None


# ── Evidence table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("Evidence for poly(n) mixing — key numbers")
print("=" * 90)
print(f"{'Code':<22}  {'n':>4}  {'N':>7}  {'δ_exact':>8}  "
      f"{'τ_auto':>7}  {'t_crn':>7}  {'t_ind':>7}  {'t_crn/n':>8}")
print("-" * 90)
for r in results:
    N_s   = str(r["N"])       if r["N"]        is not None else "—"
    dg_s  = f"{r['gap_exact']:.3f}" if r["gap_exact"] is not None else "—"
    tau_s = f"{r['tau']:.1f}"
    tcrn_s = f"{r['t_crn_mean']:.1f}"
    tind_s = f"{r['t_couple_mean']:.1f}"
    ratio  = r["t_crn_mean"] / r["n"]
    print(f"  {r['label']:<22}  {r['n']:>4}  {N_s:>7}  {dg_s:>8}  "
          f"{tau_s:>7}  {tcrn_s:>7}  {tind_s:>7}  {ratio:>8.2f}")
print("=" * 90)

print("\nPower-law fits  t_couple ~ n^α:")
for family in ["Bicycle", "Toric", "HGP"]:
    fr = [r for r in results if r["family"] == family]
    ns   = [r["n"]             for r in fr]
    tind = [r["t_couple_mean"] for r in fr]
    a_ind, r2_ind = fit_powerlaw_grow(ns, tind)
    if a_ind is not None:
        print(f"  {family:<10}  t_couple ~ n^{a_ind:.2f}  (R²={r2_ind:.3f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5.5))
gs  = gridspec.GridSpec(1, 3, wspace=0.40)

n_ref = np.linspace(7, 60, 200)


# ── Panel 1: Log-log independent coupling time vs n ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

for family in ["Bicycle", "Toric", "HGP"]:
    fr    = [r for r in results if r["family"] == family]
    ns    = np.array([r["n"]             for r in fr], dtype=float)
    ts    = np.array([r["t_couple_mean"] for r in fr], dtype=float)
    order = np.argsort(ns)
    ax1.loglog(ns[order], ts[order],
               FAM_MKR[family] + "-",
               color=FAM_COL[family], lw=2, ms=7,
               label=FAM_LABEL[family])
    # Annotate fitted slope
    alpha, r2 = fit_powerlaw_grow(ns, ts)
    if alpha is not None:
        mid_i = len(ns) // 2
        ax1.annotate(f"α≈{alpha:.2f}",
                     xy=(ns[order][mid_i], ts[order][mid_i]),
                     xytext=(5, 3), textcoords="offset points",
                     fontsize=9, color=FAM_COL[family], fontweight="bold")

# Reference lines
ax1.loglog(n_ref, 0.2 * n_ref**2, "k:", lw=1.3, alpha=0.65, label="∝ n²")
ax1.loglog(n_ref, 0.01 * n_ref**3, "k--", lw=1.0, alpha=0.40, label="∝ n³")

ax1.set_xlabel("Code length  n", fontsize=12)
ax1.set_ylabel("Mean coupling time", fontsize=12)
ax1.set_title("Coupling time ~ poly(n)\n"
              "(log-log, slope α  annotated per family)",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=8.5, loc="upper left")
ax1.grid(alpha=0.3, which="both")


# ── Panel 2: Exact spectral gap δ vs n ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

seen = set()
for r in results:
    if r["gap_exact"] is None:
        continue
    fam = r["family"]
    lbl = FAM_LABEL[fam] if fam not in seen else None
    ax2.scatter(r["n"], r["gap_exact"],
                color=FAM_COL[fam], marker=FAM_MKR[fam],
                s=90, zorder=4, label=lbl)
    # Annotate N
    ax2.annotate(f"N={r['N']}",
                 xy=(r["n"], r["gap_exact"]),
                 xytext=(4, 3), textcoords="offset points",
                 fontsize=7.5, color="gray")
    seen.add(fam)

# 1/n reference
ax2.plot(n_ref, 1.5 / n_ref, "k:", lw=1.2, alpha=0.55, label="1.5/n")
ax2.set_xlabel("Code length  n", fontsize=12)
ax2.set_ylabel("Exact spectral gap  δ", fontsize=12)
ax2.set_title("Exact gap stays bounded away from 0\n"
              "(N grows exponentially; δ does not vanish)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=8.5)
ax2.set_ylim(bottom=0)
ax2.grid(alpha=0.3)


# ── Panel 3: Autocorrelation time τ vs n ──────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])

seen = set()
for family in ["Bicycle", "Toric", "HGP"]:
    fr  = [r for r in results if r["family"] == family]
    ns  = np.array([r["n"] for r in fr], dtype=float)
    tau = np.array([r["tau"] for r in fr], dtype=float)
    order = np.argsort(ns)
    lbl = FAM_LABEL[family]
    ax3.plot(ns[order], tau[order],
             FAM_MKR[family] + "-",
             color=FAM_COL[family], lw=2, ms=7,
             label=lbl)

# Flat reference
ax3.axhline(4, color="gray", ls="--", lw=1, alpha=0.6, label="τ = 4 (constant)")
ax3.set_xlabel("Code length  n", fontsize=12)
ax3.set_ylabel("Autocorrelation time  τ", fontsize=12)
ax3.set_title("Element-indicator autocorrelation\nremains near-constant in n",
              fontsize=11, fontweight="bold")
ax3.set_ylim(bottom=0)
ax3.legend(fontsize=8.5)
ax3.grid(alpha=0.3)


fig.suptitle(
    "Evidence for poly(n) mixing of the circuit Markov chain on CSS codes\n"
    "Bicycle (quasi-cyclic), toric, and hypergraph-product families  ·  "
    "n = 8 – 50, N up to ~15 000 circuits",
    fontsize=12, fontweight="bold", y=1.02
)

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_FIG}")
plt.close()
