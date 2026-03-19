"""
Zipf's law in row usage for the PA binary matroid.

For each value of β, run the PA engine and plot the rank-frequency distribution
of row usage on log-log axes.  Fit a power law  usage ~ rank^{-s}  and show
how the Zipf exponent s grows with β.

Two panels:
  Left  — log-log rank vs. frequency for five β values (main Zipf plot)
  Right — fitted exponent s vs. β with error bars across replicates
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from matroid_core.engine import MatroidEngine
from analysis.stats import get_zipf_distribution
from scipy.optimize import curve_fit

# ── Configuration ─────────────────────────────────────────────────────────────
BETA_VALUES  = [0.2, 0.5, 1.0, 1.5, 2.0]
N_STEPS      = 3000    # columns ≈ N_STEPS  (C very small → almost no row growth)
START_R      = 50
K            = 4
C            = 0.0001
REPLICATES   = 8       # for the exponent-vs-beta panel
SEED         = 42

COLORS = plt.cm.plasma(np.linspace(0.15, 0.85, len(BETA_VALUES)))

np.random.seed(SEED)

def fit_zipf_exponent(ranks, counts):
    """Fit counts ~ ranks^{-s} on log-log scale.  Returns (s, r2)."""
    log_r = np.log(ranks.astype(float))
    log_c = np.log(counts.astype(float))
    def f(x, log_a, s): return log_a - s * x
    try:
        popt, _ = curve_fit(f, log_r, log_c, p0=[np.log(counts[0]), 1.0])
        resid = log_c - f(log_r, *popt)
        r2    = 1 - resid.var() / log_c.var() if log_c.var() > 0 else 0
        return float(popt[1]), float(r2)
    except Exception:
        return np.nan, np.nan


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Panel 1: log-log Zipf plots ────────────────────────────────────────────────
print("Panel 1: Zipf distributions …")
for beta, color in zip(BETA_VALUES, COLORS):
    engine = MatroidEngine(n_steps=N_STEPS, k_params=K, C=C, gamma=0,
                           beta=beta, start_r=START_R)
    data  = engine.run()
    ranks, counts = get_zipf_distribution(data['row_usage'])

    s, r2 = fit_zipf_exponent(ranks, counts)
    label = f"β={beta:.1f}  (s={s:.2f})"

    ax1.loglog(ranks, counts, "o", color=color, ms=4, alpha=0.7, label=label)

    # Fitted power-law line
    if not np.isnan(s):
        n_fit = np.array([ranks[0], ranks[-1]], dtype=float)
        # anchor through the first data point
        a = counts[0] * ranks[0]**s
        ax1.loglog(n_fit, a * n_fit**(-s), "-", color=color, lw=1.5, alpha=0.9)

    print(f"  β={beta:.1f}: s={s:.3f},  R²={r2:.3f}")

ax1.set_xlabel("Rank", fontsize=12)
ax1.set_ylabel("Row usage (frequency)", fontsize=12)
ax1.set_title("Zipf's law in row usage\nPA binary matroid, " +
              f"n≈{N_STEPS}, r={START_R}, k={K}", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, which="both", alpha=0.3)

# ── Panel 2: fitted exponent s vs β ───────────────────────────────────────────
print("\nPanel 2: exponent vs β …")
mean_s, std_s = [], []

for beta in BETA_VALUES:
    slopes = []
    for rep in range(REPLICATES):
        np.random.seed(rep * 100)
        engine = MatroidEngine(n_steps=N_STEPS, k_params=K, C=C, gamma=0,
                               beta=beta, start_r=START_R)
        data  = engine.run()
        ranks, counts = get_zipf_distribution(data['row_usage'])
        s, _ = fit_zipf_exponent(ranks, counts)
        if not np.isnan(s):
            slopes.append(s)
    mean_s.append(float(np.mean(slopes)) if slopes else np.nan)
    std_s.append(float(np.std(slopes))  if slopes else np.nan)
    print(f"  β={beta:.1f}: s̄={mean_s[-1]:.3f} ± {std_s[-1]:.3f}")

mean_s = np.array(mean_s); std_s = np.array(std_s)
finite = np.isfinite(mean_s)

ax2.errorbar(np.array(BETA_VALUES)[finite], mean_s[finite], yerr=std_s[finite],
             fmt="o-", color="#333333", capsize=4, lw=2, ms=7,
             label="fitted Zipf exponent s")

# Theoretical slope for PA-like models: s ≈ 1/β in the high-β limit
beta_th = np.linspace(0.2, 2.0, 100)
ax2.plot(beta_th, beta_th, "r--", lw=1.2, alpha=0.6, label="s = β (reference)")
ax2.plot(beta_th, np.ones_like(beta_th), "k:", lw=1, alpha=0.5, label="s = 1 (Zipf's law)")

ax2.set_xlabel("Attachment bias  β", fontsize=12)
ax2.set_ylabel("Fitted Zipf exponent  s", fontsize=12)
ax2.set_title("How β controls the\nrow-usage power law", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xlim(0.0, 2.2); ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.35)

fig.suptitle("Preferential attachment creates power-law row-usage  (Zipf's law)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "zipf_law.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close()
