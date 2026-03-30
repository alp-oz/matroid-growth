"""
Support frequency / Zipf experiment.

γ=0, k=4, n_steps=5000, start_r=10
varying β ∈ {0.2, 0.5, 1.0, 1.5, 2.0}  and  C ∈ {0.05, 0.2, 0.5}

Three panels (one per C), each with 5 Zipf curves (one per β), log-log.
Power-law exponent s fitted for each and shown in the legend.

Saves: zipf_beta_C.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

from core.engine import MatroidEngine
from analysis.stats import get_zipf_distribution

BETA_VALUES = [0.2, 0.5, 1.0, 1.5, 2.0]
C_VALUES    = [0.05, 0.2, 0.5]
N_STEPS     = 5000
START_R     = 10
K           = 4
SEED        = 42

COLORS = plt.cm.plasma(np.linspace(0.1, 0.9, len(BETA_VALUES)))


def fit_zipf(ranks, counts):
    log_r, log_c = np.log(ranks.astype(float)), np.log(counts.astype(float))
    def f(x, log_a, s): return log_a - s * x
    try:
        popt, _ = curve_fit(f, log_r, log_c, p0=[log_c[0], 1.0])
        resid = log_c - f(log_r, *popt)
        r2 = 1 - resid.var() / log_c.var() if log_c.var() > 0 else 0.0
        return float(popt[1]), float(r2)
    except Exception:
        return np.nan, np.nan


fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)

for ax, C in zip(axes, C_VALUES):
    for beta, color in zip(BETA_VALUES, COLORS):
        np.random.seed(SEED)
        data = MatroidEngine(n_steps=N_STEPS, k_params=K, C=C, gamma=0.0,
                             beta=beta, start_r=START_R).run()
        ranks, counts = get_zipf_distribution(data["row_usage"])
        s, r2 = fit_zipf(ranks, counts)

        label = f"β={beta:.1f}  s={s:.2f} (R²={r2:.2f})"
        ax.loglog(ranks, counts, "o", color=color, ms=3.5, alpha=0.65, label=label)

        # Fitted line anchored at the first point
        if not np.isnan(s):
            a = counts[0] * ranks[0] ** s
            rr = np.array([ranks[0], ranks[-1]], dtype=float)
            ax.loglog(rr, a * rr ** (-s), "-", color=color, lw=1.6, alpha=0.9)

        r_final = data["r"]
        print(f"  C={C}, β={beta:.1f}: s={s:.3f}, R²={r2:.3f}, r_final={r_final}")

    ax.set_xlabel("Rank", fontsize=11)
    ax.set_ylabel("Row usage (frequency)", fontsize=11)
    ax.set_title(f"C = {C}  (r_final ≈ {START_R + int(C * N_STEPS)})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)

fig.suptitle(
    f"Support frequency / Zipf's law  (γ=0, k={K}, n_steps={N_STEPS}, start_r={START_R})\n"
    "Does the exponent s depend on β? Does C shift s or only scale the distribution?",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "zipf_beta_C.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out}")
