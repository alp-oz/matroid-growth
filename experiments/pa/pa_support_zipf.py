"""
Support frequency Zipf plot: log-log, β ∈ {0.5, 1.0, 1.5}.
γ=0, k=4 (fixed), λ=0.05, n_steps=5000, start_r=10, 1 replicate each.
Saves: support_zipf.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import linregress
from core.engine import MatroidEngine

BETAS   = [0.5, 1.0, 1.5]
COLORS  = {0.5: "#2c7bb6", 1.0: "#e08214", 1.5: "#1a9641"}
OUT     = os.path.dirname(__file__)

fig, ax = plt.subplots(figsize=(8, 6))

for beta in BETAS:
    eng = MatroidEngine(
        n_steps=5000, k_params=4,
        gamma=0.0, C=0.05, beta=beta,
        start_r=10,
    )
    eng.run()

    # Count support frequencies
    freq = Counter(frozenset(s) for s in eng.attachment_supports)
    counts = np.array(sorted(freq.values(), reverse=True), dtype=float)
    ranks  = np.arange(1, len(counts) + 1, dtype=float)

    # Log-log linear fit (only ranks with count ≥ 2 to avoid noise floor)
    mask = counts >= 2
    if mask.sum() > 5:
        slope, intercept, *_ = linregress(np.log(ranks[mask]),
                                          np.log(counts[mask]))
    else:
        slope, intercept, *_ = linregress(np.log(ranks), np.log(counts))

    col = COLORS[beta]
    ax.loglog(ranks, counts, ".", color=col, ms=4, alpha=0.6)
    # Fit line over full plotted range
    r_fit = np.array([ranks[0], ranks[-1]])
    ax.loglog(r_fit, np.exp(intercept) * r_fit ** slope, "-",
              color=col, lw=2.2,
              label=fr"$\beta={beta}$,  $s={abs(slope):.2f}$")

ax.set_xlabel("Support rank", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
ax.tick_params(labelsize=13)
ax.legend(fontsize=13, framealpha=0.9)
ax.grid(True, which="both", alpha=0.25, lw=0.7)
ax.set_title(r"Support frequency: $\gamma=0$, $k=4$, $\lambda=0.05$, $n=5000$",
             fontsize=13)

plt.tight_layout()
path = os.path.join(OUT, "support_zipf.png")
fig.savefig(path, dpi=180)
print(f"→ {path}")
for beta in BETAS:
    print(f"  β={beta}  done (see legend for s)")
