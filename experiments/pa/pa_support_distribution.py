"""
Experiment 1: Support frequency distribution — clean Zipf figure.

γ=0, λ=0.05, k=4, n_steps=5000, start_r=10
β ∈ {0.5, 1.0, 1.5}, 1 replicate each.

Plots log-log support frequency vs coordinate rank with power-law fits.
Saves: support_distribution.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from core.engine import MatroidEngine

LAMBDA  = 0.05
BETAS   = [0.5, 1.0, 1.5]
N_STEPS = 5000
START_R = 10
K       = 4
COLORS  = ["#2980b9", "#27ae60", "#e74c3c"]
OUT     = os.path.dirname(__file__)

np.random.seed(42)

fig, ax = plt.subplots(figsize=(6, 5))

for beta, color in zip(BETAS, COLORS):
    eng  = MatroidEngine(n_steps=N_STEPS, k_params=K, C=LAMBDA,
                         gamma=0.0, beta=beta, start_r=START_R)
    data = eng.run()
    usage = data["row_usage"]

    sorted_u = np.sort(usage)[::-1]
    ranks    = np.arange(1, len(sorted_u) + 1)

    slope, intercept = np.polyfit(np.log(ranks), np.log(sorted_u), 1)

    ax.loglog(ranks, sorted_u, '.', color=color, alpha=0.5, markersize=3)
    ax.loglog(ranks, np.exp(intercept) * ranks ** slope, '-', color=color,
              lw=2, label=fr"$\beta={beta}$, $s={-slope:.2f}$")

ax.set_xlabel("Coordinate rank", fontsize=12)
ax.set_ylabel("Support frequency", fontsize=12)
ax.set_title(r"Support frequency distribution ($\gamma=0$, $\lambda=0.05$, $k=4$)",
             fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT, "support_distribution.png")
fig.savefig(path, dpi=150)
print(f"  → {path}")
