"""
Experiment 2: Triangle minor threshold — clean sigmoid figure.

γ=0, λ=0.05, k=4, start_r=10
β ∈ {0.7, 1.0, 1.3, 1.6}
n_steps swept over 20 log-spaced values in [50, 5000], 25 replicates each.

X-axis: attachment density ρ_t = n_t / r_t (mean over replicates).
Y-axis: P(triangle minor).

A triangle minor exists iff three distinct non-zero column bitsets a, b, c
satisfy a ⊕ b = c (GF(2) linear dependence of size 3).

Saves: minor_threshold.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from core.engine import MatroidEngine
from analysis.probe_minors import convert_to_bitsets

LAMBDA      = 0.05
BETAS       = [0.7, 1.0, 1.3, 1.6]
N_STEPS_ARR = np.unique(
    np.round(np.logspace(np.log10(50), np.log10(5000), 20)).astype(int)
).tolist()
START_R = 10
K       = 4
REPS    = 25
COLORS  = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
OUT     = os.path.dirname(__file__)


def has_triangle(columns):
    if not columns:
        return False
    bits = list(set(convert_to_bitsets(columns)) - {0})
    if len(bits) < 3:
        return False
    lookup = set(bits)
    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            c = bits[i] ^ bits[j]
            if c != 0 and c in lookup:
                return True
    return False


fig, ax = plt.subplots(figsize=(6, 5))

for beta, color in zip(BETAS, COLORS):
    rho_vals  = []
    prob_vals = []
    for n_steps in N_STEPS_ARR:
        hits    = 0
        rho_sum = 0.0
        for _ in range(REPS):
            eng  = MatroidEngine(n_steps=n_steps, k_params=K, C=LAMBDA,
                                 gamma=0.0, beta=beta, start_r=START_R)
            data = eng.run()
            rho_sum += data["n"] / data["r"]
            if has_triangle(data["columns"]):
                hits += 1
        rho_vals.append(rho_sum / REPS)
        prob_vals.append(hits / REPS)
    ax.plot(rho_vals, prob_vals, "o-", color=color, lw=2, markersize=5,
            label=fr"$\beta={beta}$")

ax.axhline(0.5, color="gray", ls="--", alpha=0.5, lw=1)
ax.set_xlabel(r"Attachment density $\rho_t = n_t/r_t$", fontsize=12)
ax.set_ylabel(r"$P(\mathrm{triangle\ minor})$", fontsize=12)
ax.set_title(r"Triangle minor threshold ($\gamma=0$, $\lambda=0.05$, $k=4$)",
             fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT, "minor_threshold.png")
fig.savefig(path, dpi=150)
print(f"  → {path}")
