"""
Experiment 3: Circuit size distribution.

γ=0, λ=0.05, k=4, n_steps=3000, start_r=10
β ∈ {0.5, 1.0, 1.5, 2.0}, 10 replicates.

For the binary matroid [I_r | A] with k=4 fixed, the minimal circuits are:
  Size 2 — parallel pairs: two non-basis columns with identical support
            (identical 4-element row sets).  High β creates PA hubs that
            attract many columns to the same top-4 rows.
  Size 5 — fundamental circuits (unique support): {non-basis col} ∪ {4 rows}.
            These are minimal iff no parallel pair exists for that support.

Left panel : fraction of circuits at each size, grouped by β.
Right panel: mean circuit size vs β.

Saves: circuit_distribution.png
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

from core.engine import MatroidEngine

LAMBDA  = 0.05
BETAS   = [0.5, 1.0, 1.5, 2.0]
N_STEPS = 3000
START_R = 10
K       = 4
REPS    = 10
COLORS  = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
OUT     = os.path.dirname(__file__)


def minimal_circuit_counts(columns):
    """
    Count minimal circuits by size.

    Groups non-basis columns by their support (frozenset of row indices).
    A group of size m contributes:
      - C(m, 2) circuits of size 2  (parallel pairs — minimal)
      - m circuits of size 5 only if m == 1 (fundamental circuit is minimal
        only when its support is unique; if m >= 2 the size-2 pairs are proper
        subsets of the size-5 set, making it non-minimal).
    """
    support_count = Counter(frozenset(col) for col in columns)
    counts = Counter()
    for m in support_count.values():
        if m == 1:
            counts[5] += 1
        else:
            counts[2] += m * (m - 1) // 2
    return counts


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

mean_sizes = []
all_counts_per_beta = []

for beta in BETAS:
    agg = Counter()
    for _ in range(REPS):
        eng  = MatroidEngine(n_steps=N_STEPS, k_params=K, C=LAMBDA,
                             gamma=0.0, beta=beta, start_r=START_R)
        data = eng.run()
        agg.update(minimal_circuit_counts(data["columns"]))
    all_counts_per_beta.append(agg)
    total = sum(agg.values())
    mean_sizes.append(sum(s * agg[s] for s in agg) / total if total else 0)

# Left: grouped bar chart
sizes = [2, 5]
n_beta = len(BETAS)
width  = 0.18
offsets = np.linspace(-(n_beta - 1) / 2, (n_beta - 1) / 2, n_beta) * width

for i, (beta, color, agg) in enumerate(zip(BETAS, COLORS, all_counts_per_beta)):
    total = sum(agg.values())
    fracs = [agg[s] / total if total else 0 for s in sizes]
    ax1.bar([s + offsets[i] for s in sizes], fracs, width=width,
            color=color, alpha=0.85, label=fr"$\beta={beta}$")

ax1.set_xticks(sizes)
ax1.set_xlabel("Circuit size $|C|$", fontsize=12)
ax1.set_ylabel("Fraction of minimal circuits", fontsize=12)
ax1.set_title(r"Circuit size distribution ($\gamma=0$, $\lambda=0.05$, $k=4$)",
              fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, axis="y", alpha=0.3)

# Right: mean size vs beta
ax2.plot(BETAS, mean_sizes, "k-", lw=2, zorder=2)
for b, m, c in zip(BETAS, mean_sizes, COLORS):
    ax2.scatter([b], [m], color=c, zorder=5, s=70)
ax2.set_xlabel(r"$\beta$", fontsize=12)
ax2.set_ylabel("Mean circuit size", fontsize=12)
ax2.set_title("Mean circuit size vs PA strength", fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(OUT, "circuit_distribution.png")
fig.savefig(path, dpi=150)
print(f"  → {path}")

print(f"\n{'β':>5}  {'mean_size':>10}  {'frac_size2':>11}  {'frac_size5':>11}")
for beta, agg, ms in zip(BETAS, all_counts_per_beta, mean_sizes):
    total = sum(agg.values())
    f2 = agg[2] / total if total else 0
    f5 = agg[5] / total if total else 0
    print(f"{beta:>5.1f}  {ms:>10.3f}  {f2:>11.3f}  {f5:>11.3f}")
