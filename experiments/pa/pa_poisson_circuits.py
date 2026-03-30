"""
Circuit size distribution with Poisson-distributed attachment size.

γ=0, λ=0.05, k_params=('poisson', 4), n_steps=3000, start_r=10
β ∈ {0.5, 1.0, 1.5, 2.0}, 10 replicates.

With k ~ Poisson(4), each non-basis column has a random support size,
so fundamental circuits have size k+1 ∈ {2, 3, 4, 5, ...} rather than
the fixed size-5 of the k=4 case.

Minimal circuit counting:
  - Columns grouped by their support (frozenset of row indices).
  - Group of size m=1: one fundamental circuit of size len(support)+1.
  - Group of size m≥2: C(m,2) parallel pairs of size 2.
    (The size-(k+1) fundamental circuit is non-minimal here.)

Left panel  : bar chart of circuit size distribution, one group per β.
Right panel : mean circuit size vs β.

Saves: poisson_circuit_distribution.png
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
K_LAM   = 4                        # Poisson mean
REPS    = 10
COLORS  = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
OUT     = os.path.dirname(__file__)


def minimal_circuit_counts(columns):
    """
    Count minimal circuits by size for [I_r | A] with variable support sizes.

    Groups by frozenset support; m=1 → fundamental circuit of size k+1;
    m≥2 → C(m,2) parallel pairs of size 2 (fundamental circuit non-minimal).
    """
    support_count = Counter(frozenset(col) for col in columns)
    counts = Counter()
    for support, m in support_count.items():
        if m == 1:
            counts[len(support) + 1] += 1
        else:
            counts[2] += m * (m - 1) // 2
    return counts


# ── Run experiments ─────────────────────────────────────────────────────────

all_counts = []
mean_sizes  = []

for beta in BETAS:
    agg = Counter()
    for _ in range(REPS):
        eng  = MatroidEngine(n_steps=N_STEPS, k_params=("poisson", K_LAM),
                             C=LAMBDA, gamma=0.0, beta=beta, start_r=START_R)
        data = eng.run()
        agg.update(minimal_circuit_counts(data["columns"]))
    all_counts.append(agg)
    total = sum(agg.values())
    mean_sizes.append(sum(s * agg[s] for s in agg) / total if total else 0)

# ── Determine x-axis range across all β ─────────────────────────────────────

all_sizes = sorted({s for agg in all_counts for s in agg})

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

n_beta  = len(BETAS)
width   = 0.18
offsets = np.linspace(-(n_beta - 1) / 2, (n_beta - 1) / 2, n_beta) * width

for i, (beta, color, agg) in enumerate(zip(BETAS, COLORS, all_counts)):
    total = sum(agg.values())
    fracs = [agg[s] / total if total else 0 for s in all_sizes]
    ax1.bar([s + offsets[i] for s in all_sizes], fracs, width=width,
            color=color, alpha=0.85, label=fr"$\beta={beta}$")

ax1.set_xticks(all_sizes)
ax1.set_xlabel("Circuit size $|C|$", fontsize=12)
ax1.set_ylabel("Fraction of minimal circuits", fontsize=12)
ax1.set_title(
    r"Circuit size distribution ($\gamma=0$, $\lambda=0.05$, $k\sim\mathrm{Poisson}(4)$)",
    fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, axis="y", alpha=0.3)

ax2.plot(BETAS, mean_sizes, "k-", lw=2, zorder=2)
for b, m, c in zip(BETAS, mean_sizes, COLORS):
    ax2.scatter([b], [m], color=c, zorder=5, s=70)
ax2.set_xlabel(r"$\beta$", fontsize=12)
ax2.set_ylabel("Mean circuit size", fontsize=12)
ax2.set_title(r"Mean circuit size vs PA strength ($k\sim\mathrm{Poisson}(4)$)",
              fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(OUT, "poisson_circuit_distribution.png")
fig.savefig(path, dpi=150)
print(f"  → {path}")

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'β':>5}  {'mean_size':>10}  " +
      "  ".join(f"frac_{s}" for s in all_sizes))
for beta, agg, ms in zip(BETAS, all_counts, mean_sizes):
    total = sum(agg.values())
    fracs = [f"{agg[s]/total:.3f}" if total else "0.000" for s in all_sizes]
    print(f"{beta:>5.1f}  {ms:>10.3f}  " + "  ".join(f"{f:>7}" for f in fracs))
