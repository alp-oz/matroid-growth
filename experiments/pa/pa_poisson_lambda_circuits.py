"""
Circuit size distribution: Poisson k, varying λ and β.

γ=0, k_params=('poisson', 4), n_steps=3000, start_r=10
λ ∈ {0.05, 0.2, 0.5},  β ∈ {0.5, 1.0, 1.5},  10 replicates.

Layout: 3 columns (one per λ), overlaid bars per β.
Each panel also shows mean circuit size in the legend.

Saves: poisson_lambda_circuit_distribution.png
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

from core.engine import MatroidEngine

LAMBDAS = [0.05, 0.2, 0.5]
BETAS   = [0.5, 1.0, 1.5]
N_STEPS = 3000
START_R = 10
K_LAM   = 4
REPS    = 10
COLORS  = ["#2980b9", "#27ae60", "#e67e22"]   # one per β
OUT     = os.path.dirname(__file__)


def minimal_circuit_counts(columns):
    support_count = Counter(frozenset(col) for col in columns)
    counts = Counter()
    for support, m in support_count.items():
        if m == 1:
            counts[len(support) + 1] += 1
        else:
            counts[2] += m * (m - 1) // 2
    return counts


# ── Collect data ─────────────────────────────────────────────────────────────

# results[lam][beta] = (Counter, mean_size)
results = {}
for lam in LAMBDAS:
    results[lam] = {}
    for beta in BETAS:
        agg = Counter()
        for _ in range(REPS):
            eng  = MatroidEngine(n_steps=N_STEPS, k_params=("poisson", K_LAM),
                                 C=lam, gamma=0.0, beta=beta, start_r=START_R)
            data = eng.run()
            agg.update(minimal_circuit_counts(data["columns"]))
        total = sum(agg.values())
        mean  = sum(s * agg[s] for s in agg) / total if total else 0
        results[lam][beta] = (agg, mean)
        print(f"  λ={lam}  β={beta}  mean_size={mean:.3f}")

# Global size range
all_sizes = sorted({s for lam in LAMBDAS for beta in BETAS
                    for s in results[lam][beta][0]})
max_size  = max(all_sizes)
plot_sizes = list(range(2, min(max_size + 1, 16)))   # cap display at 15

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

n_beta  = len(BETAS)
width   = 0.22
offsets = np.linspace(-(n_beta - 1) / 2, (n_beta - 1) / 2, n_beta) * width

for ax, lam in zip(axes, LAMBDAS):
    for i, (beta, color) in enumerate(zip(BETAS, COLORS)):
        agg, mean = results[lam][beta]
        total = sum(agg.values())
        fracs = [agg[s] / total if total else 0 for s in plot_sizes]
        ax.bar([s + offsets[i] for s in plot_sizes], fracs, width=width,
               color=color, alpha=0.85,
               label=fr"$\beta={beta}$  ($\bar{{|C|}}={mean:.2f}$)")

    ax.set_xticks([s for s in plot_sizes if s <= 12])
    ax.set_xlabel("Circuit size $|C|$", fontsize=11)
    ax.set_ylabel("Fraction", fontsize=11)
    ax.set_title(fr"$\lambda={lam}$  ($\rho\approx{1/lam:.0f}$)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle(
    r"Circuit size distribution: $\gamma=0$, $k\sim\mathrm{Poisson}(4)$, "
    r"$n_\mathrm{steps}=3000$",
    fontsize=13)
plt.tight_layout()
path = os.path.join(OUT, "poisson_lambda_circuit_distribution.png")
fig.savefig(path, dpi=150)
print(f"\n  → {path}")

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'λ':>5}  {'β':>5}  {'mean':>7}  {'f2':>7}  {'f3':>7}  {'f5':>7}  {'f≥6':>7}")
for lam in LAMBDAS:
    for beta in BETAS:
        agg, mean = results[lam][beta]
        total = sum(agg.values())
        f  = lambda s: agg[s] / total if total else 0
        fge6 = sum(agg[s] for s in agg if s >= 6) / total if total else 0
        print(f"{lam:>5}  {beta:>5}  {mean:>7.3f}  {f(2):>7.3f}  "
              f"{f(3):>7.3f}  {f(5):>7.3f}  {fge6:>7.3f}")
    print()
