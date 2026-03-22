import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

from core.engine import MatroidEngine
from core.markov_chain import fundamental_circuits, decompose_into_circuits
from core.circuits import all_circuits
from core.stationary import (
    build_transition_matrix,
    stationary_distribution,
    circuit_features,
)


# ── Reproducible setup ───────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

engine = MatroidEngine(n_steps=30, k_params=2, C=0.1, beta=0.8, start_r=2)
result = engine.run()
M, r, n = result["M"], result["r"], result["n"]
print(f"Matroid: rank={r}, #elements={n}, #non-basis={n - r}")

all_c, _, _ = all_circuits(M, r, mode='global')
circuits = sorted(all_c, key=sorted)
print(f"Total circuits: {len(circuits)}")

P_global   = build_transition_matrix(M, r, circuits, mode='global')
P_adjacent = build_transition_matrix(M, r, circuits, mode='adjacent')
pi_global   = stationary_distribution(P_global)
pi_adjacent = stationary_distribution(P_adjacent)

print("Computing features...")
feats = circuit_features(M, r, circuits)
sizes = feats["size"].astype(int)
unique_sizes = sorted(set(sizes))


# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

COLORS = {
    'global':   '#1f77b4',
    'adjacent': '#d62728',
}
ALPHA = 0.45
MS    = 18   # marker size for scatter


# ── 1. π vs size — scatter (log scale), both modes ───────────────────────────
ax1 = fig.add_subplot(gs[0, :])
jitter = np.random.uniform(-0.18, 0.18, len(circuits))
ax1.scatter(sizes + jitter, pi_global,   s=MS, alpha=ALPHA,
            color=COLORS['global'],   label='global',   zorder=3)
ax1.scatter(sizes + jitter, pi_adjacent, s=MS, alpha=ALPHA,
            color=COLORS['adjacent'], label='adjacent', zorder=3)

# Median lines per size
for sz in unique_sizes:
    mask = sizes == sz
    for pi, col in [(pi_global, COLORS['global']),
                    (pi_adjacent, COLORS['adjacent'])]:
        med = np.median(pi[mask])
        ax1.plot([sz - 0.3, sz + 0.3], [med, med], color=col, lw=2.5, zorder=4)

ax1.set_yscale('log')
ax1.set_xlabel('Circuit size  |C|', fontsize=12)
ax1.set_ylabel('Stationary probability  π  (log)', fontsize=12)
ax1.set_title('Stationary probability vs circuit size', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, which='both', alpha=0.3)

rho_g = spearmanr(sizes, pi_global).statistic
rho_a = spearmanr(sizes, pi_adjacent).statistic
ax1.text(0.98, 0.95,
         f'Spearman ρ\n  global:   {rho_g:.3f}\n  adjacent: {rho_a:.3f}',
         transform=ax1.transAxes, ha='right', va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# ── 2. Circuit count per size ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
counts = [np.sum(sizes == sz) for sz in unique_sizes]
ax2.bar(unique_sizes, counts, color='steelblue', edgecolor='white')
ax2.set_xlabel('Circuit size  |C|', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Number of circuits per size', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for sz, cnt in zip(unique_sizes, counts):
    ax2.text(sz, cnt + 0.5, str(cnt), ha='center', va='bottom', fontsize=9)


# ── 3. Median π per size (both modes) ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
med_g = [np.median(pi_global[sizes == sz])   for sz in unique_sizes]
med_a = [np.median(pi_adjacent[sizes == sz]) for sz in unique_sizes]
ax3.plot(unique_sizes, med_g, 'o-', color=COLORS['global'],
         label='global',   lw=2, ms=7)
ax3.plot(unique_sizes, med_a, 's-', color=COLORS['adjacent'],
         label='adjacent', lw=2, ms=7)
ax3.set_yscale('log')
ax3.set_xlabel('Circuit size  |C|', fontsize=11)
ax3.set_ylabel('Median π  (log)', fontsize=11)
ax3.set_title('Median stationary probability per size', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, which='both', alpha=0.3)


# ── 4. π_global vs π_adjacent scatter (coloured by size) ─────────────────────
ax4 = fig.add_subplot(gs[2, 0])
sc = ax4.scatter(pi_global, pi_adjacent, c=sizes, cmap='viridis',
                 s=22, alpha=0.6, zorder=3)
cbar = plt.colorbar(sc, ax=ax4)
cbar.set_label('Circuit size', fontsize=10)

lims = [min(pi_global.min(), pi_adjacent.min()) * 0.5,
        max(pi_global.max(), pi_adjacent.max()) * 1.5]
ax4.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='equal')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('π  global  (log)', fontsize=11)
ax4.set_ylabel('π  adjacent  (log)', fontsize=11)
ax4.set_title('π_global vs π_adjacent\n(coloured by circuit size)', fontsize=12,
              fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, which='both', alpha=0.3)


# ── 5. Ratio π_global / π_adjacent vs size ───────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ratio = pi_global / (pi_adjacent + 1e-300)
ax5.scatter(sizes + jitter, ratio, s=MS, alpha=ALPHA, color='purple', zorder=3)

med_ratio = [np.median(ratio[sizes == sz]) for sz in unique_sizes]
ax5.plot(unique_sizes, med_ratio, 'o-', color='black', lw=2, ms=7,
         label='median ratio')
ax5.axhline(1.0, color='gray', lw=1.2, linestyle='--', label='ratio = 1')
ax5.set_yscale('log')
ax5.set_xlabel('Circuit size  |C|', fontsize=11)
ax5.set_ylabel('π_global / π_adjacent  (log)', fontsize=11)
ax5.set_title('Ratio of stationary probabilities\nglobal ÷ adjacent by size',
              fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, which='both', alpha=0.3)


# ── Save ─────────────────────────────────────────────────────────────────────
out = "markov-circuits/stationary_vs_size.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out}")
plt.close()


# ── Console summary ───────────────────────────────────────────────────────────
print(f"\n{'Size':>6}  {'Count':>6}  {'Med π global':>14}  {'Med π adj':>12}  {'Ratio':>8}")
print("-" * 52)
for sz, mg, ma in zip(unique_sizes, med_g, med_a):
    cnt = np.sum(sizes == sz)
    print(f"{sz:>6}  {cnt:>6}  {mg:>14.6f}  {ma:>12.6f}  {mg/ma:>8.3f}")
