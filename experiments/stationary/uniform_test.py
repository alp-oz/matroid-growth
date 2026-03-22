"""
Verify that MHCircuitChain converges to uniform, and compare empirical
stationary distributions of adjacent, MH-uniform, and MH-size-weighted chains.
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

from core.engine import MatroidEngine
from core.markov_chain import MarkovChainCircuits, MHCircuitChain, fundamental_circuits
from core.circuits import all_circuits
from core.stationary import build_transition_matrix, stationary_distribution


# ── Setup ─────────────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

engine = MatroidEngine(n_steps=20, k_params=2, C=0.1, beta=0.8, start_r=2)
result = engine.run()
M, r, n = result["M"], result["r"], result["n"]

all_c, _, _ = all_circuits(M, r, mode='global')
circuits     = sorted(all_c, key=sorted)
idx          = {c: i for i, c in enumerate(circuits)}
N            = len(circuits)
print(f"Matroid: rank={r}, #elements={n}, #circuits={N}")

# Exact stationary distributions
P_adj = build_transition_matrix(M, r, circuits, mode='adjacent')
pi_adj = stationary_distribution(P_adj)
pi_uniform = np.ones(N) / N

# ── Empirical distributions via long run ─────────────────────────────────────
N_STEPS = 200_000
start = circuits[0]

def empirical_pi(chain, n_steps, start, idx, N):
    counts = np.zeros(N)
    traj = chain.run(n_steps=n_steps, start=start)
    for c in traj:
        i = idx.get(c)
        if i is not None:
            counts[i] += 1
    return counts / counts.sum()

print(f"\nRunning {N_STEPS:,} steps each (may take ~30s)...")

chain_adj  = MarkovChainCircuits(M, r, mode='adjacent')
chain_mh0  = MHCircuitChain(M, r, alpha=0)   # targets uniform
chain_mh1  = MHCircuitChain(M, r, alpha=1)   # targets ∝ |C|

random.seed(1); pi_emp_adj  = empirical_pi(chain_adj,  N_STEPS, start, idx, N)
random.seed(2); pi_emp_mh0  = empirical_pi(chain_mh0,  N_STEPS, start, idx, N)
random.seed(3); pi_emp_mh1  = empirical_pi(chain_mh1,  N_STEPS, start, idx, N)

sizes = np.array([len(c) for c in circuits])

# ── L1 distances to uniform ───────────────────────────────────────────────────
def l1(p, q): return np.sum(np.abs(p - q))

print(f"\n  L1 distance to uniform:")
print(f"    adjacent (exact)   : {l1(pi_adj,     pi_uniform):.4f}")
print(f"    adjacent (empirical): {l1(pi_emp_adj, pi_uniform):.4f}")
print(f"    MH α=0  (empirical): {l1(pi_emp_mh0, pi_uniform):.4f}")
print(f"    MH α=1  (empirical): {l1(pi_emp_mh1, pi_uniform):.4f}")

# ── Acceptance rates ──────────────────────────────────────────────────────────
def acceptance_rate(chain, n_steps, start):
    random.seed(99)
    accepts = 0
    state = start
    for _ in range(n_steps):
        new = chain.step(state)
        if new != state:
            accepts += 1
        state = new
    return accepts / n_steps

acc_mh0 = acceptance_rate(MHCircuitChain(M, r, alpha=0), 50_000, start)
acc_mh1 = acceptance_rate(MHCircuitChain(M, r, alpha=1), 50_000, start)
print(f"\n  MH acceptance rates:")
print(f"    α=0 (uniform target): {acc_mh0:.3f}")
print(f"    α=1 (size target)   : {acc_mh1:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Stationary distribution: adjacent vs MH-corrected chains", fontsize=13, fontweight='bold')

unique_sizes = sorted(set(sizes))
labels  = ["adjacent", "MH α=0\n(uniform)", "MH α=1\n(∝|C|)"]
pis_emp = [pi_emp_adj, pi_emp_mh0, pi_emp_mh1]
colors  = ["#d62728", "#2ca02c", "#9467bd"]

for ax, pi_emp, label, col in zip(axes, pis_emp, labels, colors):
    # Median π per size
    med = [np.median(pi_emp[sizes == sz]) for sz in unique_sizes]
    ax.bar(unique_sizes, med, color=col, alpha=0.75, edgecolor='white', label=label)
    ax.axhline(1/N, color='black', lw=1.5, ls='--', label=f'uniform 1/N={1/N:.4f}')

    ax.set_xlabel("Circuit size |C|", fontsize=11)
    ax.set_ylabel("Median empirical π", fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    l1_val = l1(pi_emp, pi_uniform)
    ax.text(0.97, 0.97, f"L1={l1_val:.4f}", transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

plt.tight_layout()
plt.savefig("markov-circuits/uniform_comparison.png", dpi=150, bbox_inches='tight')
print("\nSaved → markov-circuits/uniform_comparison.png")
plt.close()
