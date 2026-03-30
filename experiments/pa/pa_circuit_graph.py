"""
Circuit intersection graph of the PA binary matroid.

Vertices = fundamental circuits  fc[j] for each non-basis column j.
Edge (i,j): fc[i] ∩ fc[j] ≠ ∅  ⟺  bitset[i] & bitset[j] ≠ 0
            (the two circuits share at least one basis row).

Note: enumerating ALL circuits is exponential in r; fundamental circuits
are the natural generating set (n_att ≈ 2982 vertices for n_steps=3000, γ=1).

Parameters: γ=1, λ∈{0.5,1.0,2.0}, β∈{0.5,1.0,1.5,2.0},
            k=4, n_steps=3000, start_r=10, 10 replicates.

Plot 1 (circuit_graph_component.png):
    Largest component fraction vs β, one curve per λ.

Plot 2 (circuit_graph_degree.png):
    Degree distribution log-log for each β at fixed λ=1.0.

Plot 3 (circuit_graph_density.png):
    Largest component fraction vs attachment density ρ = n_t/r_t,
    for each β (density sweep by varying n_steps).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import os

from core.engine import MatroidEngine
from analysis.probe_minors import convert_to_bitsets

# ── Config ─────────────────────────────────────────────────────────────────────
LAMBDA_VALUES = [0.5, 1.0, 2.0]
BETA_VALUES   = [0.5, 1.0, 1.5, 2.0]
LAM_FIXED     = 1.0
N_STEPS       = 3000
START_R       = 10
K             = 4
GAMMA         = 1.0
REPS          = 10

# Density sweep: log-spaced n_steps
_raw = np.round(np.logspace(1.3, np.log10(N_STEPS), 22)).astype(int)
N_STEPS_SWEEP = sorted(set(_raw.tolist()))

REPS_SWEEP = 5   # replicates per density-sweep point (adjacency is O(n²))

BETA_COLORS  = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
LAM_LS       = ["-", "--", ":"]
LAM_COLORS   = ["#2c3e50", "#8e44ad", "#c0392b"]
OUT          = os.path.dirname(__file__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def circuit_graph_metrics(columns):
    """
    Given a list of attachment column supports (list of sorted row-index lists),
    build the fundamental-circuit intersection graph and return:
      - largest_frac : size of largest component / total vertices
      - degrees      : numpy array of vertex degrees
    Returns (None, None) if < 2 circuits.
    """
    if len(columns) < 2:
        return None, None

    bits = np.array(convert_to_bitsets(columns), dtype=np.int64)
    n = len(bits)

    # Adjacency: A[i,j] = True iff bits[i] & bits[j] != 0  (shared basis row)
    # Compute in one numpy broadcast; bool dtype keeps memory at n²/8 bytes.
    # For n=3000: 9M bool ≈ 9 MB — fine.
    adj_int = (bits[:, None] & bits[None, :])   # (n, n) int64
    adj = adj_int.astype(bool)
    np.fill_diagonal(adj, False)                 # no self-loops

    # Degree of each vertex
    degrees = adj.sum(axis=1)                    # (n,) int64

    # Connected components via scipy sparse
    sparse_adj = csr_matrix(adj)
    n_comp, labels = connected_components(sparse_adj, directed=False,
                                          return_labels=True)
    comp_sizes = np.bincount(labels)
    largest_frac = float(comp_sizes.max()) / n

    return largest_frac, degrees


def run_one(n_steps, beta, lam, seed):
    np.random.seed(seed)
    return MatroidEngine(n_steps=n_steps, k_params=K, C=lam, gamma=GAMMA,
                         beta=beta, start_r=START_R).run()


# ── Main runs: all (λ, β) at n_steps=N_STEPS ──────────────────────────────────
print(f"Main runs: {len(LAMBDA_VALUES)} λ × {len(BETA_VALUES)} β × {REPS} reps …")

main_results = {}  # (lam, beta) → {lf: [largest_fracs], deg: [degree arrays]}

for lam in LAMBDA_VALUES:
    for beta in BETA_VALUES:
        lf_list, deg_list = [], []
        for seed in range(REPS):
            data = run_one(N_STEPS, beta, lam, seed=seed * 37)
            cols = data["columns"]
            lf, deg = circuit_graph_metrics(cols)
            if lf is not None:
                lf_list.append(lf)
                deg_list.append(deg)
        main_results[(lam, beta)] = {
            "lf_mean": float(np.mean(lf_list)) if lf_list else 0.0,
            "lf_std":  float(np.std(lf_list))  if lf_list else 0.0,
            "degrees": np.concatenate(deg_list) if deg_list else np.array([]),
        }
        r_ex = run_one(N_STEPS, beta, lam, seed=0)["r"]
        n_ex = run_one(N_STEPS, beta, lam, seed=0)["n"]
        print(f"  λ={lam}, β={beta}: lf={main_results[(lam,beta)]['lf_mean']:.3f}"
              f"  (r≈{r_ex}, n≈{n_ex}, ρ≈{n_ex/r_ex:.0f})")

# ── Density sweep: vary n_steps, λ=LAM_FIXED, all β ──────────────────────────
print(f"\nDensity sweep: {len(N_STEPS_SWEEP)} n_steps × {len(BETA_VALUES)} β "
      f"× {REPS_SWEEP} reps …")

sweep = {}  # (n_steps, beta) → {rho, lf_mean, lf_std}

for n_s in N_STEPS_SWEEP:
    for beta in BETA_VALUES:
        lf_list, rhos = [], []
        for seed in range(REPS_SWEEP):
            data = run_one(n_s, beta, LAM_FIXED, seed=seed * 37)
            r_fin, n_fin = data["r"], data["n"]
            cols = data["columns"]
            lf, _ = circuit_graph_metrics(cols)
            if lf is not None:
                lf_list.append(lf)
                rhos.append(n_fin / r_fin if r_fin > 0 else 0)
        sweep[(n_s, beta)] = {
            "rho":     float(np.mean(rhos))    if rhos    else 0.0,
            "lf_mean": float(np.mean(lf_list)) if lf_list else 0.0,
            "lf_std":  float(np.std(lf_list))  if lf_list else 0.0,
        }
    print(f"  n_steps={n_s:5d}  done", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Largest component fraction vs β  (one curve per λ)
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlot 1 …")
fig, ax = plt.subplots(figsize=(8, 5.5))

for lam, ls, color in zip(LAMBDA_VALUES, LAM_LS, LAM_COLORS):
    means = [main_results[(lam, b)]["lf_mean"] for b in BETA_VALUES]
    stds  = [main_results[(lam, b)]["lf_std"]  for b in BETA_VALUES]
    ax.errorbar(BETA_VALUES, means, yerr=stds,
                fmt="o" + ls, color=color, lw=2, ms=7, capsize=4,
                label=f"λ={lam}")

ax.set_xlabel("Attachment bias  β", fontsize=12)
ax.set_ylabel("Largest component fraction", fontsize=12)
ax.set_title(
    "Circuit intersection graph — largest component\n"
    f"γ={GAMMA}, k={K}, n_steps={N_STEPS}, start_r={START_R}, {REPS} reps",
    fontsize=11, fontweight="bold"
)
ax.set_ylim(-0.03, 1.05)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "circuit_graph_component.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → circuit_graph_component.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Degree distribution log-log  (fixed λ=1.0, all β)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 2 …")
fig, ax = plt.subplots(figsize=(8, 5.5))

for beta, color in zip(BETA_VALUES, BETA_COLORS):
    degs = main_results[(LAM_FIXED, beta)]["degrees"]
    if len(degs) == 0:
        continue
    counts = np.bincount(degs.astype(int))
    k_vals = np.where(counts > 0)[0]
    c_vals = counts[k_vals]
    ax.loglog(k_vals, c_vals, "o", color=color, ms=4, alpha=0.7, label=f"β={beta}")
    # Fit power law to the tail (k ≥ 10th percentile of k_vals)
    if len(k_vals) >= 5:
        lo = int(np.percentile(k_vals, 10))
        mask = k_vals >= max(lo, 1)
        if mask.sum() >= 3:
            try:
                log_k = np.log(k_vals[mask].astype(float))
                log_c = np.log(c_vals[mask].astype(float))
                coeffs = np.polyfit(log_k, log_c, 1)
                alpha_fit = -coeffs[0]
                k_fit = np.array([k_vals[mask][0], k_vals[mask][-1]], dtype=float)
                c_fit = np.exp(coeffs[1]) * k_fit ** coeffs[0]
                ax.loglog(k_fit, c_fit, "-", color=color, lw=1.5, alpha=0.8,
                          label=f"  α={alpha_fit:.2f}")
            except Exception:
                pass

ax.set_xlabel("Degree  k", fontsize=12)
ax.set_ylabel("Count  P(k)", fontsize=12)
ax.set_title(
    "Degree distribution of circuit intersection graph\n"
    f"γ={GAMMA}, λ={LAM_FIXED}, k_attach={K}, n_steps={N_STEPS}, start_r={START_R}",
    fontsize=11, fontweight="bold"
)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "circuit_graph_degree.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → circuit_graph_degree.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Largest component fraction vs density  (one curve per β)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 3 …")
fig, ax = plt.subplots(figsize=(8, 5.5))

for beta, color in zip(BETA_VALUES, BETA_COLORS):
    pts = sorted(
        [(sweep[(n_s, beta)]["rho"], sweep[(n_s, beta)]["lf_mean"],
          sweep[(n_s, beta)]["lf_std"])
         for n_s in N_STEPS_SWEEP],
        key=lambda x: x[0]
    )
    rhos   = [p[0] for p in pts]
    means  = [p[1] for p in pts]
    stds   = [p[2] for p in pts]
    ax.errorbar(rhos, means, yerr=stds,
                fmt="o-", color=color, lw=2, ms=5, capsize=3,
                label=f"β={beta}")

ax.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5)
ax.set_xlabel("Attachment density  ρ = n_t / r_t", fontsize=12)
ax.set_ylabel("Largest component fraction", fontsize=12)
ax.set_title(
    "Circuit intersection graph — percolation vs density\n"
    f"γ={GAMMA}, λ={LAM_FIXED}, k={K}, start_r={START_R}, {REPS_SWEEP} reps",
    fontsize=11, fontweight="bold"
)
ax.set_ylim(-0.03, 1.05)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "circuit_graph_density.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → circuit_graph_density.png")
print("\nAll done.")
