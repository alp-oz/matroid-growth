"""
Circuit intersection graph — γ=0 regime (non-trivial percolation).

Vertices = fundamental circuits fc[j] for each non-basis column j.
Edge (i,j): fc[i] ∩ fc[j] ≠ ∅  ⟺  bitset[i] & bitset[j] ≠ 0.

With γ=0, C=λ, density ρ = n/r → 1/λ asymptotically:
  λ=0.05 → ρ≈20,  λ=0.2 → ρ≈5,  λ=0.5 → ρ≈2
Basis size r_final ≈ 10 + λ·3000, so the circuit intersection graph
is sparse enough that percolation is non-trivial.

All fundamental circuits have size k+1 = 5 (k=4 rows selected without
replacement; start_r=10 > k=4 so cap never activates).

Parameters: γ=0, λ∈{0.05,0.2,0.5}, β∈{0.5,1.0,1.5,2.0},
            k=4, n_steps=3000, start_r=10, 10 replicates.

Plot 1 (circuit_percolation_component.png):
    Largest component fraction vs β, one curve per λ.

Plot 2 (circuit_percolation_degree.png):
    Degree distribution log-log for each β at fixed λ=0.05.

Plot 3 (circuit_percolation_density.png):
    Largest component fraction vs ρ = n_t/r_t for each β at λ=0.05
    (density swept by varying n_steps).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import os

from core.engine import MatroidEngine

# ── Config ─────────────────────────────────────────────────────────────────────
LAMBDA_VALUES = [0.05, 0.2, 0.5]
BETA_VALUES   = [0.5, 1.0, 1.5, 2.0]
LAM_FIXED     = 0.05
N_STEPS       = 3000
START_R       = 10
K             = 4
GAMMA         = 0.0
REPS          = 10

# Density sweep at λ=LAM_FIXED: vary n_steps → ρ varies from ~2 to ~20
_raw = np.round(np.logspace(1.3, np.log10(N_STEPS), 24)).astype(int)
N_STEPS_SWEEP = sorted(set(_raw.tolist()))
REPS_SWEEP    = 5

BETA_COLORS = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
LAM_LS      = ["-", "--", ":"]
LAM_COLORS  = ["#2c3e50", "#8e44ad", "#c0392b"]
OUT         = os.path.dirname(__file__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def circuit_graph_metrics(columns):
    """
    Build fundamental-circuit intersection graph and return:
      largest_frac  — size of largest component / n_vertices
      degrees       — numpy array, degree of each vertex
      avg_circ_size — mean |fc[j]| = k_rows + 1 (always K+1=5 for k=4)
    Returns (None, None, None) if fewer than 2 circuits.

    Adjacency via clique-cover: for each basis row, all circuits containing
    that row form a clique.  Avoids large-integer bitset representation —
    works for any r_final (including r >> 63).
    """
    from collections import defaultdict

    n = len(columns)
    if n < 2:
        return None, None, None

    # Average circuit size = k_rows + 1 (non-basis element counts too)
    avg_circ_size = float(np.mean([len(col) for col in columns])) + 1.0

    # Build row → circuit-index array map
    row_to_circs = defaultdict(list)
    for j, col in enumerate(columns):
        for row in col:
            row_to_circs[row].append(j)

    # Vectorised clique-edge construction: for each row, use numpy meshgrid
    # to enumerate all (i<j) pairs, collect as COO data.
    src_list, dst_list = [], []
    for circ_list in row_to_circs.values():
        if len(circ_list) < 2:
            continue
        arr = np.array(circ_list, dtype=np.int32)
        ii, jj = np.meshgrid(arr, arr, indexing='ij')
        mask = ii < jj
        src_list.append(ii[mask].ravel())
        dst_list.append(jj[mask].ravel())

    if not src_list:
        return 0.0, np.zeros(n, dtype=int), avg_circ_size

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    # Symmetrise and build bool CSR (duplicates → sum → cast bool)
    all_r = np.concatenate([src, dst])
    all_c = np.concatenate([dst, src])
    data  = np.ones(len(all_r), dtype=np.int8)
    adj_csr = csr_matrix((data, (all_r, all_c)), shape=(n, n))
    adj_csr = (adj_csr > 0)          # deduplicate: bool
    degrees = np.asarray(adj_csr.sum(axis=1)).ravel()

    n_comp, labels = connected_components(adj_csr, directed=False)
    comp_sizes   = np.bincount(labels)
    largest_frac = float(comp_sizes.max()) / n

    return largest_frac, degrees, avg_circ_size


def run_one(n_steps, beta, lam, seed):
    np.random.seed(seed)
    return MatroidEngine(n_steps=n_steps, k_params=K, C=lam, gamma=GAMMA,
                         beta=beta, start_r=START_R).run()


# ── Main runs ──────────────────────────────────────────────────────────────────
print(f"Main runs: {len(LAMBDA_VALUES)} λ × {len(BETA_VALUES)} β × {REPS} reps …\n")
print(f"{'λ':<6} {'β':<6} {'n_circ':>8} {'lf':>8} {'avg_deg':>9} "
      f"{'avg_sz':>8} {'ρ':>7}")
print("-" * 55)

main = {}  # (lam, beta) → aggregated metrics

for lam in LAMBDA_VALUES:
    for beta in BETA_VALUES:
        lf_list, deg_list, sz_list = [], [], []
        n_circ_list, rho_list = [], []
        for seed in range(REPS):
            data = run_one(N_STEPS, beta, lam, seed=seed * 37)
            cols  = data["columns"]
            r_fin = data["r"]
            n_fin = data["n"]
            lf, degs, avg_sz = circuit_graph_metrics(cols)
            if lf is not None:
                lf_list.append(lf)
                deg_list.append(degs)
                sz_list.append(avg_sz)
                n_circ_list.append(len(cols))
                rho_list.append(n_fin / r_fin)

        main[(lam, beta)] = {
            "lf_mean":      float(np.mean(lf_list)),
            "lf_std":       float(np.std(lf_list)),
            "degrees":      np.concatenate(deg_list) if deg_list else np.array([]),
            "avg_sz":       float(np.mean(sz_list)),
            "n_circ_mean":  float(np.mean(n_circ_list)),
            "avg_deg_mean": float(np.mean([d.mean() for d in deg_list])),
            "rho_mean":     float(np.mean(rho_list)),
        }
        m = main[(lam, beta)]
        print(f"{lam:<6} {beta:<6} {m['n_circ_mean']:>8.0f} "
              f"{m['lf_mean']:>8.3f} {m['avg_deg_mean']:>9.1f} "
              f"{m['avg_sz']:>8.2f} {m['rho_mean']:>7.1f}")
    print()

# ── Density sweep: λ=LAM_FIXED, vary n_steps ──────────────────────────────────
print(f"Density sweep: λ={LAM_FIXED}, {len(N_STEPS_SWEEP)} n_steps × "
      f"{len(BETA_VALUES)} β × {REPS_SWEEP} reps …")

sweep = {}  # (n_steps, beta) → {rho, lf_mean, lf_std}

for n_s in N_STEPS_SWEEP:
    for beta in BETA_VALUES:
        lf_list, rho_list = [], []
        for seed in range(REPS_SWEEP):
            data = run_one(n_s, beta, LAM_FIXED, seed=seed * 37)
            r_fin, n_fin = data["r"], data["n"]
            lf, _, _ = circuit_graph_metrics(data["columns"])
            if lf is not None:
                lf_list.append(lf)
                rho_list.append(n_fin / r_fin if r_fin > 0 else 0)
        sweep[(n_s, beta)] = {
            "rho":     float(np.mean(rho_list))    if rho_list else 0,
            "lf_mean": float(np.mean(lf_list))     if lf_list  else 0,
            "lf_std":  float(np.std(lf_list))      if lf_list  else 0,
        }
    print(f"  n_steps={n_s:5d}  done", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Largest component fraction vs β
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlot 1 …")
fig, ax = plt.subplots(figsize=(8, 5.5))

for lam, ls, color in zip(LAMBDA_VALUES, LAM_LS, LAM_COLORS):
    means = [main[(lam, b)]["lf_mean"] for b in BETA_VALUES]
    stds  = [main[(lam, b)]["lf_std"]  for b in BETA_VALUES]
    rho   = np.mean([main[(lam, b)]["rho_mean"] for b in BETA_VALUES])
    ax.errorbar(BETA_VALUES, means, yerr=stds,
                fmt="o" + ls, color=color, lw=2, ms=7, capsize=4,
                label=f"λ={lam}  (ρ≈{rho:.1f})")

ax.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5)
ax.set_xlabel("Attachment bias  β", fontsize=12)
ax.set_ylabel("Largest component fraction", fontsize=12)
ax.set_title(
    "Circuit intersection graph — largest component vs β\n"
    f"γ={GAMMA}, k={K}, n_steps={N_STEPS}, start_r={START_R}, {REPS} reps",
    fontsize=11, fontweight="bold"
)
ax.set_ylim(-0.03, 1.05)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "circuit_percolation_component.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  → circuit_percolation_component.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Degree distribution log-log  (fixed λ=LAM_FIXED, all β)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 2 …")
fig, ax = plt.subplots(figsize=(8, 5.5))

for beta, color in zip(BETA_VALUES, BETA_COLORS):
    degs = main[(LAM_FIXED, beta)]["degrees"]
    if len(degs) == 0:
        continue
    counts = np.bincount(degs.astype(int))
    k_vals = np.where(counts > 0)[0]
    c_vals = counts[k_vals]
    ax.loglog(k_vals + 1, c_vals, "o", color=color, ms=3.5, alpha=0.65,
              label=f"β={beta}  (μ={degs.mean():.0f})")

    # Power-law fit on tail (top 80% of k range)
    if len(k_vals) >= 5:
        lo = int(np.percentile(k_vals, 20))
        mask = k_vals >= max(lo, 1)
        if mask.sum() >= 4:
            try:
                lk = np.log(k_vals[mask].astype(float) + 1)
                lc = np.log(c_vals[mask].astype(float))
                slope, intercept = np.polyfit(lk, lc, 1)
                k_fit = np.array([k_vals[mask][0], k_vals[mask][-1]], dtype=float) + 1
                ax.loglog(k_fit, np.exp(intercept) * k_fit ** slope,
                          "-", color=color, lw=1.5, alpha=0.85,
                          label=f"  α={-slope:.2f}")
            except Exception:
                pass

ax.set_xlabel("Degree  k", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(
    "Degree distribution of circuit intersection graph\n"
    f"γ={GAMMA}, λ={LAM_FIXED}, k_attach={K}, n_steps={N_STEPS}, start_r={START_R}",
    fontsize=11, fontweight="bold"
)
ax.legend(fontsize=8.5, ncol=2)
ax.grid(True, which="both", alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "circuit_percolation_degree.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  → circuit_percolation_degree.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Largest component fraction vs density  (λ=LAM_FIXED, all β)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 3 …")
fig, ax = plt.subplots(figsize=(8, 5.5))

for beta, color in zip(BETA_VALUES, BETA_COLORS):
    pts = sorted(
        [(sweep[(n_s, beta)]["rho"],
          sweep[(n_s, beta)]["lf_mean"],
          sweep[(n_s, beta)]["lf_std"])
         for n_s in N_STEPS_SWEEP],
        key=lambda x: x[0]
    )
    rhos  = [p[0] for p in pts]
    means = [p[1] for p in pts]
    stds  = [p[2] for p in pts]
    ax.errorbar(rhos, means, yerr=stds,
                fmt="o-", color=color, lw=2, ms=5, capsize=3,
                label=f"β={beta}")

ax.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5, label="0.5")
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
fig.savefig(os.path.join(OUT, "circuit_percolation_density.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  → circuit_percolation_density.png")
print("\nAll done.")
