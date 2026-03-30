"""
XOR circuit chain on C(M_t): spectral gap δ and mixing time τ_mix(1/4) vs β.

Simulation: γ=0, λ=0.05, α=0.3, k_t=⌊0.3·r_t⌋, n_steps=1000, start_r=10,
            5 replicates per β.
β ∈ {0.5, 0.8, 1.0, 1.1, 1.2, 1.5}.

State space: fundamental circuits F_j = {e_j} ∪ S_j  (one per attached column j).
Transition: from F_j1, draw j2 uniformly from {0,...,n_att-1}.
  • S_j1 ∩ S_j2 ≠ ∅  → F_j1 △ F_j2 = pairwise circuit, no fundamental in it → lazy.
  • S_j1 ∩ S_j2 = ∅  → F_j1 △ F_j2 = F_j1 ∪ F_j2, two fundamentals → go to
                        each with probability 1/2 (as in the XOR decomposition).
Stationary: uniform over F_0,...,F_{n_att-1}  (P is symmetric).

The "non-overlap graph" G_⊥ on {F_j} (edge iff disjoint supports) can be
disconnected: the chain restricted to each connected component is ergodic there.
We report, for each replicate:
  • n_comp       — number of connected components of G_⊥
  • frac_largest — fraction of fundamentals in the largest component
  • δ            — 1 − λ₂(P_sub), spectral gap within the largest component
  • τ_mix(1/4)   — ⌈log(4·|comp|) / δ⌉  (standard spectral bound)

Transition matrix P (n_att × n_att, symmetric):
  P[j1,j2] = 1/(2·n_att)    if j1≠j2 and S_j1 ∩ S_j2 = ∅
  P[j1,j1] = 1 − deg(j1)/(2·n_att)   deg = #{j2≠j1 : disjoint supports}

Saves: xor_chain_gap.png, xor_chain_results.json

Re-run behaviour: if xor_chain_results.json already exists the simulation is
skipped and the figure is regenerated from the cached data.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ALPHA    = 0.3
LAMBDA   = 0.05
GAMMA    = 0.0
N_STEPS  = 1000
START_R  = 10
REPS     = 5
BETAS    = [0.5, 0.8, 1.0, 1.1, 1.2, 1.5]
BETA_STAR = 1.10
OUT      = os.path.dirname(__file__)
CACHE    = os.path.join(OUT, "xor_chain_results.json")


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_pa(n_steps, alpha, C_lam, gamma, beta, start_r, seed):
    rng       = np.random.default_rng(seed)
    curr_r    = start_r
    row_usage = np.ones(start_r, dtype=np.float64)
    supports  = []
    for t in range(1, n_steps + 1):
        p_row = min(1.0, C_lam * t ** (-gamma))
        if rng.random() < p_row:
            curr_r += 1
            row_usage = np.append(row_usage, 1.0)
        else:
            k   = max(1, min(int(np.floor(alpha * curr_r)), curr_r))
            w   = row_usage ** beta;  w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            supports.append(frozenset(int(x) for x in sel))
            for idx in sel:
                row_usage[int(idx)] += 1
    return supports, curr_r


# ── Chain analysis ─────────────────────────────────────────────────────────────

def xor_chain_stats(supports, r_final):
    """
    Returns (delta, tau_mix, n_comp, frac_largest, n_att).
    Spectral gap is computed within the largest connected component of G_⊥.
    """
    n_att = len(supports)
    if n_att < 3:
        return (float("nan"),) * 4 + (n_att,)

    # Binary matrix B[j,i] = 1 iff row i ∈ S_j
    B = np.zeros((n_att, r_final), dtype=np.float32)
    for j, supp in enumerate(supports):
        for i in supp:
            B[j, i] = 1.0

    # Gram matrix: G[j1,j2] = |S_j1 ∩ S_j2|
    G = (B @ B.T).round().astype(np.int32)

    # Non-overlap adjacency (off-diagonal zeros in G)
    nonoverlap = (G == 0)
    np.fill_diagonal(nonoverlap, False)

    # Connected components of the non-overlap graph
    n_comp, labels = connected_components(csr_matrix(nonoverlap), directed=False)
    comp_sizes     = np.bincount(labels)
    largest_comp   = int(np.argmax(comp_sizes))
    frac_largest   = float(comp_sizes[largest_comp]) / n_att

    # Restrict to largest component
    mask   = labels == largest_comp
    idx    = np.where(mask)[0]
    n_sub  = len(idx)
    if n_sub < 3:
        return (float("nan"), float("nan"),
                int(n_comp), frac_largest, n_att)

    no_sub  = nonoverlap[np.ix_(idx, idx)]
    deg_sub = no_sub.sum(axis=1).astype(np.float64)

    # P_sub uses the global n_att for normalisation (same as the full chain)
    P_sub = np.zeros((n_sub, n_sub), dtype=np.float64)
    P_sub[no_sub] = 1.0 / (2.0 * n_att)
    np.fill_diagonal(P_sub, 1.0 - deg_sub / (2.0 * n_att))

    eigvals = np.linalg.eigvalsh(P_sub)   # ascending; P_sub symmetric
    lambda2 = float(eigvals[-2])
    delta   = float(1.0 - lambda2)
    if delta < 1e-14:
        return (float("nan"), float("nan"),
                int(n_comp), frac_largest, n_att)

    tau_mix = float(np.ceil(np.log(4.0 * n_sub) / delta))
    return delta, tau_mix, int(n_comp), frac_largest, n_att


# ── Load or compute ────────────────────────────────────────────────────────────

keys = ["delta", "tau", "n_comp", "frac_largest", "n_att"]

if os.path.exists(CACHE):
    print(f"Loading cached results from {CACHE}")
    with open(CACHE) as f:
        cache_data = json.load(f)
    # Reconstruct results dict from summary (single-element lists = means only)
    results = {beta: {k: [] for k in keys} for beta in BETAS}
    for row in cache_data["summary"]:
        b = row["beta"]
        if b not in results:
            continue
        results[b]["delta"].append(row["delta_mean"])
        results[b]["tau"].append(row["tau_mean"])
        results[b]["n_comp"].append(row["n_comp_mean"])
        results[b]["frac_largest"].append(row["frac_largest_mean"])
        results[b]["n_att"].append(row["n_att_mean"])
else:
    print("Running PA simulations + XOR chain spectral gap …")
    results = {beta: {k: [] for k in keys} for beta in BETAS}

    for b_idx, beta in enumerate(BETAS):
        for rep in range(REPS):
            seed = b_idx * 100000 + 77000 + rep
            sup, rf = run_pa(N_STEPS, ALPHA, LAMBDA, GAMMA, beta, START_R, seed)
            d, tau, nc, fl, na = xor_chain_stats(sup, rf)
            results[beta]["delta"].append(d)
            results[beta]["tau"].append(tau)
            results[beta]["n_comp"].append(nc)
            results[beta]["frac_largest"].append(fl)
            results[beta]["n_att"].append(na)
        d_m  = np.nanmean(results[beta]["delta"])
        nc_m = np.mean(results[beta]["n_comp"])
        fl_m = np.mean(results[beta]["frac_largest"])
        print(f"  β={beta:.1f}  n_att≈{np.mean(results[beta]['n_att']):.0f}"
              f"  n_comp≈{nc_m:.1f}  frac_largest={fl_m:.2f}"
              f"  δ={d_m:.5f}")

    # Save full per-replicate results
    cache_out = {
        "params": {"alpha": ALPHA, "lambda": LAMBDA, "gamma": GAMMA,
                   "n_steps": N_STEPS, "start_r": START_R, "reps": REPS,
                   "betas": BETAS, "beta_star": BETA_STAR},
        "per_rep": {
            str(beta): {k: [float(v) if np.isfinite(v) else None
                            for v in results[beta][k]]
                        for k in keys}
            for beta in BETAS
        },
        "summary": [
            {"beta": beta,
             "n_att_mean":      float(np.mean(results[beta]["n_att"])),
             "n_comp_mean":     float(np.mean(results[beta]["n_comp"])),
             "frac_largest_mean": float(np.mean(results[beta]["frac_largest"])),
             "delta_mean":      float(np.nanmean(results[beta]["delta"])),
             "tau_mean":        float(np.nanmean(results[beta]["tau"]))}
            for beta in BETAS
        ]
    }
    with open(CACHE, "w") as f:
        json.dump(cache_out, f, indent=2)
    print(f"\nCached → {CACHE}")


# ── Summary table ──────────────────────────────────────────────────────────────

print(f"\n{'β':>5}  {'n_att':>6}  {'n_comp':>7}  {'frac_larg':>10}  "
      f"{'δ':>9}  {'τ_mix':>10}")
print("─" * 62)
for beta in BETAS:
    r = results[beta]
    print(f"{beta:>5.1f}  {np.mean(r['n_att']):>6.0f}  "
          f"{np.mean(r['n_comp']):>7.1f}  "
          f"{np.mean(r['frac_largest']):>10.3f}  "
          f"{np.nanmean(r['delta']):>9.5f}  "
          f"{np.nanmean(r['tau']):>10.0f}")


# ── Figure ─────────────────────────────────────────────────────────────────────

beta_x = np.array(BETAS, dtype=float)

def arr(key, fn=np.nanmean):
    return np.array([fn(results[b][key]) for b in BETAS])

delta_mean  = arr("delta");    delta_std  = arr("delta", np.nanstd)
tau_mean    = arr("tau");      tau_std    = arr("tau",   np.nanstd)
ncomp_mean  = arr("n_comp",  np.mean);   ncomp_std  = arr("n_comp",  np.std)
frac_mean   = arr("frac_largest", np.mean); frac_std = arr("frac_largest", np.std)

COLORS = {"delta": "#2c7bb6", "tau": "#d7191c",
          "ncomp": "#7b2d8b", "frac": "#1a9641"}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.ravel()

def add_panel(ax, y_mean, y_std, y_raw_key, color, ylabel, title, log=False):
    ax.errorbar(beta_x, y_mean, yerr=y_std,
                fmt="o-", color=color, lw=2.2, ms=8,
                capsize=6, capthick=1.5)
    for b_idx, beta in enumerate(BETAS):
        pts = np.array(results[beta][y_raw_key])
        pts = pts[np.isfinite(pts)]
        ax.scatter(np.full(len(pts), beta), pts,
                   color=color, s=20, alpha=0.45, zorder=3)
    ax.axvline(BETA_STAR, color="dimgray", ls="--", lw=1.8,
               label=fr"$\beta^* \approx {BETA_STAR}$")
    ax.set_xlabel(r"Attachment bias $\beta$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(BETAS)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    if log:
        ax.set_yscale("log")

add_panel(axes[0], delta_mean, delta_std, "delta", COLORS["delta"],
          r"Spectral gap  $\delta = 1 - \lambda_2$",
          "Spectral gap (within largest component)")

add_panel(axes[1], tau_mean, tau_std, "tau", COLORS["tau"],
          r"$\tau_{\rm mix}(1/4)$  upper bound",
          "Mixing time (within largest component)", log=True)

add_panel(axes[2], ncomp_mean, ncomp_std, "n_comp", COLORS["ncomp"],
          "Number of components",
          r"Components of $G_\perp$ (non-overlap graph)")

add_panel(axes[3], frac_mean, frac_std, "frac_largest", COLORS["frac"],
          "Fraction in largest component",
          "Largest component size (fraction of fundamentals)")

fig.suptitle(
    r"XOR circuit chain on fundamental circuits of $M_t$:  "
    r"$\gamma=0$,  $\lambda=0.05$,  $\alpha=0.3$,  "
    r"$k_t=\lfloor 0.3\,r_t\rfloor$,  $n_{\rm steps}=1000$,  5 reps",
    fontsize=11, y=1.01)

plt.tight_layout()
path = os.path.join(OUT, "xor_chain_gap.png")
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"\n  → {path}")
