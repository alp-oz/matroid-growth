"""
Phase transition in the PA binary matroid — improved version.

The original phase_transition.py always returns P(F7)=P(W3)=1 because the
matroid is hugely overcrowded (n/r ≈ 20).  The F7 check also fires on ANY
GF(2) dependent triple, which is near-certain in a dense matroid.

Fixes:
  1. Operate near the CRITICAL DENSITY  n/r ∈ [0.5, 5] so minors are
     neither absent nor ubiquitous.
  2. Use CONTINUOUS metrics that vary smoothly:
       (a) rank deficit  ρ = (n – rank) / n  (nullity fraction)
       (b) Gini coefficient of row usage  (inequality / hub formation)
       (c) Sampled girth  (minimum circuit size)
  3. Show the JOINT EFFECT of density and β in a 2-panel figure:
       Left  — density sweep, fixed β=0.8  (three metrics)
       Right — β sweep, fixed density n/r≈2  (same three metrics)

Run takes ~1 minute.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from core.engine import MatroidEngine
from analysis.probe_minors import convert_to_bitsets, calculate_binary_rank
from analysis.stats import get_zipf_distribution

SEED       = 42
ITERS      = 12      # replicates per parameter setting
START_R    = 50      # fixed: no row growth (C ≈ 0)
K_PARAMS   = 4
C          = 1e-6    # effectively zero row growth

np.random.seed(SEED)


# ── Helper metrics ─────────────────────────────────────────────────────────────

def gini(arr):
    """Gini coefficient of a non-negative array."""
    arr = np.sort(arr[arr > 0].astype(float))
    n   = len(arr)
    if n == 0: return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n)


def rank_deficit(columns, r):
    bits = convert_to_bitsets(columns)
    rk   = calculate_binary_rank(bits)
    n    = len(columns)
    return (n - rk) / n if n > 0 else 0.0


def sampled_girth(columns, r_final, samples=500):
    """
    Estimate the girth (minimum circuit size) by sampling subsets and
    checking GF(2) rank.  Returns the smallest g such that P(rank < g) > 0.
    """
    n = len(columns)
    if n < 2:
        return None
    bits = convert_to_bitsets(columns)
    lookup = set(bits)

    # Check for girth-2 (duplicate columns)
    if len(lookup) < n:
        return 2

    # Check for girth-3 (triangles: a⊕b=c)
    unique = list(lookup - {0})
    u = len(unique)
    for _ in range(min(samples, u * (u - 1) // 2)):
        i, j = np.random.choice(u, size=2, replace=False)
        if (unique[i] ^ unique[j]) in lookup:
            return 3

    # Check for girth-4 (4-element dependent sets)
    for _ in range(samples):
        idx  = np.random.choice(n, size=4, replace=False)
        samp = [columns[i] for i in idx]
        bits4 = convert_to_bitsets(samp)
        if calculate_binary_rank(bits4) < 4:
            return 4

    return None   # girth ≥ 5 or not found


def run_one(n_steps, beta):
    engine = MatroidEngine(n_steps=n_steps, k_params=K_PARAMS,
                           C=C, gamma=0, beta=beta, start_r=START_R)
    data = engine.run()
    cols = data['columns']
    usage = data['row_usage']
    g  = gini(usage)
    rd = rank_deficit(cols, data['R_final'])
    gi = sampled_girth(cols, data['R_final'])
    return g, rd, gi


def sweep(param_vals, fixed_name, fixed_val, varying_name, n_iters=ITERS):
    """Returns mean ± std arrays for (gini, rank_deficit, girth)."""
    ginis, rds, girths = [], [], []

    for val in param_vals:
        n_steps = val if varying_name == "n_steps" else fixed_val
        beta    = val if varying_name == "beta"    else fixed_val
        # n_steps for density sweep is set so n ≈ n_steps (C≈0)
        if varying_name == "n_steps":
            n_steps = val
        elif varying_name == "beta":
            n_steps = fixed_val  # fixed n ≈ START_R * density

        g_s, rd_s, gi_s = [], [], []
        for seed in range(n_iters):
            np.random.seed(seed * 17 + int(val * 100))
            g, rd, gi = run_one(n_steps, beta)
            g_s.append(g); rd_s.append(rd)
            if gi is not None: gi_s.append(gi)

        ginis.append((np.mean(g_s), np.std(g_s)))
        rds.append((np.mean(rd_s), np.std(rd_s)))
        girths.append((np.mean(gi_s) if gi_s else np.nan,
                       np.std(gi_s)  if gi_s else np.nan))

    return (np.array([x[0] for x in ginis]),  np.array([x[1] for x in ginis]),
            np.array([x[0] for x in rds]),    np.array([x[1] for x in rds]),
            np.array([x[0] for x in girths]), np.array([x[1] for x in girths]))


# ── Sweeps ─────────────────────────────────────────────────────────────────────
BETA_FIXED    = 0.8

# Density sweep: vary n_steps (≈ n_columns) with fixed β
# n/r runs from 0.5 to 5  (r = START_R = 50)
N_DENSITY = [int(x) for x in np.round(np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]) * START_R)]
DENSITY_LABELS = [s / START_R for s in N_DENSITY]

# β sweep: vary β with n ≈ 2 × START_R (density=2)
BETA_VALS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0]
N_FIXED   = 2 * START_R   # density ≈ 2

print("Running density sweep …")
gini_d, gini_d_e, rd_d, rd_d_e, gi_d, gi_d_e = sweep(
    N_DENSITY, "beta", BETA_FIXED, "n_steps")
print("Running β sweep …")
gini_b, gini_b_e, rd_b, rd_b_e, gi_b, gi_b_e = sweep(
    BETA_VALS, "n_steps", N_FIXED, "beta")


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
(ax_rd_d, ax_rd_b), (ax_g_d, ax_g_b) = axes

COLOR_MAIN = "#2980b9"
COLOR_GINI = "#e67e22"
COLOR_GIRTH = "#27ae60"


def _band(ax, x, mean, std, color, label, marker="o"):
    ax.plot(x, mean, marker + "-", color=color, lw=2, ms=6, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)


# ── Row 1: Rank deficit ────────────────────────────────────────────────────────
_band(ax_rd_d, DENSITY_LABELS, rd_d, rd_d_e,  COLOR_MAIN,  "rank deficit ρ")
ax_rd_d_twin = ax_rd_d.twinx()
_band(ax_rd_d_twin, DENSITY_LABELS, gini_d, gini_d_e, COLOR_GINI, "Gini (usage)", "s")
ax_rd_d.set_xlabel("Matroid density  n/r", fontsize=11)
ax_rd_d.set_ylabel("Rank deficit  ρ = (n − rank)/n", fontsize=10, color=COLOR_MAIN)
ax_rd_d_twin.set_ylabel("Gini coefficient of row usage", fontsize=10, color=COLOR_GINI)
ax_rd_d.set_title(f"Density sweep  (β = {BETA_FIXED})\n"
                  "Transition from sparse to dense", fontsize=11, fontweight="bold")
lines1, labs1 = ax_rd_d.get_legend_handles_labels()
lines2, labs2 = ax_rd_d_twin.get_legend_handles_labels()
ax_rd_d.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="center right")
ax_rd_d.grid(alpha=0.3)

_band(ax_rd_b, BETA_VALS, rd_b, rd_b_e,  COLOR_MAIN,  "rank deficit ρ")
ax_rd_b_twin = ax_rd_b.twinx()
_band(ax_rd_b_twin, BETA_VALS, gini_b, gini_b_e, COLOR_GINI, "Gini (usage)", "s")
ax_rd_b.set_xlabel("Attachment bias  β", fontsize=11)
ax_rd_b.set_ylabel("Rank deficit  ρ = (n − rank)/n", fontsize=10, color=COLOR_MAIN)
ax_rd_b_twin.set_ylabel("Gini coefficient of row usage", fontsize=10, color=COLOR_GINI)
ax_rd_b.set_title(f"β sweep  (n/r ≈ 2,  r = {START_R})\n"
                  "Role of preferential attachment", fontsize=11, fontweight="bold")
lines1, labs1 = ax_rd_b.get_legend_handles_labels()
lines2, labs2 = ax_rd_b_twin.get_legend_handles_labels()
ax_rd_b.legend(lines1 + lines2, labs1 + labs2, fontsize=9)
ax_rd_b.grid(alpha=0.3)

# ── Row 2: Girth ──────────────────────────────────────────────────────────────
finite_d = np.isfinite(gi_d)
if finite_d.any():
    ax_g_d.errorbar(np.array(DENSITY_LABELS)[finite_d], gi_d[finite_d],
                    yerr=gi_d_e[finite_d], fmt="^-", color=COLOR_GIRTH,
                    capsize=4, lw=2, ms=7, label="sampled girth")
ax_g_d.axhline(3, color="gray", ls="--", lw=1, alpha=0.7, label="girth = 3 (triangles)")
ax_g_d.axhline(2, color="red",  ls=":",  lw=1, alpha=0.7, label="girth = 2 (duplicates)")
ax_g_d.set_xlabel("Matroid density  n/r", fontsize=11)
ax_g_d.set_ylabel("Minimum circuit size  (girth)", fontsize=11)
ax_g_d.set_title(f"Girth vs. density  (β = {BETA_FIXED})",
                 fontsize=11, fontweight="bold")
ax_g_d.legend(fontsize=9); ax_g_d.grid(alpha=0.3)
ax_g_d.set_ylim(1.5, 5.5); ax_g_d.set_yticks([2, 3, 4, 5])

finite_b = np.isfinite(gi_b)
if finite_b.any():
    ax_g_b.errorbar(np.array(BETA_VALS)[finite_b], gi_b[finite_b],
                    yerr=gi_b_e[finite_b], fmt="^-", color=COLOR_GIRTH,
                    capsize=4, lw=2, ms=7, label="sampled girth")
ax_g_b.axhline(3, color="gray", ls="--", lw=1, alpha=0.7, label="girth = 3")
ax_g_b.axhline(2, color="red",  ls=":",  lw=1, alpha=0.7, label="girth = 2")
ax_g_b.set_xlabel("Attachment bias  β", fontsize=11)
ax_g_b.set_ylabel("Minimum circuit size  (girth)", fontsize=11)
ax_g_b.set_title(f"Girth vs. β  (n/r ≈ 2)",
                 fontsize=11, fontweight="bold")
ax_g_b.legend(fontsize=9); ax_g_b.grid(alpha=0.3)
ax_g_b.set_ylim(1.5, 5.5); ax_g_b.set_yticks([2, 3, 4, 5])

fig.suptitle("Phase structure of PA binary matroids\n"
             f"r = {START_R} rows,  k = {K_PARAMS} columns per step,  "
             f"{ITERS} replicates per point",
             fontsize=13, fontweight="bold")
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "phase_transition_v2.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close()
