"""
Support frequency distribution (Zipf) with dynamic k_t = floor(α·r_t).

γ=0, λ=0.05, n_steps=3000, start_r=10, 5 replicates (averaged for fit).
α ∈ {0.2, 0.3, 0.5},  β ∈ {0.5, 1.0, 1.5, 2.0}.

For each (α, β): log-log plot of sorted support frequency vs coordinate rank,
power-law fit log(usage) = log(A) - s·log(rank), report exponent s and R².

Layout: 3×4 grid (rows=α, cols=β).
Saves: support_powerlaw.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

LAMBDA  = 0.05
GAMMA   = 0.0
ALPHAS  = [0.2, 0.3, 0.5]
BETAS   = [0.5, 1.0, 1.5, 2.0]
N_STEPS = 3000
START_R = 10
REPS    = 5
OUT     = os.path.dirname(__file__)

COLORS_B = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]


# ── Simulation ────────────────────────────────────────────────────────────────

def run_dynamic_k(n_steps, alpha, C, gamma, beta, start_r, seed):
    rng       = np.random.default_rng(seed)
    curr_r    = start_r
    row_usage = np.ones(start_r, dtype=np.float64)
    for t in range(1, n_steps + 1):
        p_row = min(1.0, C * (t ** (-gamma)))
        if rng.random() < p_row:
            curr_r += 1
            row_usage = np.append(row_usage, 1.0)
        else:
            k = max(1, int(np.floor(alpha * curr_r)))
            k = min(k, curr_r)
            w = row_usage ** beta;  w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            for idx in sel:
                row_usage[int(idx)] += 1
    return row_usage


def fit_powerlaw(usage):
    """
    Fit log(usage) = log(A) - s·log(rank) on sorted usage.
    Returns (s, R²).  Excludes the bottom 5% of ranks (noise).
    """
    sorted_u = np.sort(usage)[::-1]
    n        = len(sorted_u)
    cutoff   = max(3, int(n * 0.95))
    ranks    = np.arange(1, cutoff + 1, dtype=float)
    vals     = sorted_u[:cutoff]
    mask     = vals > 0
    if mask.sum() < 3:
        return np.nan, np.nan
    lr = linregress(np.log(ranks[mask]), np.log(vals[mask]))
    return -lr.slope, lr.rvalue ** 2


# ── Collect data ──────────────────────────────────────────────────────────────

# results[alpha][beta] = list of (usage array, s, r2) per rep
results = {}
print(f"\n{'α':>5}  {'β':>5}  {'s_mean':>8}  {'s_std':>7}  {'R2_mean':>8}")
print("─" * 45)

for alpha in ALPHAS:
    results[alpha] = {}
    for beta in BETAS:
        reps_usage, reps_s, reps_r2 = [], [], []
        for rep in range(REPS):
            usage = run_dynamic_k(N_STEPS, alpha, LAMBDA, GAMMA,
                                  beta, START_R,
                                  seed=int(alpha*1000)*1000 + int(beta*100)*10 + rep)
            s, r2 = fit_powerlaw(usage)
            reps_usage.append(usage)
            reps_s.append(s);  reps_r2.append(r2)
        results[alpha][beta] = dict(
            usages=reps_usage,
            s_mean=np.nanmean(reps_s), s_std=np.nanstd(reps_s),
            r2_mean=np.nanmean(reps_r2),
        )
        print(f"{alpha:>5.1f}  {beta:>5.1f}  "
              f"{np.nanmean(reps_s):>8.3f}  {np.nanstd(reps_s):>7.3f}  "
              f"{np.nanmean(reps_r2):>8.3f}")
    print()


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(ALPHAS), len(BETAS),
                         figsize=(4.5 * len(BETAS), 3.8 * len(ALPHAS)),
                         sharex=False, sharey=False)

for row, alpha in enumerate(ALPHAS):
    for col, (beta, color) in enumerate(zip(BETAS, COLORS_B)):
        ax  = axes[row][col]
        d   = results[alpha][beta]

        # Plot each replicate lightly, then the first rep more prominently
        for usage in d["usages"]:
            sorted_u = np.sort(usage)[::-1]
            ranks    = np.arange(1, len(sorted_u) + 1)
            ax.loglog(ranks, sorted_u, ".", color=color,
                      alpha=0.15, markersize=2)

        # Power-law fit line on first replicate
        usage0   = d["usages"][0]
        sorted_u = np.sort(usage0)[::-1]
        ranks    = np.arange(1, len(sorted_u) + 1)
        ax.loglog(ranks, sorted_u, ".", color=color, alpha=0.7, markersize=3)

        s, r2 = fit_powerlaw(usage0)
        if not np.isnan(s):
            mask = sorted_u > 0
            fit  = np.exp(np.log(sorted_u[mask][0]) +
                          (-s) * (np.log(ranks[mask]) - np.log(ranks[mask][0])))
            ax.loglog(ranks[mask], fit, "-", color="k", lw=1.5, alpha=0.8)

        ax.set_title(
            fr"$\alpha={alpha}$, $\beta={beta}$"
            "\n"
            fr"$s={d['s_mean']:.3f}\pm{d['s_std']:.3f}$, "
            fr"$R^2={d['r2_mean']:.3f}$",
            fontsize=9)
        ax.set_xlabel("Coordinate rank", fontsize=8)
        ax.set_ylabel("Support freq.", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", alpha=0.2)

fig.suptitle(
    fr"Support frequency: $\gamma=0$, $\lambda=0.05$, "
    fr"$k_t=\lfloor\alpha r_t\rfloor$, $n_{{\rm steps}}=3000$",
    fontsize=13, y=1.01)
plt.tight_layout()
path = os.path.join(OUT, "support_powerlaw.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"  → {path}")

# ── Summary heatmap of s ──────────────────────────────────────────────────────

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

s_grid  = np.array([[results[a][b]["s_mean"]  for b in BETAS] for a in ALPHAS])
r2_grid = np.array([[results[a][b]["r2_mean"] for b in BETAS] for a in ALPHAS])

for ax, grid, title, cmap in [
    (ax1, s_grid,  r"Power-law exponent $s$",   "plasma"),
    (ax2, r2_grid, r"Fit quality $R^2$",          "viridis"),
]:
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(range(len(BETAS)));  ax.set_xticklabels(BETAS)
    ax.set_yticks(range(len(ALPHAS))); ax.set_yticklabels(ALPHAS)
    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_ylabel(r"$\alpha$", fontsize=12)
    ax.set_title(title, fontsize=12)
    for i in range(len(ALPHAS)):
        for j in range(len(BETAS)):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if grid[i,j] < grid.mean() else "black")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
path2 = os.path.join(OUT, "support_powerlaw_summary.png")
fig2.savefig(path2, dpi=150)
print(f"  → {path2}")
