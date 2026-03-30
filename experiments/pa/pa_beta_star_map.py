"""
Critical curve β*(α, λ) for the parallel-pair transition.

γ=0, k_t = max(2, floor(α·r_t)), n_steps=1000, start_r=10, 20 replicates.
α ∈ {0.2, 0.3, 0.4, 0.5}
λ ∈ {0.05, 0.1, 0.2, 0.5}
β ∈ {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5}

β*(α,λ) estimated by linear interpolation at f_2 = 0.5.

Figures (saved separately):
  beta_star_heatmap.png — 2-D heatmap of β*(α, λ)
  beta_star_curves.png  — β* vs α (per λ) and β* vs λ (per α),
                           plus log-log check for α·λ collapse
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool, cpu_count
import os

GAMMA   = 0.0
ALPHAS  = [0.2, 0.3, 0.4, 0.5]
LAMBDAS = [0.05, 0.1, 0.2, 0.5]
BETAS   = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
N_STEPS = 1000
START_R = 10
REPS    = 20
OUT     = os.path.dirname(__file__)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_dynamic_k(n_steps, alpha, C, gamma, beta, start_r, seed):
    rng       = np.random.default_rng(seed)
    curr_r    = start_r
    row_usage = np.ones(start_r, dtype=np.float64)
    supports  = []
    for t in range(1, n_steps + 1):
        p_row = min(1.0, C * (t ** (-gamma)))
        if rng.random() < p_row:
            curr_r += 1
            row_usage = np.append(row_usage, 1.0)
        else:
            k = min(max(2, int(np.floor(alpha * curr_r))), curr_r)
            w = row_usage ** beta;  w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            supports.append(sorted(int(x) for x in sel))
            for idx in sel:
                row_usage[int(idx)] += 1
    return supports


def frac2_and_mean(columns):
    sc = Counter(frozenset(c) for c in columns)
    counts = Counter()
    for sup, m in sc.items():
        if m == 1:
            counts[len(sup) + 1] += 1
        else:
            counts[2] += m * (m - 1) // 2
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    f2 = counts[2] / total
    ms = sum(s * counts[s] for s in counts) / total
    return f2, ms


def worker(args):
    """One (alpha, lam, beta) job: return (alpha, lam, beta, f2_mean, ms_mean)."""
    alpha, lam, beta = args
    f2_l, ms_l = [], []
    for rep in range(REPS):
        seed = int(alpha * 1000) * 100000 + int(lam * 1000) * 100 + int(beta * 100) + rep
        cols = run_dynamic_k(N_STEPS, alpha, lam, GAMMA, beta, START_R, seed)
        f2, ms = frac2_and_mean(cols)
        f2_l.append(f2);  ms_l.append(ms)
    return alpha, lam, beta, float(np.mean(f2_l)), float(np.mean(ms_l))


# ── Run in parallel ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    jobs = [(a, l, b) for a in ALPHAS for l in LAMBDAS for b in BETAS]
    n_workers = min(cpu_count(), len(jobs))
    print(f"Running {len(jobs)} jobs on {n_workers} workers …")

    with Pool(n_workers) as pool:
        raw = pool.map(worker, jobs)

    # Organise into nested dict
    data = {}   # data[alpha][lam][beta] = (f2, ms)
    for alpha, lam, beta, f2, ms in raw:
        data.setdefault(alpha, {}).setdefault(lam, {})[beta] = (f2, ms)

    # ── Print table ───────────────────────────────────────────────────────────

    print(f"\n{'α':>5}  {'λ':>5}  {'β*':>7}  "
          + "  ".join(f"f2@{b:.1f}" for b in BETAS))
    print("─" * (20 + 10 * len(BETAS)))

    beta_star = {}   # beta_star[alpha][lam]
    for alpha in ALPHAS:
        beta_star[alpha] = {}
        for lam in LAMBDAS:
            f2s = [data[alpha][lam][b][0] for b in BETAS]
            # Linear interpolation for f2 = 0.5
            bs = None
            for i in range(len(BETAS) - 1):
                if f2s[i] <= 0.5 <= f2s[i + 1]:
                    t = (0.5 - f2s[i]) / (f2s[i + 1] - f2s[i])
                    bs = BETAS[i] + t * (BETAS[i + 1] - BETAS[i])
                    break
            beta_star[alpha][lam] = bs
            bs_str = f"{bs:.3f}" if bs is not None else "  N/A"
            f2_str = "  ".join(f"{v:.3f}" for v in f2s)
            print(f"{alpha:>5.1f}  {lam:>5.2f}  {bs_str:>7}  {f2_str}")
        print()

    # ── Heatmap of β*(α, λ) ──────────────────────────────────────────────────

    bs_grid = np.array([[beta_star[a][l] if beta_star[a][l] is not None else np.nan
                         for l in LAMBDAS] for a in ALPHAS])

    fig1, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(bs_grid, aspect="auto", origin="lower",
                   cmap="plasma",
                   vmin=np.nanmin(bs_grid), vmax=np.nanmax(bs_grid))
    ax.set_xticks(range(len(LAMBDAS))); ax.set_xticklabels(LAMBDAS)
    ax.set_yticks(range(len(ALPHAS)));  ax.set_yticklabels(ALPHAS)
    ax.set_xlabel(r"$\lambda$", fontsize=13)
    ax.set_ylabel(r"$\alpha$",  fontsize=13)
    ax.set_title(r"Critical curve $\beta^*(\alpha,\lambda)$", fontsize=13)
    for i, a in enumerate(ALPHAS):
        for j, l in enumerate(LAMBDAS):
            v = bs_grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=11, color="white" if v < 1.2 else "black")
    plt.colorbar(im, ax=ax, label=r"$\beta^*$")
    plt.tight_layout()
    p1 = os.path.join(OUT, "beta_star_heatmap.png")
    fig1.savefig(p1, dpi=150);  print(f"  → {p1}")

    # ── Line plots + collapse check ───────────────────────────────────────────

    COLORS_L = ["#1a6faf", "#2980b9", "#27ae60", "#e74c3c"]
    COLORS_A = ["#8e44ad", "#d35400", "#16a085", "#2c3e50"]

    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: β* vs α for each λ
    ax1 = axes[0]
    for lam, col in zip(LAMBDAS, COLORS_L):
        ys = [beta_star[a][lam] for a in ALPHAS]
        valid = [(a, y) for a, y in zip(ALPHAS, ys) if y is not None]
        if valid:
            xs, ys_v = zip(*valid)
            ax1.plot(xs, ys_v, "o-", color=col, lw=2, markersize=7,
                     label=fr"$\lambda={lam}$")
    ax1.set_xlabel(r"$\alpha$", fontsize=13)
    ax1.set_ylabel(r"$\beta^*$", fontsize=13)
    ax1.set_title(r"$\beta^*$ vs $\alpha$  (per $\lambda$)", fontsize=12)
    ax1.legend(fontsize=10);  ax1.grid(True, alpha=0.3)

    # Panel 2: β* vs λ for each α
    ax2 = axes[1]
    for alpha, col in zip(ALPHAS, COLORS_A):
        ys = [beta_star[alpha][l] for l in LAMBDAS]
        valid = [(l, y) for l, y in zip(LAMBDAS, ys) if y is not None]
        if valid:
            xs, ys_v = zip(*valid)
            ax2.plot(xs, ys_v, "s-", color=col, lw=2, markersize=7,
                     label=fr"$\alpha={alpha}$")
    ax2.set_xlabel(r"$\lambda$", fontsize=13)
    ax2.set_ylabel(r"$\beta^*$", fontsize=13)
    ax2.set_title(r"$\beta^*$ vs $\lambda$  (per $\alpha$)", fontsize=12)
    ax2.legend(fontsize=10);  ax2.grid(True, alpha=0.3)

    # Panel 3: β* vs α·λ (collapse check)
    ax3 = axes[2]
    for alpha, col in zip(ALPHAS, COLORS_A):
        for lam in LAMBDAS:
            bs = beta_star[alpha][lam]
            if bs is not None:
                ax3.scatter([alpha * lam], [bs], color=col, s=60,
                            marker="o", edgecolors="k", linewidths=0.5)
    # Overlay λ-encoded marker shapes for cross-check
    for lam, mk in zip(LAMBDAS, ["o", "s", "^", "D"]):
        for alpha in ALPHAS:
            bs = beta_star[alpha][lam]
            if bs is not None:
                ax3.scatter([alpha * lam], [bs], color="none", s=80,
                            marker=mk, edgecolors=COLORS_L[LAMBDAS.index(lam)],
                            linewidths=1.5)
    ax3.set_xlabel(r"$\alpha \cdot \lambda$", fontsize=13)
    ax3.set_ylabel(r"$\beta^*$", fontsize=13)
    ax3.set_title(r"Collapse test: $\beta^*$ vs $\alpha\cdot\lambda$", fontsize=12)
    ax3.grid(True, alpha=0.3)
    # Dummy handles for legend
    from matplotlib.lines import Line2D
    handles = ([Line2D([0],[0], marker="o", color=c, lw=0, ms=7,
                       label=fr"$\alpha={a}$") for a, c in zip(ALPHAS, COLORS_A)] +
               [Line2D([0],[0], marker=mk, color="none", ms=8, lw=0,
                       markeredgecolor=c, markeredgewidth=1.5,
                       label=fr"$\lambda={l}$")
                for l, mk, c in zip(LAMBDAS, ["o","s","^","D"], COLORS_L)])
    ax3.legend(handles=handles, fontsize=8, ncol=2)

    fig2.suptitle(
        r"Critical $\beta^*(\alpha,\lambda)$: $\gamma=0$, "
        r"$k_t=\max(2,\lfloor\alpha r_t\rfloor)$, $n_{\rm steps}=1000$",
        fontsize=13)
    plt.tight_layout()
    p2 = os.path.join(OUT, "beta_star_curves.png")
    fig2.savefig(p2, dpi=150);  print(f"  → {p2}")
