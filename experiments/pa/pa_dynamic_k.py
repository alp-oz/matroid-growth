"""
Circuit structure with dynamic attachment size k_t = max(2, floor(α · r_t)).

γ=0, λ=0.05, n_steps=1000, start_r=10, 5 replicates.
β ∈ {0.5, 1.0, 1.5},  α ∈ {0.2, 0.3, 0.5}.

At each attachment step the number of basis rows selected is
    k_t = max(2, floor(α · r_t))
so k grows proportionally to the current rank.  With γ=0, λ=0.05,
r_t ≈ 10 + 0.05·t, giving k_final ≈ {12, 18, 30} for α = {0.2, 0.3, 0.5}.

Key question: does k_t ∝ r_t prevent the parallel-pair collapse seen for
fixed k, even at high β?

Figure layout (single file):
  Top 3×3 grid  — circuit size distribution for each (α row, β column).
  Bottom row    — mean circuit size vs β (left) and total circuits vs β (right),
                  one curve per α.

Saves: dynamic_k_circuits.png
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

LAMBDA  = 0.05
GAMMA   = 0.0
BETAS   = [0.5, 1.0, 1.5]
ALPHAS  = [0.2, 0.3, 0.5]
N_STEPS = 1000
START_R = 10
REPS    = 5
COLORS_B = ["#2980b9", "#27ae60", "#e74c3c"]   # one per β
COLORS_A = ["#8e44ad", "#d35400", "#16a085"]   # one per α
OUT      = os.path.dirname(__file__)


# ── Simulation with dynamic k ─────────────────────────────────────────────────

def run_dynamic_k(n_steps, alpha, C, gamma, beta, start_r, seed):
    """
    Replicate MatroidEngine logic with k_t = max(2, floor(α · r_t)).
    Returns (r_final, attachment_supports).
    """
    rng        = np.random.default_rng(seed)
    curr_r     = start_r
    row_usage  = np.ones(start_r, dtype=np.float64)
    supports   = []

    for t in range(1, n_steps + 1):
        p_row = min(1.0, C * (t ** (-gamma)))
        if rng.random() < p_row:
            curr_r += 1
            row_usage = np.append(row_usage, 1.0)
        else:
            k = max(2, int(np.floor(alpha * curr_r)))
            k = min(k, curr_r)
            w = row_usage ** beta
            w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            supports.append(sorted(int(x) for x in sel))
            for idx in sel:
                row_usage[int(idx)] += 1

    return curr_r, supports


# ── Circuit counting ──────────────────────────────────────────────────────────

def minimal_circuit_counts(columns):
    """
    Size-2: parallel pairs (identical support).
    Size k+1: fundamental circuit for columns with unique support.
    """
    support_count = Counter(frozenset(col) for col in columns)
    counts = Counter()
    for support, m in support_count.items():
        if m == 1:
            counts[len(support) + 1] += 1
        else:
            counts[2] += m * (m - 1) // 2
    return counts


# ── Collect data ──────────────────────────────────────────────────────────────

# results[alpha][beta] = dict with aggregated stats
results = {}
print(f"\n{'α':>5}  {'β':>5}  {'r_mean':>8}  {'k_final':>8}  "
      f"{'n_circ':>8}  {'mean_sz':>8}  {'frac_2':>7}")
print("─" * 60)

for alpha in ALPHAS:
    results[alpha] = {}
    for beta in BETAS:
        all_counts = Counter()
        r_l, nc_l, ms_l, f2_l = [], [], [], []

        for rep in range(REPS):
            r, cols = run_dynamic_k(N_STEPS, alpha, LAMBDA, GAMMA,
                                    beta, START_R, seed=rep * 100 + int(beta * 10))
            counts = minimal_circuit_counts(cols)
            total  = sum(counts.values())
            mean_s = sum(s * counts[s] for s in counts) / total if total else 0
            f2     = counts[2] / total if total else 0
            k_fin  = max(2, int(np.floor(alpha * r)))

            all_counts.update(counts)
            r_l.append(r);  nc_l.append(total)
            ms_l.append(mean_s);  f2_l.append(f2)

        k_final_mean = max(2, int(np.floor(alpha * np.mean(r_l))))
        results[alpha][beta] = dict(
            agg=all_counts,
            r=np.mean(r_l), k_final=k_final_mean,
            n_circ=np.mean(nc_l), n_circ_std=np.std(nc_l),
            mean_size=np.mean(ms_l), mean_size_std=np.std(ms_l),
            frac2=np.mean(f2_l), frac2_std=np.std(f2_l),
        )
        print(f"{alpha:>5.1f}  {beta:>5.1f}  {np.mean(r_l):>8.1f}  "
              f"{k_final_mean:>8d}  {np.mean(nc_l):>8.0f}  "
              f"{np.mean(ms_l):>8.3f}  {np.mean(f2_l):>7.3f}")
    print()


# ── Build global size axis ────────────────────────────────────────────────────

all_sizes = sorted({s for alpha in ALPHAS for beta in BETAS
                    for s in results[alpha][beta]["agg"]})
# Cap display: keep sizes up to 95th percentile of counts
total_all = sum(results[alpha][beta]["agg"][s]
                for alpha in ALPHAS for beta in BETAS for s in all_sizes)
cumsum, plot_max_size = 0, all_sizes[-1]
for s in sorted(all_sizes):
    cumsum += sum(results[alpha][beta]["agg"].get(s, 0)
                  for alpha in ALPHAS for beta in BETAS)
    if cumsum / total_all >= 0.995:
        plot_max_size = s
        break
plot_sizes = [s for s in all_sizes if s <= plot_max_size]


# ── Plot ──────────────────────────────────────────────────────────────────────

n_row = len(ALPHAS) + 1    # 3 distribution rows + 1 summary row
fig = plt.figure(figsize=(16, 4 * n_row))
gs  = fig.add_gridspec(n_row, 3, hspace=0.45, wspace=0.35)

# ── Top 3×3: circuit size distributions ──────────────────────────────────────

n_beta  = len(BETAS)
bar_w   = 0.22
offsets = np.linspace(-(n_beta - 1) / 2, (n_beta - 1) / 2, n_beta) * bar_w

for row_idx, alpha in enumerate(ALPHAS):
    k_finals = [results[alpha][b]["k_final"] for b in BETAS]
    for col_idx, (beta, color) in enumerate(zip(BETAS, COLORS_B)):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        agg   = results[alpha][beta]["agg"]
        total = sum(agg.values())
        fracs = [agg.get(s, 0) / total if total else 0 for s in plot_sizes]
        ax.bar(plot_sizes, fracs, color=color, alpha=0.85, width=0.7)
        ax.set_title(fr"$\alpha={alpha}$, $\beta={beta}$  "
                     fr"($k_{{final}}\approx{results[alpha][beta]['k_final']}$)",
                     fontsize=10)
        ax.set_xlabel("Circuit size $|C|$", fontsize=9)
        ax.set_ylabel("Fraction", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ms = results[alpha][beta]["mean_size"]
        ax.axvline(ms, color="k", ls="--", lw=1.2, alpha=0.7,
                   label=fr"$\bar{{|C|}}={ms:.1f}$")
        ax.legend(fontsize=8, loc="upper right")


# ── Bottom row: summary lines ─────────────────────────────────────────────────

ax_ms = fig.add_subplot(gs[3, 0])
ax_nc = fig.add_subplot(gs[3, 1])
ax_f2 = fig.add_subplot(gs[3, 2])

for alpha, color in zip(ALPHAS, COLORS_A):
    ms_means = [results[alpha][b]["mean_size"]     for b in BETAS]
    ms_stds  = [results[alpha][b]["mean_size_std"] for b in BETAS]
    nc_means = [results[alpha][b]["n_circ"]        for b in BETAS]
    nc_stds  = [results[alpha][b]["n_circ_std"]    for b in BETAS]
    f2_means = [results[alpha][b]["frac2"]         for b in BETAS]
    f2_stds  = [results[alpha][b]["frac2_std"]     for b in BETAS]
    kw = dict(color=color, lw=2, markersize=7, capsize=4,
              label=fr"$\alpha={alpha}$")
    ax_ms.errorbar(BETAS, ms_means, yerr=ms_stds, fmt="o-", **kw)
    ax_nc.errorbar(BETAS, nc_means, yerr=nc_stds, fmt="o-", **kw)
    ax_f2.errorbar(BETAS, f2_means, yerr=f2_stds, fmt="o-", **kw)

for ax, ylabel, title in [
    (ax_ms, "Mean circuit size",       "Mean $|C|$ vs $\\beta$"),
    (ax_nc, "Total circuits",          "Circuit count vs $\\beta$"),
    (ax_f2, "Fraction size-2 circuits", "Parallel-pair fraction vs $\\beta$"),
]:
    ax.set_xlabel(r"$\beta$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
ax_f2.set_ylim(-0.05, 1.05)

fig.suptitle(
    fr"Dynamic $k_t = \max(2,\lfloor\alpha r_t\rfloor)$:  "
    fr"$\gamma=0$, $\lambda=0.05$, $n_{{\rm steps}}=1000$, $\mathrm{{start\_r}}=10$",
    fontsize=13, y=1.01)

path = os.path.join(OUT, "dynamic_k_circuits.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"  → {path}")
