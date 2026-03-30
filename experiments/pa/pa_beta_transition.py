"""
Locate the critical β* for the parallel-pair transition.

γ=0, λ=0.05, α=0.3, k_t = max(2, floor(α·r_t)), n_steps=1000,
start_r=10, 20 replicates.
β ∈ {0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5}.

Plots:
  Left  — parallel-pair fraction f_2 vs β  (mean ± std over replicates)
  Right — mean circuit size vs β

A vertical dashed line is placed at the estimated β* where f_2 = 0.5.
Saves: beta_transition.png
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

LAMBDA  = 0.05
GAMMA   = 0.0
ALPHA   = 0.3
BETAS   = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
N_STEPS = 1000
START_R = 10
REPS    = 20
OUT     = os.path.dirname(__file__)


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
            w = row_usage ** beta
            w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            supports.append(sorted(int(x) for x in sel))
            for idx in sel:
                row_usage[int(idx)] += 1
    return curr_r, supports


def circuit_stats(columns):
    support_count = Counter(frozenset(col) for col in columns)
    counts = Counter()
    for support, m in support_count.items():
        if m == 1:
            counts[len(support) + 1] += 1
        else:
            counts[2] += m * (m - 1) // 2
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    frac2  = counts[2] / total
    mean_s = sum(s * counts[s] for s in counts) / total
    return frac2, mean_s


# ── Run ───────────────────────────────────────────────────────────────────────

f2_means, f2_stds   = [], []
ms_means, ms_stds   = [], []

print(f"\n{'β':>5}  {'f2_mean':>8}  {'f2_std':>7}  {'ms_mean':>9}  {'ms_std':>8}")
print("─" * 45)

for beta in BETAS:
    f2_l, ms_l = [], []
    for rep in range(REPS):
        _, cols = run_dynamic_k(N_STEPS, ALPHA, LAMBDA, GAMMA,
                                beta, START_R, seed=rep * 137 + int(beta * 100))
        f2, ms = circuit_stats(cols)
        f2_l.append(f2);  ms_l.append(ms)

    f2_means.append(np.mean(f2_l));  f2_stds.append(np.std(f2_l))
    ms_means.append(np.mean(ms_l));  ms_stds.append(np.std(ms_l))
    print(f"{beta:>5.1f}  {np.mean(f2_l):>8.3f}  {np.std(f2_l):>7.3f}  "
          f"{np.mean(ms_l):>9.3f}  {np.std(ms_l):>8.3f}")

# ── Estimate β* by linear interpolation where f2 = 0.5 ──────────────────────

beta_star = None
for i in range(len(BETAS) - 1):
    if f2_means[i] <= 0.5 <= f2_means[i + 1]:
        t = (0.5 - f2_means[i]) / (f2_means[i + 1] - f2_means[i])
        beta_star = BETAS[i] + t * (BETAS[i + 1] - BETAS[i])
        break
if beta_star is not None:
    print(f"\nEstimated β* ≈ {beta_star:.3f}  (f_2 = 0.5 crossing)")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
betas = np.array(BETAS)

ax1.errorbar(betas, f2_means, yerr=f2_stds, fmt="o-", color="#2980b9",
             lw=2, markersize=7, capsize=4, label=r"$f_2$ (mean ± std)")
ax1.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7, label=r"$f_2 = 0.5$")
if beta_star is not None:
    ax1.axvline(beta_star, color="#e74c3c", ls=":", lw=2,
                label=fr"$\beta^* \approx {beta_star:.3f}$")
ax1.set_xlabel(r"$\beta$", fontsize=13)
ax1.set_ylabel(r"Parallel-pair fraction $f_2$", fontsize=12)
ax1.set_title(r"Parallel-pair fraction vs $\beta$", fontsize=13)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.errorbar(betas, ms_means, yerr=ms_stds, fmt="o-", color="#27ae60",
             lw=2, markersize=7, capsize=4, label=r"mean $|C|$ (mean ± std)")
if beta_star is not None:
    ax2.axvline(beta_star, color="#e74c3c", ls=":", lw=2,
                label=fr"$\beta^* \approx {beta_star:.3f}$")
ax2.set_xlabel(r"$\beta$", fontsize=13)
ax2.set_ylabel(r"Mean circuit size $\langle|C|\rangle$", fontsize=12)
ax2.set_title(r"Mean circuit size vs $\beta$", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

fig.suptitle(
    fr"Critical $\beta^*$: $\gamma=0$, $\lambda=0.05$, $\alpha=0.3$, "
    fr"$k_t=\max(2,\lfloor\alpha r_t\rfloor)$, $n_{{\rm steps}}=1000$",
    fontsize=12)
plt.tight_layout()
path = os.path.join(OUT, "beta_transition.png")
fig.savefig(path, dpi=150)
print(f"  → {path}")
