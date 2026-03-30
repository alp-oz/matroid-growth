"""
Triangle minor threshold with dynamic k_t = floor(α·r_t).

γ=0, start_r=10, 25 replicates.
n_steps: 20 log-spaced values in [50, 3000].
α ∈ {0.2, 0.3, 0.5},  λ ∈ {0.05, 0.1},  β ∈ {0.5, 1.0, 1.5}.

For each (α, λ, β, n_steps): compute P(triangle minor) and mean ρ_t = n_t/r_t.

Plot: 3×2 panel grid (rows=α, cols=λ), each panel shows P(triangle) vs ρ_t
      for the three β values.

Saves: triangle_threshold_dynk.png
"""
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os

from analysis.probe_minors import convert_to_bitsets

GAMMA   = 0.0
ALPHAS  = [0.2, 0.3, 0.5]
LAMBDAS = [0.05, 0.1]
BETAS   = [0.5, 1.0, 1.5]
N_STEPS_ARR = np.unique(
    np.round(np.logspace(np.log10(50), np.log10(3000), 20)).astype(int)
).tolist()
START_R = 10
REPS    = 25
COLORS_B = ["#2980b9", "#27ae60", "#e74c3c"]
OUT      = os.path.dirname(__file__)


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
            k = max(1, int(np.floor(alpha * curr_r)))
            k = min(k, curr_r)
            w = row_usage ** beta;  w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            supports.append(sorted(int(x) for x in sel))
            for idx in sel:
                row_usage[int(idx)] += 1
    n = curr_r + len(supports)
    return curr_r, n, supports


def has_triangle(columns):
    if not columns:
        return False
    bits = list(set(convert_to_bitsets(columns)) - {0})
    if len(bits) < 3:
        return False
    lookup = set(bits)
    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            c = bits[i] ^ bits[j]
            if c != 0 and c in lookup:
                return True
    return False


# ── Worker: one (α, λ, β) curve ───────────────────────────────────────────────

def worker(args):
    alpha, lam, beta = args
    rho_vals  = []
    prob_vals = []
    for n_steps in N_STEPS_ARR:
        hits    = 0
        rho_sum = 0.0
        for rep in range(REPS):
            seed = (int(alpha * 1000) * 100000
                    + int(lam * 10000) * 10
                    + int(beta * 100) + rep + n_steps)
            r, n, cols = run_dynamic_k(n_steps, alpha, lam, GAMMA,
                                        beta, START_R, seed)
            rho_sum += n / r
            if has_triangle(cols):
                hits += 1
        rho_vals.append(rho_sum / REPS)
        prob_vals.append(hits / REPS)
    return alpha, lam, beta, rho_vals, prob_vals


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    jobs = [(a, l, b) for a in ALPHAS for l in LAMBDAS for b in BETAS]
    n_workers = min(cpu_count(), len(jobs))
    print(f"Running {len(jobs)} curves × {len(N_STEPS_ARR)} n_steps "
          f"× {REPS} reps = "
          f"{len(jobs)*len(N_STEPS_ARR)*REPS} total runs on {n_workers} workers …")

    with Pool(n_workers) as pool:
        results = pool.map(worker, jobs)

    # Organise: res[alpha][lam][beta] = (rho_vals, prob_vals)
    res = {}
    for alpha, lam, beta, rho_vals, prob_vals in results:
        res.setdefault(alpha, {}).setdefault(lam, {})[beta] = (rho_vals, prob_vals)

    # ── Print threshold estimates (ρ where P≈0.5) ────────────────────────────

    print(f"\n{'α':>5}  {'λ':>5}  {'β':>5}  {'ρ*(P=0.5)':>10}")
    print("─" * 35)
    for alpha in ALPHAS:
        for lam in LAMBDAS:
            for beta in BETAS:
                rho_v, prob_v = res[alpha][lam][beta]
                rho_star = None
                for i in range(len(prob_v) - 1):
                    if prob_v[i] <= 0.5 <= prob_v[i + 1]:
                        t = (0.5 - prob_v[i]) / (prob_v[i + 1] - prob_v[i])
                        rho_star = rho_v[i] + t * (rho_v[i + 1] - rho_v[i])
                        break
                rho_str = f"{rho_star:.2f}" if rho_star else "  <min"
                print(f"{alpha:>5.1f}  {lam:>5.2f}  {beta:>5.1f}  {rho_str:>10}")
        print()

    # ── Plot ──────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(len(ALPHAS), len(LAMBDAS),
                             figsize=(6 * len(LAMBDAS), 4.5 * len(ALPHAS)),
                             sharey=True)

    for row, alpha in enumerate(ALPHAS):
        for col, lam in enumerate(LAMBDAS):
            ax = axes[row][col]
            for beta, color in zip(BETAS, COLORS_B):
                rho_v, prob_v = res[alpha][lam][beta]
                ax.plot(rho_v, prob_v, "o-", color=color, lw=2,
                        markersize=5, label=fr"$\beta={beta}$")

            ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
            ax.set_xlabel(r"$\rho_t = n_t / r_t$", fontsize=11)
            ax.set_ylabel(r"$P(\triangle\text{ minor})$", fontsize=11)
            ax.set_title(fr"$\alpha={alpha}$, $\lambda={lam}$", fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        fr"Triangle minor threshold: $\gamma=0$, "
        fr"$k_t=\lfloor\alpha r_t\rfloor$, $n_{{\rm steps}}\in[50,3000]$",
        fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT, "triangle_threshold_dynk.png")
    fig.savefig(path, dpi=150)
    print(f"\n  → {path}")
