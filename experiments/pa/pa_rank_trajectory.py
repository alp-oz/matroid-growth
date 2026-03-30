"""
Rank trajectory r_t vs t with fixed k=4.

γ=0, k=4, n_steps=3000, start_r=10, 10 replicates.
β ∈ {0.5, 1.0, 1.5, 2.0},  λ ∈ {0.05, 0.2, 0.5}.

Two panels per λ (arranged as 3×2 grid):
  Left  column — r_t vs t  (raw trajectories + mean, theory λt shown)
  Right column — r_t / (λt) vs t  (deviation from linear prediction)

Note: for γ=0 the discovery decision at step t is Bernoulli(λ) independent
of β, so E[r_t] = start_r + λt for ALL β. β only governs which rows are
selected during attachment, not whether a step is discovery or attachment.
The ratio r_t/(λt) should therefore converge to 1 for all β.

Saves: rank_trajectory.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

GAMMA   = 0.0
K       = 4
BETAS   = [0.5, 1.0, 1.5, 2.0]
LAMBDAS = [0.05, 0.2, 0.5]
N_STEPS = 3000
START_R = 10
REPS    = 10
COLORS  = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
OUT     = os.path.dirname(__file__)

t_arr = np.arange(0, N_STEPS + 1)


# ── Lightweight trajectory simulation ────────────────────────────────────────
# r_t depends only on the discovery events (Bernoulli(λ) each step),
# which are independent of β.  We simulate with β-specific RNG seeds
# so each β trace is an independent realisation, not a re-use of the
# same random stream.

def simulate_rank_trajectory(n_steps, C, gamma, start_r, seed):
    """Return r_t array of length n_steps+1 (index = step number)."""
    rng    = np.random.default_rng(seed)
    r      = start_r
    r_traj = np.empty(n_steps + 1, dtype=np.int32)
    r_traj[0] = r
    for t in range(1, n_steps + 1):
        p = min(1.0, C * (t ** (-gamma)))
        if rng.random() < p:
            r += 1
        r_traj[t] = r
    return r_traj


# ── Collect trajectories ──────────────────────────────────────────────────────

# trajs[lam][beta] = array of shape (REPS, N_STEPS+1)
trajs = {}
for lam in LAMBDAS:
    trajs[lam] = {}
    for beta in BETAS:
        runs = []
        for rep in range(REPS):
            seed = int(lam * 10000) * 10000 + int(beta * 100) * 100 + rep
            runs.append(simulate_rank_trajectory(
                N_STEPS, lam, GAMMA, START_R, seed))
        trajs[lam][beta] = np.array(runs, dtype=np.float64)


# ── Print final-step summary ──────────────────────────────────────────────────

print(f"\n{'λ':>5}  {'β':>5}  {'r_mean':>8}  {'r_std':>7}  "
      f"{'r/(λt)_mean':>12}  {'theory λt':>10}")
print("─" * 55)
for lam in LAMBDAS:
    for beta in BETAS:
        r_final = trajs[lam][beta][:, -1]
        theory  = START_R + lam * N_STEPS
        ratio   = r_final / (lam * N_STEPS)
        print(f"{lam:>5.2f}  {beta:>5.1f}  {r_final.mean():>8.1f}  "
              f"{r_final.std():>7.2f}  {ratio.mean():>12.4f}  {theory:>10.1f}")
    print()


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(LAMBDAS), 2,
                         figsize=(14, 4.5 * len(LAMBDAS)))

# Avoid division by zero at t=0
t_pos  = t_arr[1:]          # t = 1 … N_STEPS

for row, lam in enumerate(LAMBDAS):
    ax_r   = axes[row][0]
    ax_rat = axes[row][1]

    theory_line = START_R + lam * t_arr

    # Theory line
    ax_r.plot(t_arr, theory_line, "k--", lw=1.5, alpha=0.6,
              label=r"$\mathrm{start\_r}+\lambda t$")

    for beta, color in zip(BETAS, COLORS):
        mat = trajs[lam][beta]           # shape (REPS, N_STEPS+1)
        mean_r = mat.mean(axis=0)
        std_r  = mat.std(axis=0)

        # Individual traces (light)
        for rep in range(REPS):
            ax_r.plot(t_arr, mat[rep], color=color, alpha=0.12, lw=0.6)

        # Mean ± 1 std
        ax_r.plot(t_arr, mean_r, color=color, lw=2,
                  label=fr"$\beta={beta}$")
        ax_r.fill_between(t_arr, mean_r - std_r, mean_r + std_r,
                          color=color, alpha=0.15)

        # Ratio r_t / (λt), skip t=0
        ratio_mat  = mat[:, 1:] / (lam * t_pos)
        mean_ratio = ratio_mat.mean(axis=0)
        std_ratio  = ratio_mat.std(axis=0)

        ax_rat.plot(t_pos, mean_ratio, color=color, lw=2,
                    label=fr"$\beta={beta}$")
        ax_rat.fill_between(t_pos, mean_ratio - std_ratio,
                            mean_ratio + std_ratio,
                            color=color, alpha=0.15)

    ax_r.set_xlabel("Step $t$", fontsize=11)
    ax_r.set_ylabel(r"$r_t$", fontsize=12)
    ax_r.set_title(fr"$\lambda={lam}$: rank trajectory", fontsize=12)
    ax_r.legend(fontsize=9, loc="upper left")
    ax_r.grid(True, alpha=0.3)

    ax_rat.axhline(1.0, color="k", ls="--", lw=1.5, alpha=0.6,
                   label=r"$r_t/(\lambda t)=1$")
    ax_rat.set_xlabel("Step $t$", fontsize=11)
    ax_rat.set_ylabel(r"$r_t \,/\, (\lambda t)$", fontsize=12)
    ax_rat.set_title(fr"$\lambda={lam}$: normalised rank", fontsize=12)
    ax_rat.legend(fontsize=9, loc="upper right")
    ax_rat.grid(True, alpha=0.3)

fig.suptitle(
    r"Rank trajectory: $\gamma=0$, $k=4$, $n_{\rm steps}=3000$, "
    r"$\mathrm{start\_r}=10$",
    fontsize=13)
plt.tight_layout()
path = os.path.join(OUT, "rank_trajectory.png")
fig.savefig(path, dpi=150)
print(f"  → {path}")
