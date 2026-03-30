"""
Rank and nullity growth trajectories for the PA binary matroid.

For each configuration we track at every step t:
  r_t   = current basis size  (= start_r + #discoveries up to t)
  n_t   = total elements      (= start_r + t, deterministic)
  ν_t   = n_t - r_t           = #attachment elements up to t
  ν_t/t = nullity rate

Key question: does ν_t/t converge to a positive constant (linear
growth) or to 0, and does the transition threshold t* depend on
(γ, C) or on (β, k)?

Note: r_t and ν_t are determined entirely by the discovery/attachment
Bernoulli decisions (prob p_row = min(1, C·t^{-γ}) at step t).
Neither β nor k enters — they only affect which rows get selected
during attachment, not whether a step is discovery or attachment.

Configs:
  1. γ=0, C=0.05,  β=0.8, k=4
  2. γ=0, C=0.2,   β=0.8, k=4
  3. γ=0, C=0.5,   β=0.8, k=4
  4. γ=1, C=0.5,   β=0.8, k=4
  5. γ=1, C=2.0,   β=0.8, k=4

Output: rank_growth.png  (saved next to this script)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

CONFIGS = [
    {"label": "γ=0, C=0.05", "gamma": 0.0, "C": 0.05, "color": "#2980b9", "ls": "-"},
    {"label": "γ=0, C=0.2",  "gamma": 0.0, "C": 0.2,  "color": "#27ae60", "ls": "-"},
    {"label": "γ=0, C=0.5",  "gamma": 0.0, "C": 0.5,  "color": "#e74c3c", "ls": "-"},
    {"label": "γ=1, C=0.5",  "gamma": 1.0, "C": 0.5,  "color": "#e67e22", "ls": "--"},
    {"label": "γ=1, C=2.0",  "gamma": 1.0, "C": 2.0,  "color": "#8e44ad", "ls": "--"},
]

N_STEPS  = 5000
START_R  = 10
REPS     = 30   # replicates per config for mean ± std band

# ── Theoretical expectations ────────────────────────────────────────────────────
# γ=0: E[r_t] = start_r + C·t  →  E[ν_t/t] → 1 - C
# γ=1: E[r_t] ≈ start_r + C·∑_{s=1}^{t} min(1, C/s)/C
#             = start_r + min(C,1)·floor(min(C,t)) + C·(ln t - ln C) for t≫C
#      either way E[ν_t/t] → 1  (discoveries are o(t))

def theory_nu_rate(t_arr, C, gamma):
    """Theoretical E[ν_t / t] = 1 - E[r_t - start_r] / t."""
    if gamma == 0:
        return np.full_like(t_arr, 1.0 - C, dtype=float)
    elif gamma == 1:
        # E[discoveries by t] = ∑_{s=1}^t min(1, C/s)
        # For C < 1: all terms < 1 → ≈ C·ln(t+1)
        # For C ≥ 1: first floor(C) terms =1, rest ≈ C·(ln t - ln C)
        cumdisc = np.zeros_like(t_arr, dtype=float)
        for i, t in enumerate(t_arr):
            s = np.arange(1, t + 1, dtype=float)
            cumdisc[i] = np.sum(np.minimum(1.0, C / s))
        return 1.0 - cumdisc / t_arr
    return None


def simulate(n_steps, C, gamma, start_r, seed):
    """
    Returns r_traj[0..n_steps] where r_traj[0] = start_r and
    each step either increments r (discovery, prob min(1, C·t^{-γ}))
    or not (attachment).
    """
    rng = np.random.default_rng(seed)
    r = start_r
    r_traj = np.empty(n_steps + 1, dtype=np.int32)
    r_traj[0] = r
    for t in range(1, n_steps + 1):
        p = min(1.0, C * (t ** (-gamma)))
        if rng.random() < p:
            r += 1
        r_traj[t] = r
    return r_traj


# ── Run all configs ─────────────────────────────────────────────────────────────
print(f"Running {len(CONFIGS)} configs × {REPS} replicates × {N_STEPS} steps …")

t = np.arange(0, N_STEPS + 1, dtype=float)          # t = 0, 1, …, N_STEPS
n_t = START_R + t                                     # deterministic

results = []
for cfg in CONFIGS:
    trajs = np.stack([
        simulate(N_STEPS, cfg["C"], cfg["gamma"], START_R, seed=s)
        for s in range(REPS)
    ])  # shape (REPS, N_STEPS+1)

    r_mean = trajs.mean(axis=0)
    r_std  = trajs.std(axis=0)

    nu_mean = n_t - r_mean
    nu_std  = r_std                  # same std (n_t is deterministic)

    # ν_t / t  (skip t=0 to avoid /0)
    t_pos = t[1:]; nu_pos = n_t[1:] - trajs[:, 1:]   # (REPS, N_STEPS)
    rate_mean = nu_pos.mean(axis=0) / t_pos
    rate_std  = nu_pos.std(axis=0)  / t_pos

    results.append({
        "r_mean": r_mean, "r_std": r_std,
        "nu_mean": nu_mean, "nu_std": nu_std,
        "rate_mean": rate_mean, "rate_std": rate_std,
    })
    print(f"  {cfg['label']:20s} | final r̄={r_mean[-1]:.1f}±{r_std[-1]:.1f}"
          f"  ν̄/t={rate_mean[-1]:.4f}")

# ── Figure ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
ax_r, ax_nu, ax_rate = axes

ALPHA_BAND = 0.15

for cfg, res in zip(CONFIGS, results):
    kw = dict(color=cfg["color"], ls=cfg["ls"], lw=2)
    bkw = dict(color=cfg["color"], alpha=ALPHA_BAND)

    # Panel 1: r_t
    ax_r.plot(t, res["r_mean"], label=cfg["label"], **kw)
    ax_r.fill_between(t,
                      res["r_mean"] - res["r_std"],
                      res["r_mean"] + res["r_std"], **bkw)

    # Panel 2: ν_t
    ax_nu.plot(t, res["nu_mean"], label=cfg["label"], **kw)
    ax_nu.fill_between(t,
                       res["nu_mean"] - res["nu_std"],
                       res["nu_mean"] + res["nu_std"], **bkw)

    # Panel 3: ν_t / t
    ax_rate.plot(t[1:], res["rate_mean"], label=cfg["label"], **kw)
    ax_rate.fill_between(t[1:],
                         res["rate_mean"] - res["rate_std"],
                         res["rate_mean"] + res["rate_std"], **bkw)

# Asymptotic reference lines for ν_t / t
t_ref = np.linspace(1, N_STEPS, 500)
for cfg in CONFIGS:
    if cfg["gamma"] == 0:
        lim = 1.0 - cfg["C"]
        ax_rate.axhline(lim, color=cfg["color"], ls=":", lw=1, alpha=0.6)
# γ=1 both → 1 asymptotically
ax_rate.axhline(1.0, color="black", ls=":", lw=1, alpha=0.4, label="limit = 1  (γ=1)")

# Reference: n_t line in panel 1
ax_r.plot(t, n_t, "k--", lw=1, alpha=0.4, label="n_t = r₀ + t")

# ── Annotations ────────────────────────────────────────────────────────────────
ax_r.set_xlabel("Step  t", fontsize=12)
ax_r.set_ylabel("r_t  (basis size)", fontsize=12)
ax_r.set_title("Basis size  r_t\n(solid) vs total elements n_t (dashed)",
               fontsize=11, fontweight="bold")
ax_r.legend(fontsize=8.5); ax_r.grid(alpha=0.3)

ax_nu.set_xlabel("Step  t", fontsize=12)
ax_nu.set_ylabel("ν_t = n_t − r_t", fontsize=12)
ax_nu.set_title("Nullity  ν_t\n#attachment elements added by step t",
                fontsize=11, fontweight="bold")
ax_nu.legend(fontsize=8.5); ax_nu.grid(alpha=0.3)

ax_rate.set_xlabel("Step  t", fontsize=12)
ax_rate.set_ylabel("ν_t / t", fontsize=12)
ax_rate.set_title("Nullity rate  ν_t / t\ndotted lines = asymptotic limits",
                  fontsize=11, fontweight="bold")
ax_rate.set_ylim(-0.02, 1.05)
ax_rate.legend(fontsize=8.5); ax_rate.grid(alpha=0.3)

fig.suptitle(
    f"Rank and nullity growth  (start_r={START_R}, k=4, β=0.8,  {REPS} replicates)\n"
    "γ=0: ν_t/t → 1−C  |  γ=1: ν_t/t → 1  (discoveries sub-linear in t)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "rank_growth.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out}")
