"""
Collapse test: which variable best collapses fraction of size-2 circuits
across γ families?

β=1, k=4 (fixed), n_steps=3000, start_r=10, 10 replicates.

(γ, λ) combinations:
  γ=0  : λ ∈ {0.05, 0.1, 0.2, 0.5}
  γ=0.5: λ ∈ {0.1,  0.5, 1.0}
  γ=1  : λ ∈ {0.5,  1.0, 2.0}

For each run:
  r_t     — final rank
  n_t     — total elements
  ρ_t     — n_t / r_t
  frac_2  — fraction of minimal circuits that are size-2 (parallel pairs)
  H_t     — Shannon entropy of normalised row-usage: H = -Σ p_i log p_i,
             p_i = usage_i / Σ usage_j   (with β=1, this is the PA weight)

Three plots (columns):
  1. frac_2 vs ρ_t
  2. frac_2 vs ρ_t / r_t
  3. frac_2 vs H_t

Marker shape encodes γ; colour encodes λ.
Saves: collapse_test.png
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

from core.engine import MatroidEngine

BETA    = 1.0
K       = 4
N_STEPS = 3000
START_R = 10
REPS    = 10
OUT     = os.path.dirname(__file__)

CONFIGS = [
    (0.0,  0.05), (0.0,  0.1), (0.0,  0.2), (0.0,  0.5),
    (0.5,  0.1),  (0.5,  0.5), (0.5,  1.0),
    (1.0,  0.5),  (1.0,  1.0), (1.0,  2.0),
]

GAMMA_MARKER = {0.0: "o", 0.5: "s", 1.0: "^"}
GAMMA_LABEL  = {0.0: r"$\gamma=0$", 0.5: r"$\gamma=0.5$", 1.0: r"$\gamma=1$"}

ALL_LAMBDAS = sorted({lam for _, lam in CONFIGS})
_palette = [
    "#1a6faf", "#2980b9", "#5dade2", "#a9cce3",
    "#1e8449", "#27ae60",
    "#922b21", "#e74c3c", "#f1948a",
]
LAM_COLORS = {lam: c for lam, c in zip(ALL_LAMBDAS, _palette)}


def frac_size2(columns):
    """Fraction of minimal circuits that are size-2 parallel pairs."""
    support_count = Counter(frozenset(col) for col in columns)
    n2 = sum(m * (m - 1) // 2 for m in support_count.values() if m >= 2)
    n5 = sum(1 for m in support_count.values() if m == 1)
    total = n2 + n5
    return n2 / total if total else 0.0


def pa_entropy(row_usage):
    """Shannon entropy of the normalised row-usage distribution."""
    u = np.asarray(row_usage, dtype=float)
    p = u / u.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ── Run ──────────────────────────────────────────────────────────────────────

print(f"\n{'γ':>5}  {'λ':>5}  {'r':>7}  {'ρ':>7}  {'ρ/r':>9}  "
      f"{'H_t':>7}  {'frac_2':>7}")
print("─" * 60)

results = []

for gamma, lam in CONFIGS:
    r_l, rho_l, rhor_l, H_l, f2_l = [], [], [], [], []

    for _ in range(REPS):
        eng  = MatroidEngine(n_steps=N_STEPS, k_params=K, C=lam,
                             gamma=gamma, beta=BETA, start_r=START_R)
        data = eng.run()
        r    = data["r"]
        rho  = data["n"] / r
        H    = pa_entropy(data["row_usage"])
        f2   = frac_size2(data["columns"])

        r_l.append(r);  rho_l.append(rho)
        rhor_l.append(rho / r);  H_l.append(H);  f2_l.append(f2)

    row = dict(
        gamma=gamma, lam=lam,
        r=np.mean(r_l),         r_std=np.std(r_l),
        rho=np.mean(rho_l),     rho_std=np.std(rho_l),
        rhor=np.mean(rhor_l),   rhor_std=np.std(rhor_l),
        H=np.mean(H_l),         H_std=np.std(H_l),
        f2=np.mean(f2_l),       f2_std=np.std(f2_l),
    )
    results.append(row)
    print(f"{gamma:>5.1f}  {lam:>5.2f}  {row['r']:>7.1f}  {row['rho']:>7.2f}  "
          f"{row['rhor']:>9.4f}  {row['H']:>7.3f}  {row['f2']:>7.3f}")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

xlabels = [
    r"Attachment density $\rho_t = n_t/r_t$",
    r"Rescaled density $\rho_t / r_t$",
    r"PA entropy $H_t = -\sum_i p_i \log p_i$",
]
x_keys  = ["rho", "rhor", "H"]
x_stds  = ["rho_std", "rhor_std", "H_std"]

gamma_handles, lam_handles = {}, {}

for ax, xk, xs, xl in zip(axes, x_keys, x_stds, xlabels):
    for d in results:
        gamma, lam = d["gamma"], d["lam"]
        marker = GAMMA_MARKER[gamma]
        color  = LAM_COLORS[lam]
        kw = dict(marker=marker, color=color, markersize=9, lw=0,
                  markeredgecolor="k", markeredgewidth=0.6)
        ax.errorbar(d[xk], d["f2"],
                    xerr=d[xs], yerr=d["f2_std"],
                    capsize=3, elinewidth=1, ecolor=color, **kw)

        if gamma not in gamma_handles:
            gamma_handles[gamma] = plt.Line2D(
                [], [], marker=marker, color="k", lw=0, markersize=8,
                label=GAMMA_LABEL[gamma])
        if lam not in lam_handles:
            lam_handles[lam] = plt.Line2D(
                [], [], marker="o", color=color, lw=0, markersize=8,
                markeredgecolor="k", markeredgewidth=0.5,
                label=fr"$\lambda={lam}$")

    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel("Fraction size-2 circuits", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

axes[0].set_title(r"Plot 1: $f_2$ vs $\rho_t$", fontsize=12)
axes[1].set_title(r"Plot 2: $f_2$ vs $\rho_t / r_t$", fontsize=12)
axes[2].set_title(r"Plot 3: $f_2$ vs $H_t$", fontsize=12)

legend_handles = list(gamma_handles.values()) + list(lam_handles.values())
axes[2].legend(handles=legend_handles, fontsize=9, ncol=2,
               loc="upper right", framealpha=0.9)

fig.suptitle(
    r"Collapse test: $\beta=1$, $k=4$ fixed, $n_\mathrm{steps}=3000$",
    fontsize=13)
plt.tight_layout()
path = os.path.join(OUT, "collapse_test.png")
fig.savefig(path, dpi=150)
print(f"\n  → {path}")
