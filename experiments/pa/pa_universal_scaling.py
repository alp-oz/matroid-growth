"""
Universal scaling: mean circuit size vs attachment density ρ_t = n_t / r_t.

Tests whether mean circuit size collapses onto a single curve as a function
of ρ_t regardless of (γ, λ) — i.e., ρ_t is the natural control parameter.

β=1, k_params=('poisson', 4), n_steps=3000, start_r=10, 10 replicates.

(γ, λ) combinations:
  γ=0  : λ ∈ {0.05, 0.1, 0.2, 0.5}   → ρ ≈ 1/λ ∈ {20, 10, 5, 2}
  γ=0.5: λ ∈ {0.1,  0.5, 1.0}
  γ=1  : λ ∈ {0.5,  1.0, 2.0}         → ρ large (sublinear rank growth)

Saves: universal_scaling.png
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

from core.engine import MatroidEngine

BETA    = 1.0
N_STEPS = 3000
START_R = 10
REPS    = 10
OUT     = os.path.dirname(__file__)

CONFIGS = [
    # (gamma, lambda)
    (0.0,  0.05), (0.0,  0.1), (0.0,  0.2), (0.0,  0.5),
    (0.5,  0.1),  (0.5,  0.5), (0.5,  1.0),
    (1.0,  0.5),  (1.0,  1.0), (1.0,  2.0),
]

# Visual encoding: marker per γ, colour per λ
GAMMA_MARKER = {0.0: "o", 0.5: "s", 1.0: "^"}
GAMMA_LABEL  = {0.0: r"$\gamma=0$", 0.5: r"$\gamma=0.5$", 1.0: r"$\gamma=1$"}

ALL_LAMBDAS  = sorted({lam for _, lam in CONFIGS})
LAM_COLORS   = {lam: c for lam, c in zip(
    ALL_LAMBDAS,
    ["#1a6faf", "#2980b9", "#5dade2", "#a9cce3",   # blues  (γ=0 lambdas)
     "#1e8449", "#27ae60",                           # greens (γ=0.5)
     "#922b21", "#e74c3c", "#f1948a"]                # reds   (γ=1)
)}


def minimal_circuit_counts(columns):
    support_count = Counter(frozenset(col) for col in columns)
    counts = Counter()
    for support, m in support_count.items():
        if m == 1:
            counts[len(support) + 1] += 1
        else:
            counts[2] += m * (m - 1) // 2
    return counts


# ── Run ──────────────────────────────────────────────────────────────────────

print(f"\n{'γ':>5}  {'λ':>5}  {'r_mean':>8}  {'ρ_mean':>8}  "
      f"{'n_circ':>8}  {'mean_sz':>8}  {'frac_2':>8}")
print("─" * 65)

results = []   # list of dicts for plotting

for gamma, lam in CONFIGS:
    r_list, rho_list, nc_list, ms_list, f2_list = [], [], [], [], []

    for _ in range(REPS):
        eng  = MatroidEngine(n_steps=N_STEPS, k_params=("poisson", 4),
                             C=lam, gamma=gamma, beta=BETA, start_r=START_R)
        data = eng.run()
        counts = minimal_circuit_counts(data["columns"])
        total  = sum(counts.values())
        mean_s = sum(s * counts[s] for s in counts) / total if total else 0
        f2     = counts[2] / total if total else 0
        rho    = data["n"] / data["r"]

        r_list.append(data["r"])
        rho_list.append(rho)
        nc_list.append(total)
        ms_list.append(mean_s)
        f2_list.append(f2)

    r_m   = np.mean(r_list)
    rho_m = np.mean(rho_list)
    nc_m  = np.mean(nc_list)
    ms_m  = np.mean(ms_list)
    f2_m  = np.mean(f2_list)

    print(f"{gamma:>5.1f}  {lam:>5.2f}  {r_m:>8.1f}  {rho_m:>8.2f}  "
          f"{nc_m:>8.0f}  {ms_m:>8.3f}  {f2_m:>8.3f}")

    results.append(dict(gamma=gamma, lam=lam,
                        rho=rho_m, rho_std=np.std(rho_list),
                        mean_size=ms_m, mean_size_std=np.std(ms_list),
                        frac2=f2_m))

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Track legend entries to avoid duplicates
gamma_handles, lam_handles = {}, {}

for d in results:
    gamma, lam = d["gamma"], d["lam"]
    marker = GAMMA_MARKER[gamma]
    color  = LAM_COLORS[lam]

    kw = dict(marker=marker, color=color, markersize=9, lw=0,
              markeredgecolor="k", markeredgewidth=0.5)

    h = ax1.errorbar(d["rho"], d["mean_size"],
                     xerr=d["rho_std"], yerr=d["mean_size_std"],
                     **kw, capsize=3, elinewidth=1, ecolor=color)
    ax2.plot(d["rho"], d["frac2"], **kw)

    if gamma not in gamma_handles:
        gamma_handles[gamma] = plt.Line2D(
            [], [], marker=marker, color="k", lw=0, markersize=8,
            label=GAMMA_LABEL[gamma])
    if lam not in lam_handles:
        lam_handles[lam] = plt.Line2D(
            [], [], marker="o", color=color, lw=0, markersize=8,
            markeredgecolor="k", markeredgewidth=0.5,
            label=fr"$\lambda={lam}$")

# Smooth reference: mean_size as function of rho (for visual guide)
rhos_sorted = sorted(d["rho"] for d in results)
for ax in (ax1, ax2):
    ax.set_xlabel(r"Attachment density $\rho_t = n_t / r_t$", fontsize=12)
    ax.grid(True, alpha=0.3)

ax1.set_ylabel("Mean circuit size", fontsize=12)
ax1.set_title(r"Mean circuit size vs $\rho_t$  ($\beta=1$, $k\sim\mathrm{Poisson}(4)$)",
              fontsize=12)

ax2.set_ylabel(r"Fraction size-2 circuits", fontsize=12)
ax2.set_title(r"Parallel-pair fraction vs $\rho_t$", fontsize=12)
ax2.set_ylim(-0.05, 1.05)

# Combined legend: shape = γ, colour = λ
legend_handles = list(gamma_handles.values()) + list(lam_handles.values())
ax1.legend(handles=legend_handles, fontsize=9, ncol=2,
           loc="upper right", framealpha=0.9)
ax2.legend(handles=legend_handles, fontsize=9, ncol=2,
           loc="lower right", framealpha=0.9)

plt.tight_layout()
path = os.path.join(OUT, "universal_scaling.png")
fig.savefig(path, dpi=150)
print(f"\n  → {path}")
