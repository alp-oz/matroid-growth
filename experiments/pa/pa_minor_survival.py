"""
Survival curves P(T_N > t) vs t for triangle minor appearance.

Loads cached results from minor_appearance_time.json.
Each curve starts at 1 and either drops to 0 (minor always appears)
or plateaus above 0 (fraction of runs where minor never appeared).

Saves: figures/minor_appearance_time.png
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

CACHE   = os.path.join(os.path.dirname(__file__), "minor_appearance_time.json")
OUT_FIG = os.path.join(os.path.dirname(__file__), "..", "..", "figures",
                       "minor_appearance_time.png")

BETAS_PLOT = [0.3, 0.5, 1.0, 1.3, 1.5, 1.8, 2.0]   # drop 0.7 — very close to 0.5

with open(CACHE) as f:
    blob = json.load(f)

rows = {float(r["beta"]): r for r in blob["rows"]}

# ── Survival curves ─────────────────────────────────────────────────────────────

def survival_curve(times_raw, t_grid):
    """
    times_raw: list of ints (step of first appearance) or -1 (never appeared).
    Returns P(T > t) for each t in t_grid.
    """
    n = len(times_raw)
    times = [t if t != -1 else np.inf for t in times_raw]
    return [sum(1 for x in times if x > t) / n for t in t_grid]


# t-grid: 0 to T_MAX, denser at small t
T_MAX  = 400
t_grid = np.unique(np.concatenate([
    [0],
    np.arange(1, 30),
    np.arange(30, 100, 2),
    np.arange(100, T_MAX + 1, 5),
]))

# Colours: blue (low beta) → red (high beta)
cmap   = matplotlib.colormaps["RdYlBu_r"]
colors = {b: cmap(i / (len(BETAS_PLOT) - 1))
          for i, b in enumerate(BETAS_PLOT)}

fig, ax = plt.subplots(figsize=(7, 5))

for beta in BETAS_PLOT:
    row   = rows[beta]
    surv  = survival_curve(row["times"], t_grid)
    p_app = row["p_appeared"]
    plateau = 1.0 - p_app

    ax.step(t_grid, surv, where="post", color=colors[beta], lw=2.0,
            label=fr"$\beta={beta}$" + (f"  (plateau {plateau:.0%})"
                                         if plateau > 0.02 else ""))

    # Mark plateau with a dashed horizontal line for beta >= 1.5
    if plateau > 0.02:
        ax.axhline(plateau, color=colors[beta], ls=":", lw=1.0, alpha=0.55)

ax.axvline(0, color="none")   # padding
ax.set_xlim(0, T_MAX)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel(r"Time $t$ (steps)", fontsize=13)
ax.set_ylabel(r"$P(T_{\triangle} > t)$", fontsize=13)
ax.set_title(
    fr"Survival curves for triangle minor  "
    fr"($r_0=10,\ \gamma=0,\ \lambda=0.05,\ k=4$)",
    fontsize=12)

# Phase-transition marker
ax.axvline(0, color="none")
ax.annotate("", xy=(0, 0), xytext=(0, 0))   # dummy

ax.legend(fontsize=10.5, framealpha=0.9, loc="upper right")
ax.grid(True, alpha=0.22, lw=0.7)
plt.tight_layout()

fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
print(f"  -> {OUT_FIG}")
