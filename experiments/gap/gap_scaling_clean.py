"""
Clean log-log plot of mixing time tau vs n for six CSS code families.
Loads data from results/gap_scaling_results.json — no simulation.

Fitting: delta_ind ~ n^{-alpha} (OLS on log-log), then tau = 1/delta plotted.
Reference lines ~ n^0.5 and ~ n^1.
Saves: figures/gap_scaling_clean.png
       experiments/gap/gap_scaling_clean.json  (data snapshot for future use)
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from collections import defaultdict

# ── Load data ──────────────────────────────────────────────────────────────────

ROOT     = os.path.join(os.path.dirname(__file__), "..", "..")
SRC      = os.path.join(ROOT, "results", "gap_scaling_results.json")
OUT_FIG  = os.path.join(ROOT, "figures", "gap_scaling_clean.png")
SNAPSHOT = os.path.join(os.path.dirname(__file__), "gap_scaling_clean.json")

with open(SRC) as f:
    raw = json.load(f)

# Group by family; average delta_ind for duplicate (family, n) entries
family_pts = defaultdict(lambda: defaultdict(list))
for rec in raw:
    family_pts[rec["family"]][rec["n"]].append(rec["delta_ind"])

families = ["Bicycle", "Toric", "HGP", "BB", "FB", "QT"]
COLORS   = {
    "Bicycle": "#2166ac",
    "Toric":   "#d73027",
    "HGP":     "#1a9641",
    "BB":      "#762a83",
    "FB":      "#e08214",
    "QT":      "#1d91c0",
}

# Build per-family arrays sorted by n; tau = 1/delta_ind
data = {}
for fam in families:
    pts  = sorted(family_pts[fam].items())
    ns   = np.array([p[0] for p in pts], dtype=float)
    deltas = np.array([np.mean(p[1]) for p in pts], dtype=float)
    taus = 1.0 / deltas
    data[fam] = (ns, taus, deltas)

# Save snapshot
with open(SNAPSHOT, "w") as f:
    json.dump({fam: {"n":   data[fam][0].tolist(),
                     "tau": data[fam][1].tolist(),
                     "delta_ind": data[fam][2].tolist()}
               for fam in families}, f, indent=2)
print(f"Snapshot -> {SNAPSHOT}")

# ── Power-law fit on delta_ind ~ n^{-alpha} ────────────────────────────────────
all_n     = np.concatenate([data[fam][0] for fam in families])
all_delta = np.concatenate([data[fam][2] for fam in families])

slope, intercept, r, *_ = linregress(np.log(all_n), np.log(all_delta))
alpha  = -slope          # delta ~ n^{-alpha}  =>  tau ~ n^{alpha}
C_tau  = np.exp(-intercept)   # tau = C_tau * n^alpha
print(f"Fit: delta ~ n^{slope:.3f}  =>  tau ~ {C_tau:.3f} * n^{alpha:.3f}  "
      f"(R²={r**2:.3f})")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.8))

n_all = np.concatenate([data[fam][0] for fam in families])
n_min, n_max = n_all.min() * 0.85, n_all.max() * 1.3
n_ref = np.linspace(n_min, n_max, 300)

# Reference lines — anchored at median point
tau_all = np.concatenate([data[fam][1] for fam in families])
anchor_n   = np.median(n_all)
anchor_tau = np.median(tau_all)

for exp, ls, label in [(0.5, (0, (6, 2)),   r"$\sim n^{0.5}$"),
                       (1.0, (0, (1, 1.5)), r"$\sim n^{1}$")]:
    c = anchor_tau / anchor_n**exp
    ax.plot(n_ref, c * n_ref**exp, linestyle=ls, color="#aaaaaa",
            lw=1.1, zorder=1, label=label)

# Per-family lines with small markers
for fam in families:
    ns, taus, _ = data[fam]
    ax.plot(ns, taus, "-o", color=COLORS[fam], lw=1.6,
            markersize=5, markeredgewidth=0.5, markeredgecolor="white",
            zorder=3, label=fam)

# Fitted power law — thick black dashed
ax.plot(n_ref, C_tau * n_ref**alpha, "--", color="black", lw=2.2, zorder=4,
        label=fr"fit: $\tau \sim n^{{{alpha:.2f}}}$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Physical qubits $n$", fontsize=12)
ax.set_ylabel(r"Mixing time $\tau$", fontsize=12)
ax.set_title("Circuit walk mixing time across CSS code families",
             fontsize=11)
ax.set_xlim(n_min, n_max)

# Legend outside on the right
ax.legend(fontsize=9, framealpha=0.9, loc="upper left",
          bbox_to_anchor=(1.02, 1), borderaxespad=0)

ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=9)
plt.tight_layout(rect=[0, 0, 0.82, 1])

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
print(f"Saved -> {OUT_FIG}")
