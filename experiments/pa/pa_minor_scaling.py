"""
Triangle minor threshold — scaling with start_r.

β=1.0, k=4, λ=0.05, γ=0.
start_r ∈ {5, 10, 20, 40}.
For each start_r: 30 log-spaced n_steps, ρ_max ≈ 7.
REPS = 50.

If the sigmoid sharpens and ρ* converges as start_r grows, that is
evidence for a genuine threshold; if ρ* drifts, it is a finite-size effect.

Saves: minor_scaling.png, minor_scaling.json
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.engine import MatroidEngine
from analysis.probe_minors import convert_to_bitsets

BETA     = 1.0
LAMBDA   = 0.05
K        = 4
REPS     = 50
N_POINTS = 30
N_LOW    = 1
START_RS = [5, 10, 20, 40]
# N_HIGH chosen so ρ_max ≈ 7:
#   (sr + 0.95*nh) / (sr + 0.05*nh) = 7  →  nh ≈ 10*sr
N_HIGH_MAP = {5: 50, 10: 100, 20: 200, 40: 400}

OUT   = os.path.dirname(__file__)
CACHE = os.path.join(OUT, "minor_scaling.json")

COLORS = ["#2c7bb6", "#1a9641", "#d7191c", "#7b2d8b"]


def make_n_steps(n_high):
    return np.unique(
        np.round(np.logspace(np.log10(N_LOW), np.log10(n_high),
                             N_POINTS)).astype(int)
    ).tolist()


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


def run_sweep(start_r):
    n_steps_arr = make_n_steps(N_HIGH_MAP[start_r])
    rho_vals, prob_vals = [], []
    for n_steps in n_steps_arr:
        hits = 0; rho_sum = 0.0
        for _ in range(REPS):
            eng  = MatroidEngine(n_steps=n_steps, k_params=K, C=LAMBDA,
                                 gamma=0.0, beta=BETA, start_r=start_r)
            data = eng.run()
            rho_sum += data["n"] / data["r"]
            if has_triangle(data["columns"]):
                hits += 1
        rho_vals.append(rho_sum / REPS)
        prob_vals.append(hits / REPS)
    crossings = [rho_vals[i] for i, p in enumerate(prob_vals) if p >= 0.5]
    rho_star  = f"{crossings[0]:.2f}" if crossings else "—"
    print(f"  start_r={start_r:2d}: ρ*≈{rho_star}  "
          f"ρ∈[{rho_vals[0]:.2f},{rho_vals[-1]:.2f}]  "
          f"P_max={max(prob_vals):.2f}")
    return rho_vals, prob_vals


# ── Load or compute ────────────────────────────────────────────────────────────

PARAMS_KEY = {"beta": BETA, "lambda": LAMBDA, "k": K, "reps": REPS,
              "n_low": N_LOW, "start_rs": START_RS,
              "n_high_map": {str(k): v for k, v in N_HIGH_MAP.items()}}

blob = None
if os.path.exists(CACHE):
    with open(CACHE) as f:
        blob = json.load(f)
    if blob.get("params_key") != PARAMS_KEY:
        print("Cache parameters differ — recomputing.")
        blob = None
    else:
        print(f"Loading cached results from {CACHE}")
        curves = {int(row["start_r"]): (row["rho"], row["prob"])
                  for row in blob["curves"]}

if blob is None:
    print(f"Running scaling sweep (β={BETA}) …")
    curves = {}
    for start_r in START_RS:
        rho_vals, prob_vals = run_sweep(start_r)
        curves[start_r] = (rho_vals, prob_vals)

    cache_out = {
        "params_key": PARAMS_KEY,
        "curves": [{"start_r": sr, "rho": curves[sr][0], "prob": curves[sr][1]}
                   for sr in START_RS],
    }
    with open(CACHE, "w") as f:
        json.dump(cache_out, f, indent=2)
    print(f"\nCached → {CACHE}")


# ── Figure ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 5))

for start_r, color in zip(START_RS, COLORS):
    rho_vals, prob_vals = curves[start_r]
    ax.plot(rho_vals, prob_vals, "-", color=color, lw=2.2,
            label=fr"$r_0={start_r}$")

ax.axhline(0.5, color="gray", ls="--", lw=1.2, alpha=0.6)
ax.set_xlabel(r"Attachment density $\rho_t = n_t / r_t$", fontsize=13)
ax.set_ylabel(r"$P(\,\mathrm{triangle\ minor}\,)$", fontsize=13)
ax.set_title(
    fr"Triangle minor: scaling with $r_0$  "
    fr"($\beta={BETA},\ \gamma=0,\ \lambda={LAMBDA},\ k={K}$)",
    fontsize=12)
ax.set_ylim(-0.04, 1.04)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.25, lw=0.7)
plt.tight_layout()

path = os.path.join(OUT, "minor_scaling.png")
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"  → {path}")
