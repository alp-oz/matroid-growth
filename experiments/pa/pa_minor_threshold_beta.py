"""
P(triangle minor) vs beta for fixed attachment density d = E[n_t / r_t].

For each target density d, n_steps is chosen so that
    E[n_t/r_t] = (start_r + 0.95*n) / (start_r + 0.05*n) = d
giving n = start_r*(d-1) / (0.95 - 0.05*d).

gamma=0, k=4, lambda=0.05, start_r=10, 100 reps.
d in {3.0, 5.0, 8.0, 12.0},  beta: 20 log-spaced values in [0.1, 5.0].

Saves: figures/minor_threshold_clean.png
       experiments/pa/minor_threshold_beta.json
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.engine import MatroidEngine
from analysis.probe_minors import convert_to_bitsets

LAMBDA    = 0.05
K         = 4
START_R   = 10
DENSITIES = [3.0, 5.0, 8.0, 12.0]
BETAS     = list(np.round(np.logspace(np.log10(0.1), np.log10(5.0), 20), 4))
REPS      = 100

OUT_EXP = os.path.dirname(__file__)
OUT_FIG = os.path.join(OUT_EXP, "..", "..", "figures", "minor_threshold_clean.png")
CACHE   = os.path.join(OUT_EXP, "minor_threshold_beta.json")

COLORS = ["#4393c3", "#74c476", "#fd8d3c", "#d6604d"]


def n_steps_for_density(d):
    """n_steps such that E[n_t/r_t] = d."""
    if d <= 1.0:
        return 0
    n = START_R * (d - 1.0) / (0.95 - 0.05 * d)
    return max(1, int(round(n)))


def has_triangle(columns):
    bits = list(set(convert_to_bitsets(columns)) - {0})
    if len(bits) < 3:
        return False
    lookup = set(bits)
    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            c = bits[i] ^ bits[j]
            if c and c in lookup:
                return True
    return False


def run_sweep():
    curves = {}
    for d in DENSITIES:
        n = n_steps_for_density(d)
        probs = []
        for beta in BETAS:
            if n == 0:
                probs.append(0.0)
                continue
            hits = sum(
                1 for _ in range(REPS)
                if has_triangle(
                    MatroidEngine(n_steps=n, k_params=K, C=LAMBDA,
                                  gamma=0.0, beta=beta,
                                  start_r=START_R).run()["columns"]
                )
            )
            probs.append(hits / REPS)
        curves[d] = probs
        crossings = [BETAS[i] for i, p in enumerate(probs) if p <= 0.5]
        bstar = f"{crossings[0]:.2f}" if crossings else "—"
        print(f"  d={d:.1f}  n_steps={n:3d}  beta*~{bstar}  "
              f"P_max={max(probs):.2f}  P_min={min(probs):.2f}")
    return curves


# ── Load or compute ────────────────────────────────────────────────────────────

PARAMS_KEY = {"lambda": LAMBDA, "k": K, "start_r": START_R,
              "densities": DENSITIES, "betas": BETAS, "reps": REPS}

blob = None
if os.path.exists(CACHE):
    with open(CACHE) as f:
        blob = json.load(f)
    if blob.get("params_key") != PARAMS_KEY:
        print("Cache differs — recomputing.")
        blob = None
    else:
        print(f"Loading cache from {CACHE}")
        curves = {float(r["d"]): r["probs"] for r in blob["curves"]}

if blob is None:
    print(f"P(triangle) vs beta for fixed density  "
          f"(start_r={START_R}, reps={REPS}) ...")
    curves = run_sweep()
    with open(CACHE, "w") as f:
        json.dump({
            "params_key": PARAMS_KEY,
            "curves": [{"d": d, "betas": BETAS, "probs": curves[d]}
                       for d in DENSITIES]
        }, f, indent=2)
    print(f"\nCached -> {CACHE}")


# ── Figure ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 5))

for d, color in zip(DENSITIES, COLORS):
    probs = curves[d]
    ax.plot(BETAS, probs, "-", color=color, lw=2.2,
            label=fr"$d={d:.0f}$")

ax.axhline(0.5, color="gray", ls="--", lw=1.1, alpha=0.55)
ax.set_xscale("log")
ax.set_xlabel(r"Attachment bias $\beta$", fontsize=13)
ax.set_ylabel(r"$P(\text{triangle minor})$", fontsize=13)
ax.set_title(
    fr"Triangle minor threshold vs $\beta$  "
    fr"($r_0={START_R},\ \gamma=0,\ \lambda={LAMBDA},\ k={K}$)",
    fontsize=12)
ax.set_ylim(-0.04, 1.04)
ax.set_xlim(BETAS[0] * 0.9, BETAS[-1] * 1.1)
ax.legend(fontsize=12, framealpha=0.9, title=r"density $d=n/r$",
          title_fontsize=10)
ax.grid(True, alpha=0.22, lw=0.7, which="both")
plt.tight_layout()

fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
print(f"  -> {OUT_FIG}")
