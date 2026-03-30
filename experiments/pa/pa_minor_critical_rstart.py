"""
Direction 2: critical r_start for eventual minor appearance.

Fix large n_steps=3000 (rho_t -> ~19, near asymptotic limit).
Sweep start_r in {4,...,30}, beta in {0.5, 1.0, 1.5, 2.0}.
For each (start_r, beta): compute P(triangle minor appears within n_steps).

For beta > 1 (super-linear PA): condensation traps the matroid in a
low-diversity state. If start_r is too small, diversity never develops
and P -> 0. If start_r is large enough, initial diversity survives
condensation and P -> 1.

Expected: sigmoid in start_r with r_c*(beta) INCREASING in beta.

Saves: minor_critical_rstart.png, minor_critical_rstart.json
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.engine import MatroidEngine
from analysis.probe_minors import convert_to_bitsets

LAMBDA   = 0.05
K        = 4
BETAS    = [0.5, 1.0, 1.5, 2.0]
N_STEPS  = 3000
START_RS = [4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
REPS     = 100

OUT   = os.path.dirname(__file__)
CACHE = os.path.join(OUT, "minor_critical_rstart.json")
COLORS = ["#2c7bb6", "#1a9641", "#d7191c", "#7b2d8b"]


def has_triangle(columns):
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


def run_sweep():
    curves = {}
    for beta in BETAS:
        sr_vals, prob_vals = [], []
        for start_r in START_RS:
            hits = 0
            for _ in range(REPS):
                eng  = MatroidEngine(n_steps=N_STEPS, k_params=K, C=LAMBDA,
                                     gamma=0.0, beta=beta, start_r=start_r)
                data = eng.run()
                if has_triangle(data["columns"]):
                    hits += 1
            sr_vals.append(start_r)
            prob_vals.append(hits / REPS)
        curves[beta] = (sr_vals, prob_vals)
        crossings = [sr_vals[i] for i, p in enumerate(prob_vals) if p >= 0.5]
        rstar = f"{crossings[0]}" if crossings else "—"
        print(f"  beta={beta}: r_c*~{rstar}  "
              f"P={[f'{p:.2f}' for p in prob_vals]}")
    return curves


# ── Load or compute ────────────────────────────────────────────────────────────

PARAMS_KEY = {"lambda": LAMBDA, "k": K, "betas": BETAS,
              "n_steps": N_STEPS, "start_rs": START_RS, "reps": REPS}

blob = None
if os.path.exists(CACHE):
    with open(CACHE) as f:
        blob = json.load(f)
    if blob.get("params_key") != PARAMS_KEY:
        print("Cache differs — recomputing.")
        blob = None
    else:
        print(f"Loading cache from {CACHE}")
        curves = {float(row["beta"]): (row["start_rs"], row["probs"])
                  for row in blob["curves"]}

if blob is None:
    print(f"Direction 2: critical r_start  (n_steps={N_STEPS}) ...")
    curves = run_sweep()
    cache_out = {
        "params_key": PARAMS_KEY,
        "curves": [{"beta": b, "start_rs": curves[b][0], "probs": curves[b][1]}
                   for b in BETAS],
    }
    with open(CACHE, "w") as f:
        json.dump(cache_out, f, indent=2)
    print(f"\nCached -> {CACHE}")


# ── Figure ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 5))

for beta, color in zip(BETAS, COLORS):
    sr_vals, prob_vals = curves[beta]
    ax.plot(sr_vals, prob_vals, "-o", color=color, lw=2.2,
            markersize=5, label=fr"$\beta={beta}$")

ax.axhline(0.5, color="gray", ls="--", lw=1.2, alpha=0.6)
ax.set_xlabel(r"Initial rank $r_0 = \mathrm{start\_r}$", fontsize=13)
ax.set_ylabel(r"$P(\text{triangle minor in } M_t,\ t=" + str(N_STEPS) + r")$",
              fontsize=12)
ax.set_title(
    fr"Critical $r_0$: eventual triangle appearance  "
    fr"($\gamma=0,\ \lambda={LAMBDA},\ k={K},\ n_\mathrm{{steps}}={N_STEPS}$)",
    fontsize=11)
ax.set_ylim(-0.04, 1.04)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.25, lw=0.7)
plt.tight_layout()

path = os.path.join(OUT, "minor_critical_rstart.png")
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"  -> {path}")
