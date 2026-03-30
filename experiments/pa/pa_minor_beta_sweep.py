"""
Triangle minor threshold — β-sweep with start_r=5.

With k/r large (k=4, start_r=5), PA bias strongly controls support diversity.
β ∈ {0.5, 1.0, 1.5, 2.0}, γ=0, λ=0.05, 50 reps.
n_steps: 30 log-spaced values from 1 to 500 (ρ_max ≈ 17).

Expected: clear β-separation — large β suppresses triangle formation.

Saves: minor_beta_sweep.png, minor_beta_sweep.json
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
BETAS    = [0.5, 1.0, 1.5, 2.0]
N_LOW    = 1
N_HIGH   = 500
N_POINTS = 30
K        = 4
REPS     = 50
START_R  = 5

OUT   = os.path.dirname(__file__)
CACHE = os.path.join(OUT, "minor_beta_sweep.json")

COLORS = ["#2c7bb6", "#1a9641", "#d7191c", "#7b2d8b"]


def make_n_steps():
    return np.unique(
        np.round(np.logspace(np.log10(N_LOW), np.log10(N_HIGH),
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


def run_sweep():
    n_steps_arr = make_n_steps()
    curves = {}
    for beta in BETAS:
        rho_vals, prob_vals = [], []
        for n_steps in n_steps_arr:
            hits = 0; rho_sum = 0.0
            for _ in range(REPS):
                eng  = MatroidEngine(n_steps=n_steps, k_params=K, C=LAMBDA,
                                     gamma=0.0, beta=beta, start_r=START_R)
                data = eng.run()
                rho_sum += data["n"] / data["r"]
                if has_triangle(data["columns"]):
                    hits += 1
            rho_vals.append(rho_sum / REPS)
            prob_vals.append(hits / REPS)
        curves[beta] = (rho_vals, prob_vals)
        crossings = [rho_vals[i] for i, p in enumerate(prob_vals) if p >= 0.5]
        rho_star  = f"{crossings[0]:.2f}" if crossings else "—"
        print(f"  β={beta}: ρ*≈{rho_star}  "
              f"ρ∈[{rho_vals[0]:.2f},{rho_vals[-1]:.2f}]  "
              f"P_max={max(prob_vals):.2f}")
    return curves


# ── Load or compute ────────────────────────────────────────────────────────────

PARAMS_KEY = {"lambda": LAMBDA, "betas": BETAS, "n_low": N_LOW,
              "n_high": N_HIGH, "k": K, "reps": REPS, "start_r": START_R}

blob = None
if os.path.exists(CACHE):
    with open(CACHE) as f:
        blob = json.load(f)
    if blob.get("params_key") != PARAMS_KEY:
        print("Cache parameters differ — recomputing.")
        blob = None
    else:
        print(f"Loading cached results from {CACHE}")
        curves = {float(row["beta"]): (row["rho"], row["prob"])
                  for row in blob["curves"]}

if blob is None:
    print(f"Running β-sweep (start_r={START_R}) …")
    curves = run_sweep()
    cache_out = {
        "params_key": PARAMS_KEY,
        "curves": [{"beta": b, "rho": curves[b][0], "prob": curves[b][1]}
                   for b in BETAS],
    }
    with open(CACHE, "w") as f:
        json.dump(cache_out, f, indent=2)
    print(f"\nCached → {CACHE}")


# ── Figure ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 5))

for beta, color in zip(BETAS, COLORS):
    rho_vals, prob_vals = curves[beta]
    ax.plot(rho_vals, prob_vals, "-", color=color, lw=2.2,
            label=fr"$\beta={beta}$")

ax.axhline(0.5, color="gray", ls="--", lw=1.2, alpha=0.6)
ax.set_xlabel(r"Attachment density $\rho_t = n_t / r_t$", fontsize=13)
ax.set_ylabel(r"$P(\,\mathrm{triangle\ minor}\,)$", fontsize=13)
ax.set_title(
    fr"Triangle minor: $\beta$-sweep  "
    fr"($r_0={START_R},\ \gamma=0,\ \lambda={LAMBDA},\ k={K}$)",
    fontsize=12)
ax.set_ylim(-0.04, 1.04)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.25, lw=0.7)
plt.tight_layout()

path = os.path.join(OUT, "minor_beta_sweep.png")
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"  → {path}")
