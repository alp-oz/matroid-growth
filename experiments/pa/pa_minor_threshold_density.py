"""
P(triangle minor) vs density d = E[n_PA / r_t] for fixed beta.

Density redefined as d(n) = (1-lambda)*n / (start_r + lambda*n)
so that d=0 at n=0 (initial identity matrix, no PA columns yet).

n_steps swept over ~30 log-spaced values from 1 to 100.
Expected density: d in [0, ~6.3].

gamma=0, k=4, lambda=0.05, start_r=10, 300 reps.
beta in {0.7, 1.0, 1.3, 1.6}.

Saves: figures/minor_threshold_clean.png
       experiments/pa/minor_threshold_density.json
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBDA   = 0.05
K        = 4
START_R  = 10
BETAS    = [0.5, 1.0, 1.3, 1.6]
N_STEPS  = [int(x) for x in np.unique(np.round(
                np.logspace(0, 2, 32)
            ).astype(int))]          # 1 … 100, ~28 unique values
REPS     = 300

OUT_EXP = os.path.dirname(__file__)
OUT_FIG = os.path.join(OUT_EXP, "..", "..", "figures", "minor_threshold_clean.png")
CACHE   = os.path.join(OUT_EXP, "minor_threshold_density.json")

# colours + markers to match third-panel style
COLORS  = ["#6baed6", "#4dac26", "#6a3d9a", "#f1a340"]
MARKERS = ["o", "s", "^", "D"]


def expected_density(n):
    """d = E[#PA columns] / E[rank] = (1-lam)*n / (r0 + lam*n)."""
    return (1 - LAMBDA) * n / (START_R + LAMBDA * n)


def run_single(n_steps, beta, rng):
    """Return True if triangle minor appears within n_steps."""
    curr_r  = START_R
    degrees = np.ones(START_R, dtype=float)
    pa_bits = set()

    for _ in range(n_steps):
        if rng.random() < LAMBDA:
            degrees = np.append(degrees, 1.0)
            curr_r += 1
        else:
            probs   = degrees ** beta
            probs  /= probs.sum()
            chosen  = rng.choice(curr_r, size=K, replace=False, p=probs)
            new_bit = int(sum(1 << int(r) for r in chosen))

            if new_bit not in pa_bits:
                for b in pa_bits:
                    xor = b ^ new_bit
                    if xor and xor != b and xor in pa_bits:
                        return True
                pa_bits.add(new_bit)

            for r in chosen:
                degrees[int(r)] += 1

    return False


def run_sweep():
    curves = {}
    for beta in BETAS:
        rng   = np.random.default_rng(42 + int(beta * 100))
        probs = []
        for n in N_STEPS:
            hits = sum(1 for _ in range(REPS) if run_single(n, beta, rng))
            probs.append(hits / REPS)
        curves[beta] = probs
        dens = [expected_density(n) for n in N_STEPS]
        crossings = [dens[i] for i, p in enumerate(probs) if p >= 0.5]
        dstar = f"{crossings[0]:.2f}" if crossings else "—"
        print(f"  beta={beta:.1f}  d*~{dstar}  "
              f"P_max={max(probs):.2f}  P_min={min(probs):.2f}")
    return curves


# ── Load or compute ────────────────────────────────────────────────────────────

PARAMS_KEY = {"lambda": LAMBDA, "k": K, "start_r": START_R,
              "betas": BETAS, "n_steps": N_STEPS, "reps": REPS}

blob = None
if os.path.exists(CACHE):
    with open(CACHE) as f:
        blob = json.load(f)
    if blob.get("params_key") != PARAMS_KEY:
        print("Cache differs — recomputing.")
        blob = None
    else:
        print(f"Loading cache from {CACHE}")
        curves = {float(r["beta"]): r["probs"] for r in blob["curves"]}

if blob is None:
    print(f"P(triangle) vs density  (start_r={START_R}, reps={REPS}) ...")
    curves = run_sweep()
    with open(CACHE, "w") as f:
        json.dump({
            "params_key": PARAMS_KEY,
            "curves": [{"beta": b,
                        "n_steps": N_STEPS,
                        "densities": [expected_density(n) for n in N_STEPS],
                        "probs": curves[b]}
                       for b in BETAS]
        }, f, indent=2)
    print(f"\nCached -> {CACHE}")


# ── Figure ─────────────────────────────────────────────────────────────────────

from scipy.optimize import curve_fit

def logistic4(x, x0, k, L, U):
    """4-parameter logistic: free lower/upper asymptotes."""
    return L + (U - L) / (1.0 + np.exp(-k * (x - x0)))

densities = np.array([expected_density(n) for n in N_STEPS])
x_smooth  = np.linspace(0, 5.0, 400)

fig, ax = plt.subplots(figsize=(5.5, 4.5))

for beta, color in zip(BETAS, COLORS):
    probs = np.array(curves[beta])

    # scatter: small transparent circles
    ax.scatter(densities, probs, color=color, s=18, alpha=0.3,
               linewidths=0, zorder=2)

    # 4-parameter logistic fit
    try:
        x0_guess = float(densities[np.argmin(np.abs(probs - 0.5))])
        popt, _ = curve_fit(
            logistic4, densities, probs,
            p0=[x0_guess, 3.0, 0.0, max(probs)],
            bounds=([0.0, 0.1, -0.05, 0.7], [8.0, 50.0, 0.15, 1.05]),
            maxfev=10000)
        ax.plot(x_smooth, logistic4(x_smooth, *popt), color=color,
                lw=1.4, zorder=3, label=fr"$\beta={beta}$")
    except Exception:
        ax.plot(densities, probs, color=color, lw=1.4,
                zorder=3, label=fr"$\beta={beta}$")

ax.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
ax.set_xlabel(r"Attachment density: $\rho = n_t / r_t$", fontsize=11)
ax.set_ylabel(r"$P(\text{triangle minor})$", fontsize=11)
ax.set_title(
    fr"Triangle minor threshold  ($\lambda={LAMBDA},\ k={K},\ r_0={START_R},\ \gamma=0$)",
    fontsize=10)
ax.set_ylim(-0.04, 1.04)
ax.set_xlim(-0.15, 5.0)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.legend(fontsize=9.5, framealpha=0.85, loc="upper left",
          title=r"$\beta$", title_fontsize=9)
ax.grid(True, lw=0.5, alpha=0.4)
ax.tick_params(labelsize=9)
plt.tight_layout()

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
print(f"  -> {OUT_FIG}")
