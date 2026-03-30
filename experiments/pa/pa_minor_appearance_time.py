"""
First appearance time T of triangle minor in PA binary matroid.

gamma=0, k=4, lambda=0.05, start_r=10, 100 reps.
beta in {0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0}.

T = number of steps (column + row additions) until triangle first appears.
Runs where triangle never appears within MAX_STEPS are recorded as NaN.

Plot: mean T vs beta, error bars = +/-1 std.
      Annotate each point with P(appeared).

Saves: figures/minor_appearance_time.png
       experiments/pa/minor_appearance_time.json
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBDA    = 0.05
K         = 4
START_R   = 10
BETAS     = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0]
REPS      = 100
MAX_STEPS = 5000

OUT_EXP = os.path.dirname(__file__)
OUT_FIG = os.path.join(OUT_EXP, "..", "..", "figures")
CACHE   = os.path.join(OUT_EXP, "minor_appearance_time.json")


def run_single(beta, rng):
    """Return step t at which triangle first appears, or None."""
    curr_r   = START_R
    degrees  = np.ones(START_R, dtype=float)
    pa_bits  = set()

    for t in range(1, MAX_STEPS + 1):
        if rng.random() < LAMBDA:
            # Row addition: new row gets degree 1 (identity column)
            degrees = np.append(degrees, 1.0)
            curr_r += 1
        else:
            # Column addition by PA
            probs  = degrees ** beta
            probs /= probs.sum()
            chosen  = rng.choice(curr_r, size=K, replace=False, p=probs)
            new_bit = int(sum(1 << int(r) for r in chosen))

            if new_bit not in pa_bits:
                # Incremental triangle check
                for b in pa_bits:
                    xor = b ^ new_bit
                    if xor and xor != b and xor in pa_bits:
                        return t
                pa_bits.add(new_bit)

            for r in chosen:
                degrees[int(r)] += 1

    return None


def run_beta(beta, seed):
    rng     = np.random.default_rng(seed)
    times   = [run_single(beta, rng) for _ in range(REPS)]
    obs     = [t for t in times if t is not None]
    p_app   = len(obs) / REPS
    mean_t  = float(np.mean(obs))  if obs else float("nan")
    std_t   = float(np.std(obs))   if obs else float("nan")
    sem_t   = float(np.std(obs) / np.sqrt(len(obs))) if obs else float("nan")
    print(f"  beta={beta:.1f}: mean_T={mean_t:7.1f}  std={std_t:7.1f}  "
          f"P(appeared)={p_app:.2f}  n={len(obs)}")
    return {"beta": beta, "times": [t if t is not None else -1 for t in times],
            "mean": mean_t, "std": std_t, "sem": sem_t, "p_appeared": p_app}


# ── Load or compute ────────────────────────────────────────────────────────────

PARAMS_KEY = {"lambda": LAMBDA, "k": K, "start_r": START_R,
              "betas": BETAS, "reps": REPS, "max_steps": MAX_STEPS}

blob = None
if os.path.exists(CACHE):
    with open(CACHE) as f:
        blob = json.load(f)
    if blob.get("params_key") != PARAMS_KEY:
        print("Cache differs — recomputing.")
        blob = None
    else:
        print(f"Loading cache from {CACHE}")
        rows = blob["rows"]

if blob is None:
    print(f"Recording triangle appearance times  "
          f"(start_r={START_R}, max_steps={MAX_STEPS}) ...")
    rows = [run_beta(beta, seed=7 + i) for i, beta in enumerate(BETAS)]
    with open(CACHE, "w") as f:
        json.dump({"params_key": PARAMS_KEY, "rows": rows}, f, indent=2)
    print(f"\nCached -> {CACHE}")


# ── Figure ─────────────────────────────────────────────────────────────────────

betas   = [r["beta"]      for r in rows]
means   = [r["mean"]      for r in rows]
stds    = [r["std"]       for r in rows]
p_apps  = [r["p_appeared"] for r in rows]

fig, ax = plt.subplots(figsize=(7, 5))

ax.errorbar(betas, means, yerr=stds, fmt="o-", color="#2c7bb6",
            lw=2.2, markersize=7, capsize=5, capthick=1.8,
            elinewidth=1.5, label=r"mean $T$ $\pm$ 1 std")

# Annotate P(appeared) below each point
for b, m, p in zip(betas, means, p_apps):
    if not np.isnan(m):
        ax.annotate(f"{p:.0%}", xy=(b, m), xytext=(0, -18),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, color="gray")

ax.axvline(1.0, color="gray", ls="--", lw=1.2, alpha=0.7,
           label=r"$\beta = 1$ (linear PA)")

ax.set_xlabel(r"Attachment bias $\beta$", fontsize=13)
ax.set_ylabel(r"Mean first appearance time $\mathbb{E}[T]$", fontsize=13)
ax.set_title(
    fr"Triangle minor appearance time vs $\beta$  "
    fr"($r_0={START_R},\ \gamma=0,\ \lambda={LAMBDA},\ k={K}$)",
    fontsize=12)
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.25, lw=0.7)
plt.tight_layout()

path = os.path.join(OUT_FIG, "minor_appearance_time.png")
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"  -> {path}")
