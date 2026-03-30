"""
First appearance time of F_7 (Fano plane) as a minor of M_t.

F_7 is detected as a restriction: 7 weight-k PA columns forming all
7 nonzero vectors of a 3-dim GF(2) subspace, i.e. there exist b1,b2,b3
(linearly independent) such that all of
    {b1, b2, b3, b1^b2, b1^b3, b2^b3, b1^b2^b3}
are present in the PA column set. Since all XOR combinations must also
be weight-k PA columns, the weight check is automatic.

Incremental check: when new_bit is added, iterate over all pairs
(b1, b2) in existing pa_bits and test with b3=new_bit — O(n^2) per
new distinct column.

gamma=0, k=4, lambda=0.05, start_r=10, 100 reps.
beta in {0.5, 1.0, 1.5, 2.0}, MAX_STEPS=5000.

Plots survival curves P(T_F7 > t) vs t.
Saves: figures/fano_appearance_time.png
       experiments/pa/fano_appearance_time.json
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
BETAS     = [0.5, 1.0, 1.5, 2.0]
REPS      = 100
MAX_STEPS = 5000

OUT_EXP = os.path.dirname(__file__)
OUT_FIG = os.path.join(OUT_EXP, "..", "..", "figures", "fano_appearance_time.png")
CACHE   = os.path.join(OUT_EXP, "fano_appearance_time.json")

COLORS  = ["#2c7bb6", "#1a9641", "#d7191c", "#7b2d8b"]


# ── Fano detection ─────────────────────────────────────────────────────────────

def fano_check_new(pa_bits, new_bit):
    """
    Check if new_bit completes an F_7 with any two existing PA columns.
    Treats new_bit as the third basis element b3; iterates over all
    pairs (b1, b2) from pa_bits and checks all 7 Fano combinations.
    O(|pa_bits|^2).
    """
    bits = list(pa_bits)
    b3 = new_bit
    for i, b1 in enumerate(bits):
        for b2 in bits[i + 1:]:
            b12  = b1 ^ b2
            b13  = b1 ^ b3
            b23  = b2 ^ b3
            b123 = b1 ^ b2 ^ b3
            if (b12 and b13 and b23 and b123
                    and b12 in pa_bits
                    and b13 in pa_bits
                    and b23 in pa_bits
                    and b123 in pa_bits
                    and len({b1, b2, b3, b12, b13, b23, b123}) == 7):
                return True
    return False


# ── Single run ─────────────────────────────────────────────────────────────────

def run_single(beta, rng):
    """Return step t at which F_7 first appears as restriction, or None."""
    curr_r  = START_R
    degrees = np.ones(START_R, dtype=float)
    pa_bits = set()

    for t in range(1, MAX_STEPS + 1):
        if rng.random() < LAMBDA:
            degrees = np.append(degrees, 1.0)
            curr_r += 1
        else:
            probs   = degrees ** beta
            probs  /= probs.sum()
            chosen  = rng.choice(curr_r, size=K, replace=False, p=probs)
            new_bit = int(sum(1 << int(r) for r in chosen))

            if new_bit not in pa_bits:
                if len(pa_bits) >= 6 and fano_check_new(pa_bits, new_bit):
                    return t
                pa_bits.add(new_bit)

            for r in chosen:
                degrees[int(r)] += 1

    return None


# ── Sweep ──────────────────────────────────────────────────────────────────────

def run_beta(beta, seed):
    rng   = np.random.default_rng(seed)
    times = [run_single(beta, rng) for _ in range(REPS)]
    obs   = [t for t in times if t is not None]
    p_app = len(obs) / REPS
    mean_t = float(np.mean(obs)) if obs else float("nan")
    print(f"  beta={beta}: mean_T={mean_t:8.1f}  "
          f"P(appeared)={p_app:.2f}  n={len(obs)}")
    return {"beta": beta,
            "times": [t if t is not None else -1 for t in times],
            "p_appeared": p_app}


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
    print(f"Recording F_7 appearance times  "
          f"(start_r={START_R}, max_steps={MAX_STEPS}) ...")
    rows = [run_beta(beta, seed=13 + i) for i, beta in enumerate(BETAS)]
    with open(CACHE, "w") as f:
        json.dump({"params_key": PARAMS_KEY, "rows": rows}, f, indent=2)
    print(f"\nCached -> {CACHE}")


# ── Survival curves ────────────────────────────────────────────────────────────

def survival_curve(times_raw, t_grid):
    n     = len(times_raw)
    times = [t if t != -1 else np.inf for t in times_raw]
    return [sum(1 for x in times if x > t) / n for t in t_grid]


T_MAX  = MAX_STEPS
t_grid = np.unique(np.concatenate([
    [0],
    np.arange(1, 50),
    np.arange(50, 300, 5),
    np.arange(300, 1000, 20),
    np.arange(1000, T_MAX + 1, 100),
]))

fig, ax = plt.subplots(figsize=(7, 5))

for row, color in zip(rows, COLORS):
    beta    = row["beta"]
    surv    = survival_curve(row["times"], t_grid)
    plateau = 1.0 - row["p_appeared"]

    ax.step(t_grid, surv, where="post", color=color, lw=2.0,
            label=fr"$\beta={beta}$" +
                  (f"  (plateau {plateau:.0%})" if plateau > 0.02 else ""))

    if plateau > 0.02:
        ax.axhline(plateau, color=color, ls=":", lw=1.0, alpha=0.5)

ax.set_xlim(0, min(T_MAX, 2000))
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel(r"Time $t$ (steps)", fontsize=13)
ax.set_ylabel(r"$P(T_{F_7} > t)$", fontsize=13)
ax.set_title(
    fr"Survival curves for $F_7$ minor  "
    fr"($r_0={START_R},\ \gamma=0,\ \lambda={LAMBDA},\ k={K}$)",
    fontsize=12)
ax.legend(fontsize=11, framealpha=0.9, loc="upper right")
ax.grid(True, alpha=0.22, lw=0.7)
plt.tight_layout()

fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
print(f"  -> {OUT_FIG}")
