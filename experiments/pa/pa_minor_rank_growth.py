"""
Direction 1: P(triangle minor appeared by r_t = r) vs r_t.

Inline PA from start_r = k = 4 (minimal initial conditions).
Run until r_t reaches R_MAX. After each step, check for triangle minor
among PA-attached columns (weight-k bitsets only; identity columns cannot
participate in triangles when k=4).

Plot CDF: P(first_appearance_r <= r) vs r for beta in {0.5, 1.0, 1.5, 2.0}.

Expected: sigmoid in r_t, r_c INCREASING in beta (stronger PA = more
concentrated = needs higher rank before diverse XOR-triples appear).

Saves: minor_rank_growth.png, minor_rank_growth.json
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBDA   = 0.05
K        = 4
BETAS    = [0.5, 1.0, 1.5, 2.0]
R_MAX    = 60    # run until r_t reaches this
REPS     = 300
START_R  = K     # minimal initial conditions

OUT   = os.path.dirname(__file__)
CACHE = os.path.join(OUT, "minor_rank_growth.json")
COLORS = ["#2c7bb6", "#1a9641", "#d7191c", "#7b2d8b"]


def run_single(beta, rng):
    """
    Run inline PA from start_r=K until r_t=R_MAX.
    Returns r_t at first triangle appearance, or None.
    """
    curr_r = K
    degrees = np.ones(K, dtype=float)
    pa_bits = set()   # distinct weight-K column bitsets (no identity)
    max_steps = int(R_MAX / LAMBDA * 5)

    for _ in range(max_steps):
        if curr_r >= R_MAX:
            break

        if rng.random() < LAMBDA:
            # Row addition: new row gets degree 1 (via its identity column)
            degrees = np.append(degrees, 1.0)
            curr_r += 1
        else:
            # Column addition: choose K rows by PA
            probs = degrees ** beta
            probs /= probs.sum()
            chosen = rng.choice(curr_r, size=K, replace=False, p=probs)
            new_bit = int(sum(1 << int(r) for r in chosen))

            if new_bit not in pa_bits:
                # Incremental triangle check: new_bit vs all existing PA bits
                for b in pa_bits:
                    xor = b ^ new_bit
                    if xor != 0 and xor != b and xor in pa_bits:
                        return curr_r   # triangle found at current r_t
                pa_bits.add(new_bit)

            for r in chosen:
                degrees[int(r)] += 1

    return None   # never appeared within R_MAX


def run_beta(beta, seed):
    rng = np.random.default_rng(seed)
    first_rs = [run_single(beta, rng) for _ in range(REPS)]
    appeared = sum(1 for x in first_rs if x is not None)
    crossings = [r for r in range(K, R_MAX+1)
                 if sum(1 for x in first_rs if x is not None and x <= r) / REPS >= 0.5]
    rstar = crossings[0] if crossings else "—"
    print(f"  beta={beta}: r*~{rstar}  "
          f"P(ever)={appeared/REPS:.2f}  "
          f"(median first_r={int(np.median([x for x in first_rs if x is not None])) if appeared else '—'})")
    return first_rs


# ── CDF computation ────────────────────────────────────────────────────────────

def compute_cdf(first_rs, r_grid):
    n = len(first_rs)
    return [sum(1 for x in first_rs if x is not None and x <= r) / n
            for r in r_grid]


# ── Load or compute ────────────────────────────────────────────────────────────

PARAMS_KEY = {"lambda": LAMBDA, "k": K, "betas": BETAS,
              "r_max": R_MAX, "reps": REPS, "start_r": START_R}

blob = None
if os.path.exists(CACHE):
    with open(CACHE) as f:
        blob = json.load(f)
    if blob.get("params_key") != PARAMS_KEY:
        print("Cache differs — recomputing.")
        blob = None
    else:
        print(f"Loading cache from {CACHE}")
        results = {float(row["beta"]): row["first_rs"]
                   for row in blob["results"]}

if blob is None:
    print(f"Direction 1: threshold in r_t  (start_r={START_R}, R_MAX={R_MAX}) ...")
    results = {}
    for i, beta in enumerate(BETAS):
        first_rs = run_beta(beta, seed=42 + i)
        results[beta] = first_rs
    cache_out = {
        "params_key": PARAMS_KEY,
        "results": [{"beta": b, "first_rs": [x if x is not None else -1
                                              for x in results[b]]}
                    for b in BETAS],
    }
    with open(CACHE, "w") as f:
        json.dump(cache_out, f, indent=2)
    print(f"\nCached -> {CACHE}")

# Restore None from -1 in cache
for b in BETAS:
    key = float(b) if float(b) in results else b
    results[b] = [None if x == -1 else x for x in results[key]]


# ── Figure ─────────────────────────────────────────────────────────────────────

r_grid = list(range(K, R_MAX + 1))
fig, ax = plt.subplots(figsize=(6.5, 5))

for beta, color in zip(BETAS, COLORS):
    cdf = compute_cdf(results[beta], r_grid)
    ax.plot(r_grid, cdf, "-", color=color, lw=2.2, label=fr"$\beta={beta}$")

ax.axhline(0.5, color="gray", ls="--", lw=1.2, alpha=0.6)
ax.set_xlabel(r"Rank $r_t$", fontsize=13)
ax.set_ylabel(r"$P(\text{triangle minor appeared by } r_t)$", fontsize=13)
ax.set_title(
    fr"Triangle minor: threshold in $r_t$  "
    fr"($r_0={START_R},\ \gamma=0,\ \lambda={LAMBDA},\ k={K}$)",
    fontsize=12)
ax.set_ylim(-0.04, 1.04)
ax.set_xlim(K - 0.5, R_MAX + 0.5)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.25, lw=0.7)
plt.tight_layout()

path = os.path.join(OUT, "minor_rank_growth.png")
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"  -> {path}")
