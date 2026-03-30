"""
Exact Tutte polynomial via deletion-contraction for PA binary matroids.

NOTE ON FEASIBILITY
-------------------
The user requested n_steps=150 (→ n = start_r + n_steps = 155 elements).
Deletion-contraction is O(2^n) worst-case; even with memoisation n=155 is
astronomically infeasible. The practical ceiling is n ≈ 20–25.
This script therefore uses n_steps=15 (→ n=20).

Parameters: γ=0, k=4, start_r=5, n_steps=15, 5 replicates.
            β ∈ {0.5, 1.0, 1.5, 2.0},  λ ∈ {0.05, 0.2}.

Tutte polynomial T(M; x, y) is computed exactly.
The matroid is represented as a tuple of GF(2) column bitsets (integers).

Evaluations reported:
  T(1,1) — number of bases
  T(2,1) — number of independent sets
  T(1,2) — number of spanning sets

Saves: tutte_evaluations.png
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import lru_cache
import os

from core.engine import MatroidEngine

BETA_VALUES = [0.5, 1.0, 1.5, 2.0]
LAMBDAS     = [0.05, 0.2]
GAMMA       = 0.0
K           = 4
N_STEPS     = 15         # n = start_r + N_STEPS = 20
START_R     = 5
REPS        = 5
COLORS_LAM  = {0.05: "#2980b9", 0.2: "#e74c3c"}
OUT         = os.path.dirname(__file__)


# ── GF(2) utilities ──────────────────────────────────────────────────────────

def gf2_rank(cols):
    """Rank of binary matroid given as tuple of integer column bitsets."""
    basis = []
    for c in cols:
        v = c
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
    return len(basis)


# ── Deletion-contraction Tutte polynomial ────────────────────────────────────

@lru_cache(maxsize=None)
def _tutte(cols):
    """
    Return Tutte polynomial as a dict {(i, j): coefficient}.

    cols : sorted tuple of non-negative int bitsets representing the GF(2)
           column vectors of the matroid (duplicates allowed).
    """
    if not cols:
        return {(0, 0): 1}

    r = gf2_rank(cols)

    # Pick the first element
    e = cols[0]
    rest = cols[1:]          # deletion M\e

    if e == 0:
        # Loop → T(M) = y · T(M\e)
        sub = _tutte(rest)
        return {(i, j + 1): v for (i, j), v in sub.items()}

    # Check coloop: rank drops on deletion
    if gf2_rank(rest) < r:
        # Coloop → T(M) = x · T(M\e)
        sub = _tutte(rest)
        return {(i + 1, j): v for (i, j), v in sub.items()}

    # Ordinary element: T(M) = T(M\e) + T(M/e)
    t_del = _tutte(rest)

    # Contraction M/e: pivot on highest set bit b of e, then remove row b
    b = e.bit_length() - 1
    contracted = [c ^ e if (c >> b) & 1 else c for c in rest]
    lower_mask = (1 << b) - 1
    con_cols = tuple(sorted(
        ((c >> (b + 1)) << b) | (c & lower_mask)
        for c in contracted
    ))
    t_con = _tutte(con_cols)

    result = dict(t_del)
    for k, v in t_con.items():
        result[k] = result.get(k, 0) + v
    return result


def tutte_poly(r, attachment_supports):
    """Build column-bitset representation and compute Tutte polynomial."""
    _tutte.cache_clear()
    cols = []
    for i in range(r):               # identity block
        cols.append(1 << i)
    for support in attachment_supports:  # attachment block
        cols.append(sum(1 << row for row in support))
    return _tutte(tuple(sorted(cols)))


def eval_tutte(poly, x, y):
    return sum(v * (x ** i) * (y ** j) for (i, j), v in poly.items())


# ── Run experiments ───────────────────────────────────────────────────────────

# results[lam][beta] = list of dicts (one per rep)
results = {lam: {beta: [] for beta in BETA_VALUES} for lam in LAMBDAS}

print(f"\n{'λ':>5}  {'β':>5}  {'r':>4}  {'n':>4}  "
      f"{'T(1,1)':>10}  {'T(2,1)':>12}  {'T(1,2)':>12}")
print("─" * 65)

for lam in LAMBDAS:
    for beta in BETA_VALUES:
        t11_l, t21_l, t12_l = [], [], []
        poly_l = []
        for _ in range(REPS):
            eng  = MatroidEngine(n_steps=N_STEPS, k_params=K, C=lam,
                                 gamma=GAMMA, beta=beta, start_r=START_R)
            data = eng.run()
            r    = data["r"]
            poly = tutte_poly(r, data["attachment_supports"])
            t11  = eval_tutte(poly, 1, 1)
            t21  = eval_tutte(poly, 2, 1)
            t12  = eval_tutte(poly, 1, 2)
            t11_l.append(t11); t21_l.append(t21); t12_l.append(t12)
            poly_l.append(poly)
            results[lam][beta].append(
                dict(r=r, n=data["n"], poly=poly, t11=t11, t21=t21, t12=t12))

        print(f"{lam:>5.2f}  {beta:>5.1f}  "
              f"{np.mean([d['r'] for d in results[lam][beta]]):>4.1f}  "
              f"{np.mean([d['n'] for d in results[lam][beta]]):>4.1f}  "
              f"{np.mean(t11_l):>10.1f}  {np.mean(t21_l):>12.1f}  "
              f"{np.mean(t12_l):>12.1f}")

    print()

# ── Print one example coefficient vector per (λ, β) ─────────────────────────

print("\nSample coefficient vectors T(M; x, y) = Σ t_{ij} x^i y^j")
print("(single run, rep 0):\n")
for lam in LAMBDAS:
    for beta in BETA_VALUES:
        poly = results[lam][beta][0]["poly"]
        r    = results[lam][beta][0]["r"]
        n    = results[lam][beta][0]["n"]
        max_i = max(i for i, j in poly)
        max_j = max(j for i, j in poly)
        coeff_str = "  ".join(
            f"t_{i}{j}={poly.get((i,j),0)}"
            for i in range(max_i + 1)
            for j in range(max_j + 1)
            if poly.get((i, j), 0) != 0
        )
        print(f"  λ={lam}  β={beta}  r={r}  n={n}:  {coeff_str}")
    print()

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles  = [r"$T(M;1,1)$ — \#bases",
           r"$T(M;2,1)$ — \#independent sets",
           r"$T(M;1,2)$ — \#spanning sets"]
keys    = ["t11", "t21", "t12"]

for ax, title, key in zip(axes, titles, keys):
    for lam in LAMBDAS:
        color = COLORS_LAM[lam]
        means = [np.mean([d[key] for d in results[lam][beta]])
                 for beta in BETA_VALUES]
        stds  = [np.std([d[key] for d in results[lam][beta]])
                 for beta in BETA_VALUES]
        ax.errorbar(BETA_VALUES, means, yerr=stds, fmt="o-", color=color,
                    lw=2, markersize=7, capsize=4,
                    label=fr"$\lambda={lam}$")
    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Count (mean ± std over replicates)", fontsize=11)
fig.suptitle(
    fr"Tutte polynomial evaluations: $\gamma=0$, $k=4$, "
    fr"$n_{{\rm steps}}={N_STEPS}$, $\mathrm{{start\_r}}={START_R}$",
    fontsize=13)
plt.tight_layout()
path = os.path.join(OUT, "tutte_evaluations.png")
fig.savefig(path, dpi=150)
print(f"  → {path}")
