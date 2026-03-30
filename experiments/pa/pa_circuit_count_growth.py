"""
Circuit count growth: |C(M_t)| vs t on a log-log scale.

γ=0, λ=0.05, k_t=floor(α·r_t), n_steps=2000, start_r=10, 5 replicates.
α ∈ {0.2, 0.3, 0.4, 0.5}, β ∈ {0.5, 1.0}.

CIRCUIT COUNTING (exact lower bound)
-------------------------------------
For [I_r | A] with basis B = {e_1,...,e_r} and non-basis columns {c_j}
with support S_j ⊆ [r]:

  1. Fundamental circuit of c_j:
       C_j = {c_j} ∪ S_j,  size = k_t + 1.
     One per non-basis column → |C_fund| = n_att.  Grows linearly in t.

  2. Pairwise circuit of (c_j1, c_j2) whenever S_j1 ∩ S_j2 ≠ ∅:
       - S_j1 = S_j2 (parallel pair):  circuit = {c_j1, c_j2},  size 2.
       - S_j1 ≠ S_j2, S_j1 ∩ S_j2 ≠ ∅: circuit = {c_j1,c_j2}∪(S_j1 △ S_j2).
     In both cases the set is a minimal dependent set (the intersection
     condition ensures no fundamental circuit is a subset).
     Count: #{pairs with non-empty intersection} ~ O(n_att²) = O(t²).

Higher-order circuits (3+ non-basis elements forming a GF(2) dependency
without a smaller circuit as subset) are not counted.

Implemented via incremental bitmask overlap detection (2×uint64 per column,
vectorised with numpy).  Each new column costs O(n_att) per step.

Prints |C|/n_att at t = 500, 1000, 2000.
Saves: circuit_count_growth.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

LAMBDA  = 0.05
GAMMA   = 0.0
ALPHAS  = [0.2, 0.3, 0.4, 0.5]
BETAS   = [0.5, 1.0]
N_STEPS = 2000
START_R = 10
REPS    = 5
COLORS  = {0.5: "#2980b9", 1.0: "#e74c3c"}
OUT     = os.path.dirname(__file__)
MASK64  = (1 << 64) - 1
T_PRINT = [500, 1000, 2000]


# ── Simulation + incremental circuit count ────────────────────────────────────

def run_and_count(n_steps, alpha, C_lam, gamma, beta, start_r, seed):
    """
    Returns arrays (t_arr, natt_arr, r_arr, circ_arr) indexed by attachment
    events (one entry per non-basis column added).

    circ_arr[i] = n_att + n_pairwise  (lower bound on |C(M_t)|).
    """
    rng       = np.random.default_rng(seed)
    curr_r    = start_r
    row_usage = np.ones(start_r, dtype=np.float64)

    max_att = n_steps + 1
    low_m   = np.zeros(max_att, dtype=np.uint64)   # low 64 bits of support mask
    high_m  = np.zeros(max_att, dtype=np.uint64)   # high 64 bits (rows 64–127)

    n_att      = 0
    n_pairwise = 0  # cumulative pairwise circuits (overlap condition)

    t_arr    = []
    natt_arr = []
    r_arr    = []
    circ_arr = []

    for t in range(1, n_steps + 1):
        p_row = min(1.0, C_lam * (t ** (-gamma)))
        if rng.random() < p_row:
            curr_r += 1
            row_usage = np.append(row_usage, 1.0)
        else:
            k   = max(1, int(np.floor(alpha * curr_r)))
            k   = min(k, curr_r)
            w   = row_usage ** beta;  w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)

            # Build support bitmask (split into 2 × uint64)
            mask_int = 0
            for i in sel:
                mask_int |= (1 << int(i))
            low  = np.uint64(mask_int & MASK64)
            high = np.uint64((mask_int >> 64) & MASK64)

            # Count existing columns whose support intersects the new one
            if n_att > 0:
                overlap = ((low_m[:n_att] & low) | (high_m[:n_att] & high)) != np.uint64(0)
                n_pairwise += int(overlap.sum())

            low_m[n_att]  = low
            high_m[n_att] = high
            n_att += 1

            for idx in sel:
                row_usage[int(idx)] += 1

            t_arr.append(t)
            natt_arr.append(n_att)
            r_arr.append(curr_r)
            circ_arr.append(n_att + n_pairwise)

    return (np.array(t_arr), np.array(natt_arr),
            np.array(r_arr),  np.array(circ_arr))


# ── Collect data ──────────────────────────────────────────────────────────────

results = {}
print("Running simulations …")
for alpha in ALPHAS:
    for beta in BETAS:
        reps = []
        for rep in range(REPS):
            seed = int(alpha * 1000) * 10000 + int(beta * 100) * 1000 + rep
            data = run_and_count(N_STEPS, alpha, LAMBDA, GAMMA,
                                 beta, START_R, seed)
            reps.append(data)
            print(f"  α={alpha}  β={beta}  rep={rep}  "
                  f"|C|_final={data[3][-1]:,d}  n_att={data[1][-1]}")
        results[(alpha, beta)] = reps


# ── Print ratio table ─────────────────────────────────────────────────────────

print(f"\n{'α':>5}  {'β':>5}  {'t':>6}  "
      f"{'|C|/n_att':>12}  {'|C|':>12}  {'n_att':>8}  {'r':>6}")
print("─" * 62)

for alpha in ALPHAS:
    for beta in BETAS:
        for t_check in T_PRINT:
            ratios, circs, natts, rs = [], [], [], []
            for (t_arr, natt_arr, r_arr, circ_arr) in results[(alpha, beta)]:
                idx = min(np.searchsorted(t_arr, t_check), len(t_arr) - 1)
                ratios.append(circ_arr[idx] / natt_arr[idx])
                circs.append(circ_arr[idx])
                natts.append(natt_arr[idx])
                rs.append(r_arr[idx])
            print(f"{alpha:>5.1f}  {beta:>5.1f}  {t_check:>6d}  "
                  f"{np.mean(ratios):>12.1f}  "
                  f"{np.mean(circs):>12.0f}  "
                  f"{np.mean(natts):>8.0f}  "
                  f"{np.mean(rs):>6.1f}")
    print()


# ── Fit log-log slope for each (α, β) ────────────────────────────────────────

print(f"\n{'α':>5}  {'β':>5}  {'slope (log|C|/log t)':>22}  {'R²':>6}")
print("─" * 42)
slopes = {}
for alpha in ALPHAS:
    for beta in BETAS:
        all_slopes = []
        for (t_arr, natt_arr, r_arr, circ_arr) in results[(alpha, beta)]:
            # Fit on t > 100 to skip transient
            mask = t_arr > 100
            if mask.sum() > 10:
                sl, _, _, _, _ = linregress(np.log(t_arr[mask]),
                                            np.log(circ_arr[mask]))
                all_slopes.append(sl)
        s = np.mean(all_slopes)
        slopes[(alpha, beta)] = s
        print(f"{alpha:>5.1f}  {beta:>5.1f}  {s:>22.3f}")
    print()


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=False)
axes = axes.ravel()

t_guide = np.array([100.0, 2000.0])

for ax_idx, alpha in enumerate(ALPHAS):
    ax = axes[ax_idx]
    ref_set = False

    for beta in BETAS:
        color = COLORS[beta]
        all_t = [d[0] for d in results[(alpha, beta)]]
        all_c = [d[3] for d in results[(alpha, beta)]]

        # Thin replicate lines
        for t_arr, circ_arr in zip(all_t, all_c):
            ax.loglog(t_arr, circ_arr, "-", color=color, lw=0.7, alpha=0.3)

        # Mean on common fine grid
        t_grid = np.exp(np.linspace(np.log(max(10, all_t[0][0])),
                                    np.log(all_t[0][-1]), 300))
        mean_c = np.zeros(300)
        for t_arr, circ_arr in zip(all_t, all_c):
            mean_c += np.interp(t_grid, t_arr, circ_arr.astype(float))
        mean_c /= REPS
        sl = slopes[(alpha, beta)]
        ax.loglog(t_grid, mean_c, "-", color=color, lw=2,
                  label=fr"$\beta={beta}$ (slope$\approx{sl:.2f}$)")

        # Scale reference lines once (from β=1.0 at t=2000)
        if not ref_set and beta == 1.0:
            c_ref = float(np.interp(2000, t_grid, mean_c))
            ax.loglog(t_guide, c_ref * (t_guide / 2000) ** 2,
                      "k--", lw=1.5, alpha=0.55, label=r"$\propto t^2$")
            ax.loglog(t_guide, c_ref * (t_guide / 2000) ** 1,
                      "k:",  lw=1.5, alpha=0.55, label=r"$\propto t$")
            ref_set = True

    ax.set_xlabel("Step $t$", fontsize=11)
    ax.set_ylabel(r"$|\mathcal{C}(M_t)|$ (lower bound)", fontsize=11)
    ax.set_title(fr"$\alpha = {alpha}$", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)

fig.suptitle(
    fr"Circuit count growth: $\gamma=0$, $\lambda={LAMBDA}$, "
    fr"$k_t=\lfloor\alpha r_t\rfloor$, $n_{{\rm steps}}={N_STEPS}$  "
    r"[lower bound: fundamental + pairwise circuits]",
    fontsize=12)
plt.tight_layout()
path = os.path.join(OUT, "circuit_count_growth.png")
fig.savefig(path, dpi=150)
print(f"\n  → {path}")
print("""
Key:
  slope ≈ 2.0  →  |C(M_t)| ~ t²  (quadratic, pairwise circuits dominate)
  slope ≈ 1.0  →  |C(M_t)| ~ t   (linear, only fundamental circuits)
  ratio |C|/n_att ~ t             →  circuits per attachment event grows linearly

  Higher-order circuits (≥3 non-basis elements) excluded from count.
  Pairwise circuits {c_j1,c_j2}∪(S_j1 △ S_j2) exist for all pairs with
  non-empty support intersection; fraction overlapping ≈ 1-(1-α)^k → ~84%
  for α=0.3, k=18.
""")
