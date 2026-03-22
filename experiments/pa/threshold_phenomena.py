"""
Threshold phenomena in the PA binary matroid.

Two sharp transitions as matroid density  d = n/r  increases:

1. RANK SATURATION THRESHOLD
   Rank grows linearly for d < d* (most new columns are independent),
   then hits a sharp knee and saturates at r.
   The knee location d* shifts with β: higher β → earlier saturation
   (hub-and-spoke clusters columns → faster linear-dependence onset).

   Shown as: rank/r  vs  n/r  for β ∈ {0.2, 0.8, 1.5}.

2. MINOR APPEARANCE THRESHOLD
   P(girth ≤ 3)  (first GF(2)-dependent triple) goes from 0 to 1
   through a sigmoid transition around some critical density d**.
   Higher β → lower d** (hubs create collisions sooner).

   Shown as: P(girth ≤ 3)  vs  n/r  for the same β values.

Each panel also shows the uniform-random GF(2) theory curve where known.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys, os
from core.engine import MatroidEngine
from analysis.probe_minors import convert_to_bitsets

# ── Config ─────────────────────────────────────────────────────────────────────
START_R    = 50
K_PARAMS   = 4
C          = 1e-7        # effectively no row growth
BETA_VALUES = [0.7, 1.0, 1.3, 1.6]
COLORS      = ["#2980b9", "#e67e22", "#8e44ad", "#27ae60"]

# Density grid: n/r from 0.1 to 10  (wider to capture all four sigmoids)
N_OVER_R   = np.concatenate([np.linspace(0.1, 1.0, 10),
                              np.linspace(1.1, 4.0, 18),
                              np.linspace(4.5, 10.0, 8)])
N_STEPS_VALS = [max(2, int(d * START_R)) for d in N_OVER_R]

# start_r sweep: fix n≈150, β ∈ same set, vary start_r
# Extended to 260 to capture β=1.6 threshold (expected near r*≈190)
N_FIXED_FOR_R_SWEEP = 150
START_R_VALS = (list(range(10, 40, 4)) +    # flat-1 region (all β)
                list(range(40, 100, 5)) +    # transition zone β=0.7, 1.0
                list(range(100, 175, 8)) +   # transition zone β=1.3
                list(range(175, 325, 12)))   # transition zone β=1.6 + flat-0

ITERS      = 25      # replicates for density sweep
ITERS_R    = 80      # replicates for start_r sweep
SEED       = 42


# ── Incremental rank tracker ───────────────────────────────────────────────────

def incremental_ranks(columns):
    """Return rank after each column added (O(n·r) total, not O(n²))."""
    basis  = []
    ranks  = []
    for col in columns:
        v = 0
        for row in col:
            v |= (1 << row)
        # Gaussian elimination step
        for b in basis:
            v = min(v, v ^ b)
        if v > 0:
            basis.append(v)
            basis.sort(reverse=True)
        ranks.append(len(basis))
    return ranks


def has_girth_le_3(columns):
    """
    True iff the matroid contains a circuit of size ≤ 3.
    - Size 2: duplicate column (a == b)
    - Size 3: GF(2) triangle  (a ⊕ b = c, all distinct non-zero)
    """
    bits   = convert_to_bitsets(columns)
    lookup = set(bits)
    # girth 2: duplicate
    if len(lookup) < len(bits):
        return True
    # girth 3: any pair whose XOR is also present
    unique = [b for b in lookup if b != 0]
    n = len(unique)
    for i in range(n):
        for j in range(i + 1, n):
            if (unique[i] ^ unique[j]) in lookup:
                return True
    return False


# ── Sweep ──────────────────────────────────────────────────────────────────────
# For each (β, n), compute:
#   mean_rank_ratio  = mean(rank / START_R)
#   p_girth_le3      = fraction of runs with girth ≤ 3

print(f"Running threshold sweep  (r={START_R}, k={K_PARAMS}, {ITERS} replicates) …")
print(f"{'β':<6}  {'n/r':<6}  {'rank/r':>7}  {'P(g≤3)':>8}")
print("-" * 35)

# Store: results[beta_idx] = {"rank": array, "p_girth": array, "rank_std": array}
results = []

for bi, beta in enumerate(BETA_VALUES):
    rank_means, rank_stds, p_girths = [], [], []
    for n_steps, d in zip(N_STEPS_VALS, N_OVER_R):
        rr_list, girth_list = [], []
        for seed in range(ITERS):
            np.random.seed(seed * 31 + bi * 1000)
            engine = MatroidEngine(n_steps=n_steps, k_params=K_PARAMS,
                                   C=C, gamma=0, beta=beta, start_r=START_R)
            data = engine.run()
            cols = data['columns']
            if not cols:
                rr_list.append(0.0); girth_list.append(False); continue
            rk = incremental_ranks(cols)[-1]
            rr_list.append(rk / START_R)
            girth_list.append(has_girth_le_3(cols))

        rm = float(np.mean(rr_list))
        rs = float(np.std(rr_list))
        pg = float(np.mean(girth_list))
        rank_means.append(rm); rank_stds.append(rs); p_girths.append(pg)

        if abs(d - round(d)) < 0.05:   # print at integer densities
            print(f"  β={beta:.1f}  d={d:.1f}   rank/r={rm:.3f}   P(g≤3)={pg:.2f}")

    results.append({
        "rank": np.array(rank_means),
        "rank_std": np.array(rank_stds),
        "p_girth": np.array(p_girths),
    })

# ── start_r sweep ─────────────────────────────────────────────────────────────
print("\nRunning start_r sweep …")
print(f"{'β':<6}  {'start_r':<8}  {'d=n/r':>6}  {'P(g≤3)':>8}")
print("-" * 38)

results_r = []
for bi, beta in enumerate(BETA_VALUES):
    p_list = []
    for start_r_val in START_R_VALS:
        hits = []
        d_here = N_FIXED_FOR_R_SWEEP / start_r_val
        for seed in range(ITERS_R):
            np.random.seed(seed * 31 + bi * 1000 + start_r_val)
            engine = MatroidEngine(n_steps=N_FIXED_FOR_R_SWEEP, k_params=K_PARAMS,
                                   C=C, gamma=0, beta=beta, start_r=start_r_val)
            data = engine.run()
            cols = data['columns']
            hits.append(has_girth_le_3(cols) if cols else False)
        pg = float(np.mean(hits))
        p_list.append(pg)
        if start_r_val in [15, 30, 50, 80, 120, 160, 200, 240]:
            print(f"  β={beta:.1f}  r={start_r_val:<6}  d={d_here:.2f}   P(g≤3)={pg:.2f}")
    results_r.append(np.array(p_list))

print("\nDone. Plotting …")

# ── Theoretical reference (uniform random GF(2), k-sparse) ───────────────────
# For a random k-uniform binary matrix (each col has k ones placed uniformly),
# the rank grows approximately as  r(1 - exp(-n·k / r))  only if columns are
# full (k=r).  For sparse k, the threshold is around n·k ≈ r.
# We use the simpler  rank/r ≈ 1 - exp(-d)  as a loose lower reference.
d_ref = np.linspace(0.0, 6.0, 300)
rank_ref = 1.0 - np.exp(-d_ref)   # rough reference (uniform, not sparse)


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

# ── Panel 1: Rank saturation ──────────────────────────────────────────────────
ax1.plot(d_ref, rank_ref, "k:", lw=1.2, alpha=0.5, label="1 − e^{−d}  (dense uniform ref.)")

for bi, (beta, color) in enumerate(zip(BETA_VALUES, COLORS)):
    rm  = results[bi]["rank"]
    rs  = results[bi]["rank_std"]
    ax1.plot(N_OVER_R, rm, "o-", color=color, lw=2, ms=5,
             label=f"β = {beta}")
    ax1.fill_between(N_OVER_R, rm - rs, rm + rs, alpha=0.15, color=color)

# Mark the knee region (rank/r = 0.5 and 0.9)
ax1.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
ax1.axhline(0.5, color="gray", ls=":",  lw=0.8, alpha=0.4, label="rank/r = 0.5")

ax1.set_xlabel("Matroid density  d = n / r", fontsize=12)
ax1.set_ylabel("Normalised rank  rank / r", fontsize=12)
ax1.set_title("Rank saturation threshold\n"
              "Sharp knee: from linear growth to saturation",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.set_xlim(0, 6); ax1.set_ylim(0, 1.05)
ax1.grid(alpha=0.3)

# ── Panel 2: Minor appearance threshold ──────────────────────────────────────
for bi, (beta, color) in enumerate(zip(BETA_VALUES, COLORS)):
    pg = results[bi]["p_girth"]
    ax2.plot(N_OVER_R, pg, "s-", color=color, lw=2, ms=5,
             label=f"β = {beta}")
    # Find approximate threshold d** where P crosses 0.5
    idx = np.searchsorted(pg, 0.5)
    if 0 < idx < len(N_OVER_R):
        d_thresh = N_OVER_R[idx]
        ax2.axvline(d_thresh, color=color, ls=":", lw=1.2, alpha=0.6)
        ax2.annotate(f"d*≈{d_thresh:.1f}",
                     xy=(d_thresh, 0.5),
                     xytext=(5, -15), textcoords="offset points",
                     fontsize=8.5, color=color)

ax2.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5, label="P = 0.5")
ax2.set_xlabel("Matroid density  d = n / r", fontsize=12)
ax2.set_ylabel("P(girth ≤ 3)  =  P(triangle minor)", fontsize=12)
ax2.set_title("Minor appearance threshold\n"
              "First GF(2)-dependent triple: sigmoid transition",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xlim(0, 10); ax2.set_ylim(-0.03, 1.05)
ax2.grid(alpha=0.3)

# ── Panel 3: start_r sweep ────────────────────────────────────────────────────
def logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(k * (x - x0)))

x_smooth = np.linspace(min(START_R_VALS), max(START_R_VALS), 400)

for bi, (beta, color) in enumerate(zip(BETA_VALUES, COLORS)):
    pg = results_r[bi]
    xr = np.array(START_R_VALS, dtype=float)
    # Raw data as faint dots
    ax3.plot(xr, pg, "o", color=color, ms=4, alpha=0.35)
    # Logistic fit as solid line
    try:
        # Initial guess: x0 = midpoint, k = slope ~ 0.05
        x0_guess = xr[np.argmin(np.abs(pg - 0.5))]
        popt, _ = curve_fit(logistic, xr, pg, p0=[x0_guess, 0.05],
                            bounds=([xr[0], 0.001], [xr[-1], 1.0]),
                            maxfev=5000)
        y_fit = logistic(x_smooth, *popt)
        ax3.plot(x_smooth, y_fit, "-", color=color, lw=2.2,
                 label=f"β = {beta}  (r*≈{popt[0]:.0f})")
        ax3.axvline(popt[0], color=color, ls=":", lw=1.1, alpha=0.55)
        ax3.annotate(f"r*≈{popt[0]:.0f}",
                     xy=(popt[0], 0.5), xytext=(4, 6),
                     textcoords="offset points", fontsize=8.5, color=color)
    except Exception:
        ax3.plot(xr, pg, "-", color=color, lw=2, label=f"β = {beta}")

ax3.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5, label="P = 0.5")
ax3.set_xlabel("Initial row count  r₀", fontsize=12)
ax3.set_ylabel("P(girth ≤ 3)  =  P(triangle minor)", fontsize=12)
ax3.set_title(f"start_r threshold  (n = {N_FIXED_FOR_R_SWEEP} fixed)\n"
              "More rows → sparser matroid → minor disappears",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.set_ylim(-0.03, 1.05)
ax3.grid(alpha=0.3)

fig.suptitle(f"Threshold phenomena in PA binary matroids"
             f"  (k = {K_PARAMS},  {ITERS} replicates/point)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "threshold_phenomena.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
