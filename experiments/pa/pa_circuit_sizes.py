"""
Circuit size distribution: N_s = #{circuits of size exactly s} for
s ∈ {3, 5, 8, 10, 12, 15} across 50 replicates.

γ=0, λ=0.05, k_t=floor(0.3·r_t), β=0.5, n_steps=2000, start_r=10.

CIRCUITS COUNTED  (exact for the two lowest-order families)
------------------------------------------------------------
For the PA matroid [I_r | A] (basis = standard rows, non-basis = attached columns):

  1. Fundamental circuit of column c_j with support S_j:
       C_j = {c_j} ∪ S_j,  size = |S_j| + 1 = k_j + 1.

  2. Pairwise circuit of (c_j1, c_j2) whenever S_j1 ∩ S_j2 ≠ ∅:
       P_{j1,j2} = {c_j1, c_j2} ∪ (S_j1 △ S_j2),
       size = 2 + |S_j1 △ S_j2| = 2 + k_j1 + k_j2 - 2|S_j1 ∩ S_j2|.
     (The non-empty-intersection condition ensures no fundamental circuit
      is a proper subset, so P_{j1,j2} is minimal.)

Parity note:
  Same-k pairs (k_j1 = k_j2) always have |S_j1 △ S_j2| = 2(k-inter) = even,
  so same-k pairwise circuits have EVEN size.  Odd-size pairwise circuits
  (s=3, 5, 15) can only come from CROSS-k pairs.

N_s implementation:
  Build binary matrix B (n_att × r), Gram matrix G = B Bᵀ (integer overlaps),
  pairwise symmetric-difference size D = k_j1 + k_j2 − 2·G[j1,j2].
  Count upper-triangle entries with D = s−2 and G > 0, plus fundamentals.

Saves: circuit_sizes_hist.png, circuit_sizes_stats.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

ALPHA   = 0.3
BETA    = 0.5
LAMBDA  = 0.05
GAMMA   = 0.0
N_STEPS = 2000
START_R = 10
REPS    = 50
SIZES   = [3, 5, 8, 10, 12, 15]
OUT     = os.path.dirname(__file__)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_and_collect(n_steps, alpha, C_lam, gamma, beta, start_r, seed):
    """
    Returns (supports, k_vals, r_final).
    supports[j]: numpy array of row indices for non-basis column j.
    k_vals[j]:   int, support size of column j.
    """
    rng       = np.random.default_rng(seed)
    curr_r    = start_r
    row_usage = np.ones(start_r, dtype=np.float64)
    supports  = []
    k_list    = []
    for t in range(1, n_steps + 1):
        p_row = min(1.0, C_lam * t ** (-gamma))
        if rng.random() < p_row:
            curr_r += 1
            row_usage = np.append(row_usage, 1.0)
        else:
            k   = max(1, int(np.floor(alpha * curr_r)))
            k   = min(k, curr_r)
            w   = row_usage ** beta;  w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            supports.append(sel)
            k_list.append(k)
            for idx in sel:
                row_usage[int(idx)] += 1
    return supports, np.array(k_list, dtype=np.int32), curr_r


# ── Circuit counting ──────────────────────────────────────────────────────────

def count_by_size(supports, k_vals, r_final, target_sizes):
    """
    Returns dict {s: N_s} for s in target_sizes.
    Counts fundamental circuits + pairwise circuits (overlap condition).
    """
    n_att  = len(supports)
    result = {s: 0 for s in target_sizes}

    # 1. Fundamental circuits: size = k_j + 1
    for s in target_sizes:
        result[s] += int(np.sum(k_vals == s - 1))

    if n_att == 0:
        return result

    # 2. Binary matrix B: shape (n_att, r_final)
    B = np.zeros((n_att, r_final), dtype=np.float32)
    for j, sel in enumerate(supports):
        B[j, sel] = 1.0

    # Gram matrix: G[j1,j2] = |S_j1 ∩ S_j2|  (exact via float32, values ≤ 50)
    G = (B @ B.T).round().astype(np.int32)

    # Pairwise △-size: D[j1,j2] = k_j1 + k_j2 - 2·G[j1,j2]
    k32 = k_vals.astype(np.int32)
    D   = k32[:, None] + k32[None, :] - 2 * G   # (n_att, n_att), non-negative

    # Mask: upper triangle, non-empty intersection
    tri     = np.triu(np.ones((n_att, n_att), dtype=bool), k=1)
    overlap = (G > 0) & tri

    # 3. Pairwise circuits for each target size
    for s in target_sizes:
        result[s] += int(np.sum((D == s - 2) & overlap))

    return result


# ── Collect across replicates ─────────────────────────────────────────────────

print("Running 50 replicates …")
all_counts = {s: [] for s in SIZES}

for rep in range(REPS):
    seed     = 99000 + rep
    sup, kv, r_fin = run_and_collect(
        N_STEPS, ALPHA, LAMBDA, GAMMA, BETA, START_R, seed)
    counts   = count_by_size(sup, kv, r_fin, SIZES)
    for s in SIZES:
        all_counts[s].append(counts[s])
    if (rep + 1) % 10 == 0:
        summary = "  ".join(f"N_{s}={counts[s]}" for s in SIZES)
        print(f"  rep {rep+1:2d}/50  r={r_fin}  {summary}")

count_arr = {s: np.array(all_counts[s]) for s in SIZES}


# ── Statistics table ──────────────────────────────────────────────────────────

print(f"\n{'s':>4}  {'mean':>12}  {'var':>14}  {'var/mean':>10}  "
      f"{'min':>8}  {'max':>8}  {'W p-val':>10}")
print("─" * 72)
for s in SIZES:
    arr  = count_arr[s]
    m    = float(arr.mean())
    v    = float(arr.var(ddof=1))
    disp = v / m if m > 0 else float("nan")
    # Shapiro-Wilk normality test
    _, p_sw = stats.shapiro(arr) if len(arr) >= 3 else (0, float("nan"))
    print(f"{s:>4}  {m:>12.2f}  {v:>14.2f}  {disp:>10.4f}  "
          f"{arr.min():>8d}  {arr.max():>8d}  {p_sw:>10.4f}")

print()
print("var/mean  > 1 → overdispersed (super-Poisson)")
print("var/mean ≈ 1 → consistent with Poisson")
print("var/mean  < 1 → underdispersed")
print("Shapiro-Wilk p > 0.05 → consistent with normality")


# ── Plot 1: histograms ────────────────────────────────────────────────────────

fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9))
axes1 = axes1.ravel()

COLORS_HIST = {"hist": "#2c7bb6", "normal": "#d7191c", "poisson": "#1a9641"}

for idx, s in enumerate(SIZES):
    ax  = axes1[idx]
    arr = count_arr[s]
    m   = float(arr.mean())
    v   = float(arr.var(ddof=1))
    std = float(np.sqrt(v))
    disp = v / m if m > 0 else float("nan")

    # Adaptive bins
    n_bins = max(10, min(30, int(arr.max() - arr.min() + 1)))
    cnts, bins, _ = ax.hist(arr, bins=n_bins, density=False,
                             color=COLORS_HIST["hist"], edgecolor="white",
                             alpha=0.75, label=f"50 replicates")
    bw = bins[1] - bins[0]

    # Normal overlay
    xs = np.linspace(max(0, m - 4.5*std), m + 4.5*std, 300)
    ax.plot(xs, stats.norm.pdf(xs, m, std) * REPS * bw, "-",
            color=COLORS_HIST["normal"], lw=2.2, label="Normal$(\\mu,\\sigma^2)$")

    # Poisson overlay (only when mean is tractable)
    if 0 < m < 500:
        k_lo = max(0, int(m - 4.5 * max(1, np.sqrt(m))))
        k_hi = int(m + 4.5 * max(1, np.sqrt(m))) + 1
        kk   = np.arange(k_lo, k_hi)
        pois = stats.poisson.pmf(kk, m) * REPS
        ax.bar(kk, pois, width=bw, color=COLORS_HIST["poisson"],
               alpha=0.35, label=f"Poisson($\\lambda$={m:.1f})")

    # Annotation
    ax.set_title(
        fr"$s = {s}$" + f"\n"
        fr"$\mu={m:.1f}$  $\sigma^2={v:.1f}$  "
        fr"$\sigma^2/\mu={disp:.3f}$",
        fontsize=10)
    ax.set_xlabel(fr"$N_{{{s}}}$", fontsize=11)
    ax.set_ylabel("Count (replicates)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

fig1.suptitle(
    fr"Distribution of $N_s$ (circuits of size $s$): $\gamma=0$, $\lambda=0.05$, "
    fr"$\alpha=0.3$, $\beta=0.5$, $k_t=\lfloor 0.3\,r_t\rfloor$, "
    fr"$n_{{\rm steps}}=2000$, 50 reps",
    fontsize=12)
plt.tight_layout()
path1 = os.path.join(OUT, "circuit_sizes_hist.png")
fig1.savefig(path1, dpi=150)
print(f"\n  → {path1}")


# ── Plot 2: mean and variance vs s + Poisson check ───────────────────────────

means  = np.array([float(count_arr[s].mean())         for s in SIZES])
vars_  = np.array([float(count_arr[s].var(ddof=1))    for s in SIZES])
stds   = np.sqrt(vars_)
disps  = vars_ / np.where(means > 0, means, np.nan)

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

# Panel a: mean ± std and variance vs s (log scale)
ax = axes2[0]
s_arr = np.array(SIZES)
ax.semilogy(s_arr, means, "bo-", lw=2, ms=9, label="mean $N_s$", zorder=4)
ax.fill_between(s_arr, np.maximum(1, means - stds), means + stds,
                color="blue", alpha=0.15, label="±1 std")
ax.semilogy(s_arr, vars_, "rs--", lw=2, ms=9, label="var $N_s$", zorder=4)
# Dispersion ratio on secondary axis
ax2 = ax.twinx()
ax2.plot(s_arr, disps, "g^:", lw=1.5, ms=7, alpha=0.8, label="var/mean")
ax2.axhline(1.0, color="green", lw=1, ls=":", alpha=0.5)
ax2.set_ylabel("var / mean  (dispersion)", fontsize=11, color="green")
ax2.tick_params(axis="y", labelcolor="green")
ax.set_xlabel("Circuit size $s$", fontsize=12)
ax.set_ylabel("Value (log scale)", fontsize=12)
ax.set_title("Mean, variance and dispersion of $N_s$ vs $s$", fontsize=12)
ax.set_xticks(SIZES)
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="lower left")
ax.grid(True, alpha=0.25, which="both")

# Panel b: variance vs mean scatter (Poisson diagnostic)
ax = axes2[1]
sc = ax.scatter(means, vars_, c=SIZES, cmap="plasma", s=130, zorder=5,
                edgecolors="k", linewidths=0.8)
for i, s in enumerate(SIZES):
    ax.annotate(f"$s={s}$", (means[i], vars_[i]),
                textcoords="offset points", xytext=(7, 3), fontsize=10)
cb = fig2.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label("Circuit size $s$", fontsize=10)

# Reference lines
lo, hi = 0, max(means.max(), vars_.max()) * 1.3
ax.plot([lo, hi], [lo, hi],   "k--", lw=1.8, alpha=0.7, label="Poisson (var=mean)")
ax.plot([lo, hi], [0, hi**2/hi if hi > 0 else 1], "k:", lw=1.2, alpha=0.4)  # slope-2 guide

ax.set_xlabel("mean $N_s$", fontsize=12)
ax.set_ylabel("var $N_s$", fontsize=12)
ax.set_title("Poisson check: var vs mean", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)

fig2.suptitle(
    fr"$N_s$ statistics across 50 replicates: "
    fr"$\gamma=0$, $\lambda=0.05$, $\alpha=0.3$, $\beta=0.5$",
    fontsize=12)
plt.tight_layout()
path2 = os.path.join(OUT, "circuit_sizes_stats.png")
fig2.savefig(path2, dpi=150)
print(f"  → {path2}")
print("""
Interpretation guide:
  var/mean ≈ 1  → Poisson (independent rare events)
  var/mean >> 1 → super-Poisson / overdispersed (positive correlations between circuits)
  var/mean << 1 → underdispersed (rare, mutually exclusive configurations)
  Shapiro-Wilk p > 0.05 → distribution consistent with Normal (central-limit regime)

  Odd s (3,5,15): pairwise circuits from cross-k pairs only → expect small N_s, Poisson?
  Even s (8,10,12): pairwise circuits from same-k AND cross-k pairs → large N_s, Normal?
""")
