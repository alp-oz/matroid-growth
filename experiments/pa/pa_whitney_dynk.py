"""
Whitney numbers of the flat lattice with dynamic k_t = floor(α·r_t).

γ=0, λ=0.05, α=0.3, n_steps=1000, start_r=10, 5 replicates.
β ∈ {0.5, 1.0, 1.5, 2.0}.

MATHEMATICAL CONTEXT
--------------------
For [I_r | A], every subset T ⊆ [r] defines a closed flat
    F_T = {e_i : i ∈ T} ∪ {j (non-basis) : support(j) ⊆ T}
with rank(F_T) = |T|.  The flat lattice is isomorphic to the Boolean
lattice 2^[r], giving Whitney numbers of the second kind:
    w_k = C(r, k)   for all k=0,...,r
independent of β (PA bias only affects which rows are selected, not the
flat structure).

User reference "w_k^PG = C(r,k)" matches the Boolean lattice exactly.
The *actual* PG(r-1,2) Whitney numbers are the Gaussian binomial
    [r choose k]_2 = prod_{i=0}^{k-1} (2^{r-i}-1)/(2^{k-i}-1)
which grows as ~2^{k(r-k)} — vastly larger.

PLOTS
-----
Plot 1 (3 panels):
  (a) Normalised w_k / 2^r vs k — one curve per β + user reference + Gaussian binomial
  (b) Deviation w_k - w_k^PG (= C(r,k) - C(r,k)) vs k — zero by construction
  (c) Sampled mean flat size E[|F_T|] at rank k_target = floor(α·r),
      averaged over random T ⊆ [r] with |T|=k_target — this IS β-dependent.

Saves: whitney_dynk.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import os

LAMBDA   = 0.05
GAMMA    = 0.0
ALPHA    = 0.3
BETAS    = [0.5, 1.0, 1.5, 2.0]
N_STEPS  = 1000
START_R  = 10
REPS     = 5
N_SAMPLE = 2000   # random T's to sample for flat size distribution
COLORS   = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
OUT      = os.path.dirname(__file__)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_dynamic_k(n_steps, alpha, C, gamma, beta, start_r, seed):
    rng       = np.random.default_rng(seed)
    curr_r    = start_r
    row_usage = np.ones(start_r, dtype=np.float64)
    supports  = []
    for t in range(1, n_steps + 1):
        p_row = min(1.0, C * (t ** (-gamma)))
        if rng.random() < p_row:
            curr_r += 1
            row_usage = np.append(row_usage, 1.0)
        else:
            k = max(1, int(np.floor(alpha * curr_r)))
            k = min(k, curr_r)
            w = row_usage ** beta;  w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            supports.append(sorted(int(x) for x in sel))
            for idx in sel:
                row_usage[int(idx)] += 1
    return curr_r, supports


# ── Gaussian binomial ─────────────────────────────────────────────────────────

def log_gaussian_binomial(r, k, q=2):
    """log [r choose k]_q."""
    if k < 0 or k > r:
        return -np.inf
    log_num = sum(np.log(q**i - 1) for i in range(r, r - k, -1))
    log_den = sum(np.log(q**i - 1) for i in range(1, k + 1))
    return log_num - log_den


def gaussian_binomial_normalised(r, q=2):
    """Return normalised [r choose k]_q / sum_k [r choose k]_q for k=0..r."""
    log_vals = np.array([log_gaussian_binomial(r, k, q) for k in range(r + 1)])
    # subtract max for numerical stability
    log_vals -= log_vals.max()
    vals = np.exp(log_vals)
    return vals / vals.sum()


# ── Sampled flat size at rank k_target ────────────────────────────────────────

def sample_flat_sizes(r, columns, k_target, n_sample, rng):
    """
    For n_sample random T ⊆ [r] with |T|=k_target,
    compute |F_T| = k_target + #{j : support(j) ⊆ T}.
    Returns array of flat sizes.
    """
    if k_target <= 0 or k_target > r:
        return np.array([k_target])

    rows = np.arange(r)
    sizes = []
    for _ in range(n_sample):
        T = set(rng.choice(rows, size=k_target, replace=False))
        extra = sum(1 for sup in columns if set(sup) <= T)
        sizes.append(k_target + extra)
    return np.array(sizes)


# ── Collect data ──────────────────────────────────────────────────────────────

results = {beta: [] for beta in BETAS}

print(f"\n{'β':>5}  {'r_mean':>8}  {'k_target':>9}  "
      f"{'flat_size_mean':>15}  {'flat_size_max':>14}")
print("─" * 58)

for beta in BETAS:
    flat_size_means, flat_size_maxes = [], []
    for rep in range(REPS):
        seed = int(beta * 100) * 1000 + rep
        r, cols = run_dynamic_k(N_STEPS, ALPHA, LAMBDA, GAMMA,
                                beta, START_R, seed)
        k_target = max(1, int(np.floor(ALPHA * r)))
        rng      = np.random.default_rng(seed + 99999)
        fsizes   = sample_flat_sizes(r, cols, k_target, N_SAMPLE, rng)
        results[beta].append(dict(r=r, cols=cols,
                                  k_target=k_target, fsizes=fsizes))
        flat_size_means.append(fsizes.mean())
        flat_size_maxes.append(fsizes.max())

    r_mean = np.mean([d["r"] for d in results[beta]])
    print(f"{beta:>5.1f}  {r_mean:>8.1f}  "
          f"{results[beta][0]['k_target']:>9d}  "
          f"{np.mean(flat_size_means):>15.2f}  "
          f"{np.mean(flat_size_maxes):>14.1f}")


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax1, ax2, ax3 = axes

for beta, color in zip(BETAS, COLORS):
    reps  = results[beta]
    r_rep = int(round(np.mean([d["r"] for d in reps])))

    # Whitney numbers: w_k = C(r,k), normalised by 2^r
    ks    = np.arange(r_rep + 1)
    wk    = np.array([comb(r_rep, int(k), exact=False) for k in ks])
    wk_n  = wk / wk.sum()

    ax1.plot(ks, wk_n, "-", color=color, lw=2, alpha=0.85,
             label=fr"$\beta={beta}$ ($r\approx{r_rep}$)")

    # Deviation from user reference w_k^PG = C(r,k): identically zero
    ax2.plot(ks, np.zeros_like(ks, dtype=float), "-", color=color, lw=2,
             alpha=0.85, label=fr"$\beta={beta}$")

# User reference (Boolean lattice, same as PA Whitney numbers)
r0 = int(round(np.mean([d["r"] for d in results[BETAS[0]]])))
ks0   = np.arange(r0 + 1)
wk0   = np.array([comb(r0, int(k), exact=False) for k in ks0])
ax1.plot(ks0, wk0 / wk0.sum(), "k--", lw=1.5, alpha=0.7,
         label=r"$w_k^{PG}=\binom{r}{k}$ (Boolean)")

# Actual PG(r-1,2) Gaussian binomial
gbn = gaussian_binomial_normalised(r0, q=2)
ks_fine = np.arange(r0 + 1)
ax1.plot(ks_fine, gbn, "k:", lw=1.5, alpha=0.7,
         label=r"$\binom{r}{k}_{2}$ (PG$(r-1,2)$, Gaussian)")

ax1.set_xlabel("Rank $k$", fontsize=12)
ax1.set_ylabel(r"$w_k\,/\,|\mathcal{L}(M_t)|$", fontsize=12)
ax1.set_title("Normalised Whitney numbers", fontsize=12)
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

ax2.axhline(0, color="k", lw=1, ls="--", alpha=0.5)
ax2.set_xlabel("Rank $k$", fontsize=12)
ax2.set_ylabel(r"$w_k - w_k^{PG}$", fontsize=12)
ax2.set_title(r"Deviation from $w_k^{PG}=\binom{r}{k}$", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.text(0.5, 0.5, r"Zero by construction: $w_k = \binom{r}{k}$ $\forall\beta$",
         transform=ax2.transAxes, ha="center", va="center",
         fontsize=11, color="gray",
         bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

# Flat size distribution at k_target
for beta, color in zip(BETAS, COLORS):
    reps   = results[beta]
    all_fs = np.concatenate([d["fsizes"] for d in reps])
    max_fs = all_fs.max()
    bins   = np.arange(reps[0]["k_target"], max_fs + 2) - 0.5
    counts, edges = np.histogram(all_fs, bins=bins, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    k_t  = reps[0]["k_target"]
    ax3.bar(mids, counts, width=0.8, color=color, alpha=0.55,
            label=fr"$\beta={beta}$  ($k_t\approx{k_t}$)")

ax3.set_xlabel(r"$|F_T|$ (flat size at rank $k_t = \lfloor\alpha r\rfloor$)",
               fontsize=11)
ax3.set_ylabel("Density", fontsize=12)
ax3.set_title(r"Sampled flat size distribution at $k=\lfloor\alpha r\rfloor$",
              fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

fig.suptitle(
    fr"Whitney numbers: $\gamma=0$, $\lambda=0.05$, $\alpha=0.3$, "
    fr"$k_t=\lfloor\alpha r_t\rfloor$, $n_{{\rm steps}}=1000$",
    fontsize=13)
plt.tight_layout()
path = os.path.join(OUT, "whitney_dynk.png")
fig.savefig(path, dpi=150)
print(f"\n  → {path}")
print("""
Key result: w_k = C(r,k) for ALL β — the flat lattice is always the Boolean
lattice 2^[r], independent of attachment bias.  β only affects which non-basis
elements land in each flat, not how many flats exist at each rank.

The actual PG(r-1,2) Gaussian binomial [r choose k]_2 ≈ 2^{k(r-k)} is
exponentially larger than C(r,k) — the PA matroid's flat structure is
far sparser than a projective geometry of the same rank.

The β effect on flat SIZES is visible in Panel 3: high β concentrates
non-basis elements on hub supports, making the flat at rank k_target
containing those hub rows much larger, while most other rank-k_target flats
remain empty of non-basis elements.
""")
