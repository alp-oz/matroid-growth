"""
Two-panel publication figure.

Left:  Three stacked histograms of N_8 across 50 replicates for
       β ∈ {0.5, 1.0, 1.5} with fitted NB(r̂, p̂) curves overlaid.

Right: r̂(β) vs β for s ∈ {5, 8, 10} with 95 % bootstrap CIs,
       β* ≈ 1.10 marked as a vertical dashed line.

Parameters: γ=0, λ=0.05, α=0.3, k_t=⌊0.3 r_t⌋, n_steps=2000,
            start_r=10, 50 replicates.
Seeds identical to pa_circuit_nb_beta.py.

Saves: nb_figure.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import gammaln, betainc
from scipy.optimize import minimize_scalar
import os

ALPHA    = 0.3
LAMBDA   = 0.05
GAMMA    = 0.0
N_STEPS  = 2000
START_R  = 10
REPS     = 50
BETAS    = [0.5, 1.0, 1.5]
SIZES    = [5, 8, 10]
N_BOOT   = 1000
BETA_STAR = 1.10
OUT      = os.path.dirname(__file__)

BETA_COLOR  = {0.5: "#2c7bb6", 1.0: "#e08214", 1.5: "#1a9641"}
SIZE_COLOR  = {5:  "#7b2d8b", 8: "#d7191c", 10: "#2c7bb6"}
SIZE_MARKER = {5: "o",        8: "s",       10: "^"}


# ── Simulation ────────────────────────────────────────────────────────────────

def run_and_collect(n_steps, alpha, C_lam, gamma, beta, start_r, seed):
    rng       = np.random.default_rng(seed)
    curr_r    = start_r
    row_usage = np.ones(start_r, dtype=np.float64)
    supports, k_list = [], []
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
            supports.append(sel);  k_list.append(k)
            for idx in sel:
                row_usage[int(idx)] += 1
    return supports, np.array(k_list, dtype=np.int32), curr_r


def count_by_size(supports, k_vals, r_final, target_sizes):
    n_att  = len(supports)
    result = {s: 0 for s in target_sizes}
    for s in target_sizes:
        result[s] += int(np.sum(k_vals == s - 1))
    if n_att == 0:
        return result
    B = np.zeros((n_att, r_final), dtype=np.float32)
    for j, sel in enumerate(supports):
        B[j, sel] = 1.0
    G   = (B @ B.T).round().astype(np.int32)
    k32 = k_vals.astype(np.int32)
    D   = k32[:, None] + k32[None, :] - 2 * G
    tri = np.triu(np.ones((n_att, n_att), dtype=bool), k=1)
    for s in target_sizes:
        result[s] += int(np.sum((D == s - 2) & (G > 0) & tri))
    return result


# ── NB fitting ────────────────────────────────────────────────────────────────

def nb_logpmf(k, r, p):
    k = np.asarray(k, dtype=float)
    return (gammaln(k + r) - gammaln(r) - gammaln(k + 1)
            + r * np.log(p) + k * np.log(1.0 - p))

def nb_cdf(k, r, p):
    return float(betainc(r, float(k) + 1, p)) if k >= 0 else 0.0

def fit_nb(data):
    data = np.asarray(data, dtype=float)
    mu   = data.mean()
    if mu <= 0:
        return float("nan"), float("nan"), False
    def neg_ll(log_r):
        r = np.exp(log_r)
        p = r / (r + mu)
        return -np.sum(nb_logpmf(data, r, p))
    res   = minimize_scalar(neg_ll, bounds=(-5, 16), method="bounded",
                            options={"xatol": 1e-8})
    r_hat = np.exp(res.x)
    p_hat = r_hat / (r_hat + mu)
    return float(r_hat), float(p_hat), res.success

def bootstrap_r_ci(data, n_boot=1000, seed=None, alpha=0.05):
    rng  = np.random.default_rng(seed)
    data = np.asarray(data)
    rs   = [fit_nb(rng.choice(data, size=len(data), replace=True))[0]
            for _ in range(n_boot)]
    rs   = [r for r in rs if np.isfinite(r)]
    return (float(np.percentile(rs, 100*alpha/2)),
            float(np.percentile(rs, 100*(1-alpha/2))))


# ── Collect data ──────────────────────────────────────────────────────────────

print("Running simulations …")
count_arr = {beta: {s: [] for s in SIZES} for beta in BETAS}

for b_idx, beta in enumerate(BETAS):
    for rep in range(REPS):
        seed = b_idx * 100000 + 99000 + rep   # same seeds as pa_circuit_nb_beta.py
        sup, kv, rf = run_and_collect(N_STEPS, ALPHA, LAMBDA, GAMMA,
                                      beta, START_R, seed)
        counts = count_by_size(sup, kv, rf, SIZES)
        for s in SIZES:
            count_arr[beta][s].append(counts[s])
    print(f"  β={beta}  done")

for beta in BETAS:
    for s in SIZES:
        count_arr[beta][s] = np.array(count_arr[beta][s])

# ── Fit NB ────────────────────────────────────────────────────────────────────

print("Fitting NB + bootstrap CIs …")
rng_boot = np.random.default_rng(4242)
fits = {beta: {} for beta in BETAS}

for beta in BETAS:
    for s in SIZES:
        arr             = count_arr[beta][s]
        r_hat, p_hat, _ = fit_nb(arr)
        ci_lo, ci_hi    = bootstrap_r_ci(arr, n_boot=N_BOOT, seed=rng_boot)
        fits[beta][s]   = dict(r=r_hat, p=p_hat,
                               mu=float(arr.mean()),
                               ci_lo=ci_lo, ci_hi=ci_hi)
        print(f"  β={beta}  s={s}  r̂={r_hat:.3f} [{ci_lo:.3f},{ci_hi:.3f}]")


# ── Figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 8))
outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38,
                          left=0.07, right=0.97, top=0.90, bottom=0.10)

# Left: 3 stacked sub-axes (one per β), share no x-axis
left_gs  = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0],
                                             hspace=0.55)
ax_hist  = [fig.add_subplot(left_gs[i]) for i in range(3)]

# Right: single axis for r̂(β)
ax_right = fig.add_subplot(outer[1])


# ── Left panel ────────────────────────────────────────────────────────────────

s_hist = 8

for row, beta in enumerate(BETAS):
    ax   = ax_hist[row]
    arr  = count_arr[beta][s_hist]
    f    = fits[beta][s_hist]
    r_h, p_h, mu_h = f["r"], f["p"], f["mu"]
    col  = BETA_COLOR[beta]

    # Histogram (density so NB PMF overlays correctly)
    n_bins = 20
    cnts, bins, patches = ax.hist(
        arr, bins=n_bins, density=True,
        color=col, alpha=0.55, edgecolor="white", linewidth=0.5)
    bw = bins[1] - bins[0]

    # NB PMF curve evaluated on fine grid within data range
    pad  = max(bw, (arr.max() - arr.min()) * 0.08)
    x_lo = max(0, int(arr.min() - pad))
    x_hi = int(arr.max() + pad)
    grid = np.linspace(x_lo, x_hi, 1200).astype(int)
    pmf  = np.exp(nb_logpmf(grid, r_h, p_h))
    ax.plot(grid, pmf, "-", color=col, lw=2.5,
            label=fr"NB $\hat r = {r_h:.2f}$")
    ax.fill_between(grid, 0, pmf, color=col, alpha=0.18)

    # Formatting
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlim(x_lo, x_hi)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.18)

    # β label box on the right
    ax.text(0.985, 0.86, fr"$\beta = {beta}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=col, alpha=0.85, lw=1.3))

    # r̂ annotation
    ax.text(0.985, 0.55, fr"$\hat r = {r_h:.2f}$   $\hat\mu = {mu_h:,.0f}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="k")

    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.18, lw=0.7)

    # x-label only on bottom sub-axis
    if row == 2:
        ax.set_xlabel(fr"$N_8$  (circuits of size 8)", fontsize=10)
    else:
        ax.set_xticklabels([])

# Shared left-panel title (placed above top sub-axis)
ax_hist[0].set_title(
    fr"$N_8$ distribution across 50 replicates  $\pm$ NB fit",
    fontsize=10.5, pad=6)


# ── Right panel ───────────────────────────────────────────────────────────────

beta_arr = np.array(BETAS)
JITTER   = {5: -0.035, 8: 0.0, 10: 0.035}

for s in SIZES:
    rs    = np.array([fits[b][s]["r"]     for b in BETAS])
    ci_lo = np.array([fits[b][s]["ci_lo"] for b in BETAS])
    ci_hi = np.array([fits[b][s]["ci_hi"] for b in BETAS])
    xp    = beta_arr + JITTER[s]
    ax_right.errorbar(
        xp, rs, yerr=[rs - ci_lo, ci_hi - rs],
        fmt=SIZE_MARKER[s] + "-", color=SIZE_COLOR[s],
        lw=2.2, ms=8, capsize=5, capthick=1.5,
        label=fr"$s = {s}$")
    for i, b in enumerate(BETAS):
        ax_right.annotate(
            f"{rs[i]:.1f}", (xp[i], rs[i]),
            textcoords="offset points",
            xytext=(7 if i < 2 else -30, 4),
            fontsize=8, color=SIZE_COLOR[s])

# β* line
ax_right.set_yscale("log")

ax_right.axvline(BETA_STAR, color="dimgray", ls="--", lw=1.8,
                 label=fr"$\beta^* \approx {BETA_STAR}$")
ax_right.text(BETA_STAR + 0.02, ax_right.get_ylim()[0] * 1.8,
              fr"$\beta^*$", color="dimgray", fontsize=11, va="bottom")
ax_right.set_xlabel(r"Attachment bias $\beta$", fontsize=12)
ax_right.set_ylabel(r"NB dispersion $\hat r(\beta)$  [log scale]", fontsize=12)
ax_right.set_title("")
ax_right.set_xticks(BETAS)
ax_right.legend(fontsize=10, title="Circuit size $s$",
                title_fontsize=9, loc="upper left")
ax_right.grid(True, which="both", alpha=0.2, lw=0.7)

# Add phase labels
ylim = ax_right.get_ylim()
ax_right.text(0.72, 0.10,  "diverse-hub\nphase",
              transform=ax_right.transAxes, ha="center",
              fontsize=8.5, color="dimgray", style="italic")
ax_right.text(0.88, 0.10, "hub-collapse\nphase",
              transform=ax_right.transAxes, ha="center",
              fontsize=8.5, color="dimgray", style="italic")
ax_right.axvspan(BETA_STAR, 1.65, alpha=0.06, color="dimgray")


# ── Supertitle ────────────────────────────────────────────────────────────────

fig.suptitle(
    r"Negative binomial fit to circuit counts $N_s$:  "
    r"$\gamma=0$,  $\lambda=0.05$,  $\alpha=0.3$,  "
    r"$k_t = \lfloor 0.3\,r_t \rfloor$,  $n_{\rm steps}=2000$,  50 reps",
    fontsize=11.5, y=0.97)

path = os.path.join(OUT, "nb_figure.png")
fig.savefig(path, dpi=180)
print(f"\n  → {path}")

# ── Print r̂ table ────────────────────────────────────────────────────────────
print(f"\n{'β':>5}  {'s':>4}  {'r̂':>8}  {'95% CI':>18}  {'μ̂':>12}")
print("─" * 55)
for beta in BETAS:
    for s in SIZES:
        f = fits[beta][s]
        print(f"{beta:>5.1f}  {s:>4d}  {f['r']:>8.3f}  "
              f"[{f['ci_lo']:>6.3f}, {f['ci_hi']:>6.3f}]  "
              f"{f['mu']:>12,.0f}")
    print()
