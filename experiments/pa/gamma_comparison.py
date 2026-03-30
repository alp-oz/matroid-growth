"""
Discovery-rate regime comparison for PA binary matroid.

Runs zipf, phase-transition, and threshold experiments across three
(gamma, C) regimes calibrated to give comparable rank growth (~120-160
new rows over 3000 steps with start_r=50):

  Regime A: γ=0,   C=0.05  — constant discovery rate (~150 new rows)
  Regime B: γ=0.5, C=1.5   — polynomial decay        (~160 new rows)
  Regime C: γ=1,   C=20    — logarithmic decay        (~120 new rows)

Output: gamma_zipf.png, gamma_phase.png, gamma_threshold.png
        (saved next to this script)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

from core.engine import MatroidEngine
from analysis.stats import get_zipf_distribution
from analysis.probe_minors import convert_to_bitsets

# ── Config ─────────────────────────────────────────────────────────────────────
REGIMES = [
    {"label": "γ=0,   C=0.05", "gamma": 0.0, "C": 0.05, "ls": "-",  "color": "#2980b9"},
    {"label": "γ=0.5, C=1.5",  "gamma": 0.5, "C": 1.5,  "ls": "--", "color": "#e67e22"},
    {"label": "γ=1,   C=20",   "gamma": 1.0, "C": 20.0, "ls": ":",  "color": "#8e44ad"},
]

START_R  = 50
K_PARAMS = 4

# Zipf
N_STEPS_ZIPF = 3000
BETA_VALUES  = [0.2, 0.5, 1.0, 1.5, 2.0]
ITERS_ZIPF   = 8

# Phase transition
BETA_FIXED  = 0.8
N_DENSITY   = [max(2, int(d * START_R)) for d in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]]
DENSITY_NOM = [s / START_R for s in N_DENSITY]
BETA_VALS   = [0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0]
N_FIXED     = 2 * START_R
ITERS_PHASE = 8

# Threshold
N_OVER_R     = np.concatenate([np.linspace(0.1, 1.0, 7),
                                np.linspace(1.2, 4.0, 14),
                                np.linspace(4.5, 10.0, 5)])
N_STEPS_T    = [max(2, int(d * START_R)) for d in N_OVER_R]
START_R_VALS = (list(range(10, 40, 5)) +
                list(range(40, 100, 8)) +
                list(range(100, 180, 12)) +
                list(range(180, 300, 18)))
ITERS_THRESH = 15
ITERS_R      = 40

OUT = os.path.dirname(__file__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def gini(arr):
    arr = np.sort(arr[arr > 0].astype(float))
    n = len(arr)
    if n == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * arr) / (n * arr.sum())) - (n + 1) / n)


def has_girth_le_3(columns):
    if not columns:
        return False
    bits = convert_to_bitsets(columns)
    lookup = set(bits)
    if len(lookup) < len(bits):
        return True
    unique = [b for b in lookup if b != 0]
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            if (unique[i] ^ unique[j]) in lookup:
                return True
    return False


def incremental_rank(columns):
    basis = []
    for col in columns:
        v = 0
        for row in col:
            v |= (1 << row)
        for b in basis:
            v = min(v, v ^ b)
        if v > 0:
            basis.append(v)
            basis.sort(reverse=True)
    return len(basis)


def fit_zipf(ranks, counts):
    log_r = np.log(ranks.astype(float))
    log_c = np.log(counts.astype(float))
    def f(x, log_a, s): return log_a - s * x
    try:
        popt, _ = curve_fit(f, log_r, log_c, p0=[np.log(counts[0]), 1.0])
        resid = log_c - f(log_r, *popt)
        r2 = 1 - resid.var() / log_c.var() if log_c.var() > 0 else 0.0
        return float(popt[1]), float(r2)
    except Exception:
        return np.nan, np.nan


def run_eng(n_steps, beta, gamma, C, start_r=START_R, seed=0):
    np.random.seed(seed)
    return MatroidEngine(n_steps=n_steps, k_params=K_PARAMS, C=C,
                         gamma=gamma, beta=beta, start_r=start_r).run()


# ── Figure 1: Zipf ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Figure 1: Zipf comparison")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: Zipf exponent s vs beta — 3 regimes overlaid
for reg in REGIMES:
    mean_s, std_s = [], []
    for beta in BETA_VALUES:
        slopes = []
        for rep in range(ITERS_ZIPF):
            data = run_eng(N_STEPS_ZIPF, beta, reg["gamma"], reg["C"],
                           seed=rep * 100)
            ranks, counts = get_zipf_distribution(data["row_usage"])
            s, _ = fit_zipf(ranks, counts)
            if not np.isnan(s):
                slopes.append(s)
        mean_s.append(np.mean(slopes) if slopes else np.nan)
        std_s.append(np.std(slopes)  if slopes else np.nan)
    mean_s = np.array(mean_s); std_s = np.array(std_s)
    finite = np.isfinite(mean_s)
    ax1.errorbar(np.array(BETA_VALUES)[finite], mean_s[finite], yerr=std_s[finite],
                 fmt="o" + reg["ls"], color=reg["color"], capsize=3, lw=2, ms=6,
                 label=reg["label"])
    print(f"  {reg['label']}: s = {mean_s}")

beta_ref = np.linspace(0.2, 2.0, 100)
ax1.plot(beta_ref, beta_ref, "r--", lw=1, alpha=0.5, label="s=β (ref)")
ax1.plot(beta_ref, np.ones_like(beta_ref), "k:", lw=1, alpha=0.4, label="s=1 (Zipf)")
ax1.set_xlabel("Attachment bias  β", fontsize=12)
ax1.set_ylabel("Fitted Zipf exponent  s", fontsize=12)
ax1.set_title("Zipf exponent vs β\nacross discovery regimes", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9); ax1.set_xlim(0, 2.2); ax1.set_ylim(bottom=0); ax1.grid(alpha=0.3)

# Panel 2: Raw Zipf distributions for beta=1.0 — 3 regimes overlaid
COLORS_RAW = plt.cm.plasma(np.linspace(0.15, 0.85, len(REGIMES)))
for reg, color in zip(REGIMES, COLORS_RAW):
    data = run_eng(N_STEPS_ZIPF, 1.0, reg["gamma"], reg["C"], seed=42)
    ranks, counts = get_zipf_distribution(data["row_usage"])
    s, _ = fit_zipf(ranks, counts)
    ax2.loglog(ranks, counts, "o", color=color, ms=4, alpha=0.7,
               label=f"{reg['label']}  (s={s:.2f}, r={data['r']})")
    if not np.isnan(s):
        a = counts[0] * ranks[0] ** s
        rr = np.array([ranks[0], ranks[-1]], dtype=float)
        ax2.loglog(rr, a * rr ** (-s), "-", color=color, lw=1.5, alpha=0.9)

ax2.set_xlabel("Rank", fontsize=12)
ax2.set_ylabel("Row usage (frequency)", fontsize=12)
ax2.set_title("Zipf distributions  (β=1.0)\nfinal rank r shown in legend", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(True, which="both", alpha=0.3)

fig.suptitle("Effect of discovery-rate decay on Zipf's law in row usage",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "gamma_zipf.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved gamma_zipf.png\n")


# ── Figure 2: Phase transition ─────────────────────────────────────────────────
print("Figure 2: Phase transition comparison")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
(ax_rd_d, ax_rd_b), (ax_g_d, ax_g_b) = axes

for reg in REGIMES:
    # Density sweep (fixed beta=BETA_FIXED)
    rd_means, gi_means, gini_means = [], [], []
    for n_steps in N_DENSITY:
        rd_s, gi_s, gini_s = [], [], []
        for seed in range(ITERS_PHASE):
            data = run_eng(n_steps, BETA_FIXED, reg["gamma"], reg["C"], seed=seed * 17)
            cols = data["columns"]
            if not cols:
                rd_s.append(0.0); gi_s.append(np.nan); gini_s.append(0.0); continue
            rk = incremental_rank(cols)
            n_cols = len(cols)
            rd_s.append((n_cols - rk) / n_cols if n_cols > 0 else 0.0)
            gi_s.append(has_girth_le_3(cols))
            gini_s.append(gini(data["row_usage"]))
        rd_means.append(np.mean(rd_s))
        gi_means.append(np.nanmean(gi_s))
        gini_means.append(np.mean(gini_s))

    ax_rd_d.plot(DENSITY_NOM, rd_means, "o" + reg["ls"], color=reg["color"],
                 lw=2, ms=5, label=reg["label"])
    ax_g_d.plot(DENSITY_NOM, gi_means, "s" + reg["ls"], color=reg["color"],
                lw=2, ms=5, label=reg["label"])

    # Beta sweep (fixed nominal density ≈ 2)
    rd_b, gi_b, gini_b = [], [], []
    for beta in BETA_VALS:
        rd_s, gi_s, gini_s = [], [], []
        for seed in range(ITERS_PHASE):
            data = run_eng(N_FIXED, beta, reg["gamma"], reg["C"], seed=seed * 17)
            cols = data["columns"]
            if not cols:
                rd_s.append(0.0); gi_s.append(np.nan); gini_s.append(0.0); continue
            rk = incremental_rank(cols)
            n_cols = len(cols)
            rd_s.append((n_cols - rk) / n_cols if n_cols > 0 else 0.0)
            gi_s.append(has_girth_le_3(cols))
            gini_s.append(gini(data["row_usage"]))
        rd_b.append(np.mean(rd_s))
        gi_b.append(np.nanmean(gi_s))
        gini_b.append(np.mean(gini_s))

    ax_rd_b.plot(BETA_VALS, rd_b, "o" + reg["ls"], color=reg["color"],
                 lw=2, ms=5, label=reg["label"])
    ax_g_b.plot(BETA_VALS, gi_b, "s" + reg["ls"], color=reg["color"],
                lw=2, ms=5, label=reg["label"])

    print(f"  {reg['label']} done")

ax_rd_d.set_xlabel("Nominal density  n_steps / r₀", fontsize=11)
ax_rd_d.set_ylabel("Rank deficit  ρ = (n_att − rank) / n_att", fontsize=10)
ax_rd_d.set_title(f"Rank deficit vs density  (β={BETA_FIXED})", fontsize=11, fontweight="bold")
ax_rd_d.legend(fontsize=9); ax_rd_d.grid(alpha=0.3)

ax_rd_b.set_xlabel("Attachment bias  β", fontsize=11)
ax_rd_b.set_ylabel("Rank deficit  ρ", fontsize=10)
ax_rd_b.set_title(f"Rank deficit vs β  (nom. density≈2)", fontsize=11, fontweight="bold")
ax_rd_b.legend(fontsize=9); ax_rd_b.grid(alpha=0.3)

ax_g_d.set_xlabel("Nominal density  n_steps / r₀", fontsize=11)
ax_g_d.set_ylabel("P(girth ≤ 3)", fontsize=11)
ax_g_d.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5)
ax_g_d.set_title(f"P(triangle minor) vs density  (β={BETA_FIXED})", fontsize=11, fontweight="bold")
ax_g_d.legend(fontsize=9); ax_g_d.set_ylim(-0.03, 1.05); ax_g_d.grid(alpha=0.3)

ax_g_b.set_xlabel("Attachment bias  β", fontsize=11)
ax_g_b.set_ylabel("P(girth ≤ 3)", fontsize=11)
ax_g_b.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5)
ax_g_b.set_title(f"P(triangle minor) vs β  (nom. density≈2)", fontsize=11, fontweight="bold")
ax_g_b.legend(fontsize=9); ax_g_b.set_ylim(-0.03, 1.05); ax_g_b.grid(alpha=0.3)

fig.suptitle(f"Phase structure comparison across discovery regimes"
             f"  (r₀={START_R}, k={K_PARAMS}, {ITERS_PHASE} reps)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "gamma_phase.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved gamma_phase.png\n")


# ── Figure 3: Threshold phenomena ─────────────────────────────────────────────
print("Figure 3: Threshold comparison")
print("=" * 60)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

for reg in REGIMES:
    # Density sweep — rank saturation and P(girth≤3)
    rank_means, p_girths = [], []
    for n_steps, d in zip(N_STEPS_T, N_OVER_R):
        rr_list, girth_list = [], []
        for seed in range(ITERS_THRESH):
            np.random.seed(seed * 31)
            data = run_eng(n_steps, BETA_FIXED, reg["gamma"], reg["C"], seed=seed * 31)
            cols = data["columns"]
            r_fin = data["r"]
            if not cols:
                rr_list.append(0.0); girth_list.append(False); continue
            rk = incremental_rank(cols)
            rr_list.append(rk / r_fin if r_fin > 0 else 0.0)
            girth_list.append(has_girth_le_3(cols))
        rank_means.append(np.mean(rr_list))
        p_girths.append(np.mean(girth_list))

    ax1.plot(N_OVER_R, rank_means, "o-", color=reg["color"], lw=2, ms=4,
             ls=reg["ls"], label=reg["label"])
    ax2.plot(N_OVER_R, p_girths, "s-", color=reg["color"], lw=2, ms=4,
             ls=reg["ls"], label=reg["label"])

    # start_r sweep
    p_list = []
    for start_r_val in START_R_VALS:
        hits = []
        for seed in range(ITERS_R):
            np.random.seed(seed * 31 + start_r_val)
            data = MatroidEngine(n_steps=150, k_params=K_PARAMS, C=reg["C"],
                                 gamma=reg["gamma"], beta=BETA_FIXED,
                                 start_r=start_r_val).run()
            hits.append(has_girth_le_3(data["columns"]) if data["columns"] else False)
        p_list.append(np.mean(hits))

    # Logistic fit
    def logistic(x, x0, k): return 1.0 / (1.0 + np.exp(k * (x - x0)))
    xr = np.array(START_R_VALS, dtype=float)
    try:
        x0_guess = xr[np.argmin(np.abs(np.array(p_list) - 0.5))]
        popt, _ = curve_fit(logistic, xr, p_list, p0=[x0_guess, 0.05],
                            bounds=([xr[0], 0.001], [xr[-1], 1.0]), maxfev=5000)
        x_sm = np.linspace(xr[0], xr[-1], 400)
        ax3.plot(x_sm, logistic(x_sm, *popt), "-", color=reg["color"], lw=2.2,
                 ls=reg["ls"], label=f"{reg['label']}  (r*≈{popt[0]:.0f})")
        ax3.axvline(popt[0], color=reg["color"], ls=":", lw=1, alpha=0.5)
    except Exception:
        ax3.plot(xr, p_list, "-", color=reg["color"], lw=2, ls=reg["ls"],
                 label=reg["label"])
    ax3.plot(xr, p_list, "o", color=reg["color"], ms=3, alpha=0.3)

    print(f"  {reg['label']} done")

ax1.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
ax1.axhline(0.5, color="gray", ls=":",  lw=0.8, alpha=0.4)
ax1.set_xlabel("Nominal density  n_steps / r₀", fontsize=12)
ax1.set_ylabel("rank(attachments) / r_final", fontsize=12)
ax1.set_title("Rank saturation\n(attachment columns vs final basis size)",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=9); ax1.set_xlim(0, 6); ax1.set_ylim(0, 1.05); ax1.grid(alpha=0.3)

ax2.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5)
ax2.set_xlabel("Nominal density  n_steps / r₀", fontsize=12)
ax2.set_ylabel("P(girth ≤ 3)", fontsize=12)
ax2.set_title("Minor appearance threshold\nP(first triangle) vs density",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9); ax2.set_xlim(0, 10); ax2.set_ylim(-0.03, 1.05); ax2.grid(alpha=0.3)

ax3.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5)
ax3.set_xlabel("Initial row count  r₀", fontsize=12)
ax3.set_ylabel("P(girth ≤ 3)", fontsize=12)
ax3.set_title(f"start_r threshold  (n_steps=150 fixed)\nr* in legend",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=9); ax3.set_ylim(-0.03, 1.05); ax3.grid(alpha=0.3)

fig.suptitle(f"Threshold phenomena comparison across discovery regimes"
             f"  (β={BETA_FIXED}, k={K_PARAMS})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "gamma_threshold.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved gamma_threshold.png")
print("\nAll done.")
