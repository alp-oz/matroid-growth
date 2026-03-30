"""
PA binary matroids — three replacement plots, γ=1 regime.

  λ ∈ {0.5, 1.0, 2.0}  (= C in engine)
  β ∈ {0.5, 1.0, 1.5, 2.0}
  γ=1, k=4, n_steps=5000, start_r=10

Generates (saving alongside this script):
  zipf_law.png            — support frequency distribution
  phase_transition_v2.png — support concentration vs density / β
  threshold_phenomena.png — rank saturation + triangle minor threshold

Terminology:
  'support frequency'    replaces 'row usage'
  'support concentration' replaces 'Gini' (= fraction in top 10% of coords)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

from core.engine import MatroidEngine
from analysis.stats import get_zipf_distribution
from analysis.probe_minors import convert_to_bitsets, calculate_binary_rank

# ── Config ─────────────────────────────────────────────────────────────────────
LAMBDA_VALUES = [0.5, 1.0, 2.0]
BETA_VALUES   = [0.5, 1.0, 1.5, 2.0]
LAM_FIXED     = 1.0
N_STEPS       = 5000
START_R       = 10
K             = 4
GAMMA         = 1.0
REPS          = 20

# Density sweep: log-spaced n_steps from ~13 to 5000
_raw = np.round(np.logspace(1.1, 3.7, 30)).astype(int)
N_STEPS_SWEEP = sorted(set(_raw.tolist()))

# Fixed n_steps for 'vary λ' panels  (gives ρ ≈ 12–16 across all λ)
N_STEPS_FIXED = 150

BETA_COLORS  = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
LAM_LS       = ["-", "--", ":"]
LAM_COLORS   = ["#2c3e50", "#8e44ad", "#c0392b"]
OUT          = os.path.dirname(__file__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def concentration(usage):
    """Fraction of total support absorbed by top 10% of coordinates."""
    u = np.sort(usage[usage > 0])[::-1]
    if len(u) == 0 or u.sum() == 0:
        return 0.0
    top_n = max(1, int(np.ceil(len(u) * 0.1)))
    return float(u[:top_n].sum() / u.sum())


def has_triangle(columns):
    """
    True iff attachment columns contain a GF(2) triangle:
    three distinct non-zero bitsets a, b, c with a ⊕ b = c.
    Uses early termination — fast when triangles are dense.
    """
    if not columns:
        return False
    bits = list(set(convert_to_bitsets(columns)) - {0})
    if len(bits) < 3:
        return False
    lookup = set(bits)
    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            c = bits[i] ^ bits[j]
            if c != 0 and c in lookup:
                return True
    return False


def fit_zipf(ranks, counts):
    def f(x, log_a, s): return log_a - s * x
    try:
        lr, lc = np.log(ranks.astype(float)), np.log(counts.astype(float))
        popt, _ = curve_fit(f, lr, lc, p0=[lc[0], 1.0])
        r2 = 1 - (lc - f(lr, *popt)).var() / lc.var()
        return float(popt[1]), float(r2)
    except Exception:
        return np.nan, np.nan


def run_one(n_steps, beta, lam, seed):
    np.random.seed(seed)
    return MatroidEngine(n_steps=n_steps, k_params=K, C=lam, gamma=GAMMA,
                         beta=beta, start_r=START_R).run()


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Support frequency distribution  (zipf_law.png)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 1: support frequency distribution …")

fig, ax = plt.subplots(figsize=(8, 6))

for beta, color in zip(BETA_VALUES, BETA_COLORS):
    data = run_one(N_STEPS, beta, LAM_FIXED, seed=42)
    ranks, counts = get_zipf_distribution(data["row_usage"])
    s, r2 = fit_zipf(ranks, counts)
    lbl = f"β={beta}  s={s:.2f}, R²={r2:.2f}  (r={data['r']})"
    ax.loglog(ranks, counts, "o", color=color, ms=5.5, alpha=0.8, label=lbl)
    if not np.isnan(s):
        a0 = float(counts[0]) * float(ranks[0]) ** s
        rr = np.array([ranks[0], ranks[-1]], dtype=float)
        ax.loglog(rr, a0 * rr ** (-s), "-", color=color, lw=2, alpha=0.9)
    print(f"  β={beta}: s={s:.3f}, R²={r2:.3f}, r_final={data['r']}, "
          f"ρ={data['n']/data['r']:.1f}")

ax.set_xlabel("Coordinate rank (by frequency)", fontsize=12)
ax.set_ylabel("Support frequency count", fontsize=12)
ax.set_title(
    "Support frequency distribution in PA binary matroids\n"
    f"γ={GAMMA}, λ={LAM_FIXED}, k={K}, n_steps={N_STEPS}, start_r={START_R}",
    fontsize=11, fontweight="bold"
)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "zipf_law.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → zipf_law.png saved\n")


# ══════════════════════════════════════════════════════════════════════════════
# Shared density sweep  (λ=LAM_FIXED, all β, all n_steps in N_STEPS_SWEEP)
# Computes: ρ, support concentration, rank_frac, P(triangle)
# ══════════════════════════════════════════════════════════════════════════════
print("Running density sweep (shared for Plots 2 & 3) …")
print(f"  {len(N_STEPS_SWEEP)} n_steps × {len(BETA_VALUES)} β × {REPS} reps = "
      f"{len(N_STEPS_SWEEP)*len(BETA_VALUES)*REPS} runs")

sweep = {}  # (n_steps, beta) → dict of mean metrics

for n_s in N_STEPS_SWEEP:
    for beta in BETA_VALUES:
        rhos, concs, rfracs, tris = [], [], [], []
        for seed in range(REPS):
            data = run_one(n_s, beta, LAM_FIXED, seed=seed * 41)
            r_fin = data["r"]
            n_fin = data["n"]
            cols  = data["columns"]
            rhos.append(n_fin / r_fin if r_fin > 0 else 1.0)
            concs.append(concentration(data["row_usage"]))
            if cols and r_fin > 0:
                rk = calculate_binary_rank(convert_to_bitsets(cols))
                rfracs.append(rk / r_fin)
            else:
                rfracs.append(0.0)
            tris.append(has_triangle(cols))
        sweep[(n_s, beta)] = {
            "rho":    np.mean(rhos),
            "conc":   np.mean(concs),
            "rfrac":  np.mean(rfracs),
            "p_tri":  np.mean(tris),
        }
    print(f"  n_steps={n_s:5d}  done", flush=True)

# Fixed-density sweep: vary λ and β at N_STEPS_FIXED
print(f"\nRunning fixed-density sweep (n_steps={N_STEPS_FIXED}, vary λ) …")

fixed = {}  # (lam, beta) → dict

for lam in LAMBDA_VALUES:
    for beta in BETA_VALUES:
        rhos, concs, tris = [], [], []
        for seed in range(REPS):
            data = run_one(N_STEPS_FIXED, beta, lam, seed=seed * 41)
            r_fin = data["r"]
            n_fin = data["n"]
            rhos.append(n_fin / r_fin if r_fin > 0 else 1.0)
            concs.append(concentration(data["row_usage"]))
            tris.append(has_triangle(data["columns"]))
        fixed[(lam, beta)] = {
            "rho":   np.mean(rhos),
            "conc":  np.mean(concs),
            "p_tri": np.mean(tris),
        }
print("  done\n")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Phase structure  (phase_transition_v2.png)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 2: phase structure …")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: concentration vs ρ for each β  (λ=LAM_FIXED)
for beta, color in zip(BETA_VALUES, BETA_COLORS):
    pts = sorted(
        [(sweep[(n_s, beta)]["rho"], sweep[(n_s, beta)]["conc"])
         for n_s in N_STEPS_SWEEP],
        key=lambda x: x[0]
    )
    axL.plot([p[0] for p in pts], [p[1] for p in pts],
             "o-", color=color, lw=2, ms=5, label=f"β={beta}")

axL.set_xlabel("Attachment density  ρ = n_t / r_t", fontsize=11)
axL.set_ylabel("Support concentration\n(fraction in top 10% of coordinates)", fontsize=11)
axL.set_title(f"Support concentration vs density  (λ={LAM_FIXED}, γ={GAMMA})",
              fontsize=11, fontweight="bold")
axL.set_ylim(0, 1.05)
axL.legend(fontsize=9)
axL.grid(alpha=0.3)

# Right: concentration vs β for each λ  (fixed n_steps=N_STEPS_FIXED)
for lam, ls, color in zip(LAMBDA_VALUES, LAM_LS, LAM_COLORS):
    rho_mean = np.mean([fixed[(lam, b)]["rho"] for b in BETA_VALUES])
    concs    = [fixed[(lam, b)]["conc"] for b in BETA_VALUES]
    axR.plot(BETA_VALUES, concs, "o" + ls, color=color, lw=2, ms=7,
             label=f"λ={lam}  (ρ̄≈{rho_mean:.0f})")

axR.set_xlabel("Attachment bias  β", fontsize=11)
axR.set_ylabel("Support concentration", fontsize=11)
axR.set_title(f"Support concentration vs β  (fixed n_steps={N_STEPS_FIXED}, γ={GAMMA})",
              fontsize=11, fontweight="bold")
axR.set_ylim(0, 1.05)
axR.legend(fontsize=9)
axR.grid(alpha=0.3)

fig.suptitle(
    f"Phase structure of PA binary matroids  "
    f"(γ={GAMMA}, k={K}, start_r={START_R}, {REPS} reps)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "phase_transition_v2.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → phase_transition_v2.png saved\n")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Threshold phenomena  (threshold_phenomena.png)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 3: threshold phenomena …")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

# Left + Middle: density sweep for each β  (λ=LAM_FIXED)
for beta, color in zip(BETA_VALUES, BETA_COLORS):
    pts = sorted(
        [(sweep[(n_s, beta)]["rho"],
          sweep[(n_s, beta)]["rfrac"],
          sweep[(n_s, beta)]["p_tri"])
         for n_s in N_STEPS_SWEEP],
        key=lambda x: x[0]
    )
    rhos   = [p[0] for p in pts]
    rfracs = [p[1] for p in pts]
    p_tris = [p[2] for p in pts]
    ax1.plot(rhos, rfracs, "o-", color=color, lw=2, ms=5, label=f"β={beta}")
    ax2.plot(rhos, p_tris, "s-", color=color, lw=2, ms=5, label=f"β={beta}")

ax1.axhline(1.0, color="gray", ls="--", lw=0.9, alpha=0.5)
ax1.set_xlabel("Attachment density  ρ = n_t / r_t", fontsize=11)
ax1.set_ylabel("Normalised rank  r_t / r_max", fontsize=11)
ax1.set_title(f"Rank saturation  (λ={LAM_FIXED}, γ={GAMMA})",
              fontsize=11, fontweight="bold")
ax1.set_ylim(-0.03, 1.1)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5)
ax2.set_xlabel("Attachment density  ρ = n_t / r_t", fontsize=11)
ax2.set_ylabel("P(triangle minor)", fontsize=11)
ax2.set_title(f"Triangle minor threshold  (λ={LAM_FIXED}, γ={GAMMA})",
              fontsize=11, fontweight="bold")
ax2.set_ylim(-0.03, 1.05)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Right: P(triangle) vs λ for each β  (fixed n_steps=N_STEPS_FIXED)
for beta, color in zip(BETA_VALUES, BETA_COLORS):
    p_tris = [fixed[(lam, beta)]["p_tri"] for lam in LAMBDA_VALUES]
    ax3.plot(LAMBDA_VALUES, p_tris, "o-", color=color, lw=2, ms=7, label=f"β={beta}")

ax3.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5)
ax3.set_xlabel("Discovery rate scale  λ", fontsize=11)
ax3.set_ylabel("P(triangle minor)", fontsize=11)
ax3.set_title(f"P(triangle) vs λ  (fixed n_steps={N_STEPS_FIXED}, γ={GAMMA})",
              fontsize=11, fontweight="bold")
ax3.set_xticks(LAMBDA_VALUES)
ax3.set_ylim(-0.03, 1.05)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

fig.suptitle(
    f"Threshold phenomena in PA binary matroids  "
    f"(γ={GAMMA}, k={K}, start_r={START_R}, {REPS} reps)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "threshold_phenomena.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → threshold_phenomena.png saved")
print("\nAll done.")
