"""
Fano minor (GF(2) triangle) appearance threshold experiment.

γ=0, C=0.05, k=4, n_steps=5000
varying start_r ∈ {5, 10, 20, 50, 100}  and  β ∈ {0.5, 1.0, 1.5, 2.0}
20 replicates per (start_r, β) combination.

check_fano_minor detects any GF(2) triangle among the attachment columns:
three distinct non-zero column vectors a, b, c with a ⊕ b = c.

Panel 1: P(Fano minor) vs start_r  for each β  [main result]
Panel 2: P(Fano minor) vs β        for each start_r  [β sensitivity]
Panel 3: Same as Panel 1 but for C=0.5 to check C dependence

Saves: fano_threshold.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from core.engine import MatroidEngine
from analysis.probe_minors import convert_to_bitsets, check_fano_minor

BETA_VALUES  = [0.5, 1.0, 1.5, 2.0]
START_R_VALS = [5, 10, 20, 50, 100]
C_MAIN       = 0.05
C_ALT        = 0.5     # second C to test whether threshold shifts
N_STEPS      = 5000
K            = 4
REPS         = 20

BETA_COLORS = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
SR_COLORS   = plt.cm.viridis(np.linspace(0.1, 0.9, len(START_R_VALS)))


def p_fano(start_r, beta, C, reps=REPS):
    hits = []
    for seed in range(reps):
        np.random.seed(seed * 37 + int(beta * 100) + start_r)
        data = MatroidEngine(n_steps=N_STEPS, k_params=K, C=C, gamma=0.0,
                             beta=beta, start_r=start_r).run()
        cols = data["columns"]
        if not cols:
            hits.append(False); continue
        bits = convert_to_bitsets(cols)
        hits.append(check_fano_minor(bits))
    return float(np.mean(hits))


# ── Collect results ─────────────────────────────────────────────────────────────
print(f"Running Fano threshold experiment  ({REPS} reps per cell) …")
print(f"{'C':<6} {'β':<6} {'start_r':<10} {'P(Fano)':>8}")
print("-" * 35)

# Main: C=0.05
results_main = {}   # (beta, start_r) → P
for beta in BETA_VALUES:
    for sr in START_R_VALS:
        p = p_fano(sr, beta, C_MAIN)
        results_main[(beta, sr)] = p
        print(f"{C_MAIN:<6} {beta:<6.1f} {sr:<10} {p:>8.3f}")

# Alt: C=0.5
print()
results_alt = {}
for beta in BETA_VALUES:
    for sr in START_R_VALS:
        p = p_fano(sr, beta, C_ALT)
        results_alt[(beta, sr)] = p
        print(f"{C_ALT:<6} {beta:<6.1f} {sr:<10} {p:>8.3f}")

# ── Figure ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.5))

# Panel 1: P vs start_r, one curve per β, C=C_MAIN
for beta, color in zip(BETA_VALUES, BETA_COLORS):
    ys = [results_main[(beta, sr)] for sr in START_R_VALS]
    ax1.plot(START_R_VALS, ys, "o-", color=color, lw=2, ms=7, label=f"β={beta}")

ax1.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="P=0.5")
ax1.set_xlabel("Initial row count  start_r", fontsize=11)
ax1.set_ylabel("P(Fano / GF(2) triangle minor)", fontsize=11)
ax1.set_title(f"P(Fano minor) vs start_r\n(C={C_MAIN}, γ=0)",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=9); ax1.set_ylim(-0.03, 1.05); ax1.grid(alpha=0.3)

# Panel 2: P vs β, one curve per start_r, C=C_MAIN
for sr, color in zip(START_R_VALS, SR_COLORS):
    ys = [results_main[(beta, sr)] for beta in BETA_VALUES]
    ax2.plot(BETA_VALUES, ys, "o-", color=color, lw=2, ms=7, label=f"r₀={sr}")

ax2.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
ax2.set_xlabel("Attachment bias  β", fontsize=11)
ax2.set_ylabel("P(Fano / GF(2) triangle minor)", fontsize=11)
ax2.set_title(f"P(Fano minor) vs β\n(C={C_MAIN}, γ=0)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, title="start_r"); ax2.set_ylim(-0.03, 1.05); ax2.grid(alpha=0.3)

# Panel 3: P vs start_r for C=C_ALT (does C shift the threshold?)
for beta, color in zip(BETA_VALUES, BETA_COLORS):
    ys_main = [results_main[(beta, sr)] for sr in START_R_VALS]
    ys_alt  = [results_alt[(beta,  sr)] for sr in START_R_VALS]
    ax3.plot(START_R_VALS, ys_main, "o-",  color=color, lw=2, ms=6,
             label=f"β={beta}, C={C_MAIN}")
    ax3.plot(START_R_VALS, ys_alt,  "s--", color=color, lw=1.5, ms=6,
             label=f"β={beta}, C={C_ALT}", alpha=0.7)

ax3.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
ax3.set_xlabel("Initial row count  start_r", fontsize=11)
ax3.set_ylabel("P(Fano / GF(2) triangle minor)", fontsize=11)
ax3.set_title(f"C={C_MAIN} (solid) vs C={C_ALT} (dashed)\nDoes C shift the threshold?",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=7.5, ncol=2); ax3.set_ylim(-0.03, 1.05); ax3.grid(alpha=0.3)

fig.suptitle(
    f"Fano minor (GF(2) triangle) appearance  (γ=0, k={K}, n_steps={N_STEPS}, {REPS} reps)\n"
    "check_fano_minor = True iff ∃ columns a,b,c with a⊕b=c  (circuit of size ≤ 3)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "fano_threshold.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out}")
