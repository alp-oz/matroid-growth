"""
Experiment 4: Whitney numbers of the flat lattice.

γ=0, λ=0.05, k=4, n_steps=3000, start_r=10
β ∈ {0.5, 1.0, 1.5, 2.0}, 10 replicates.

For the binary matroid [I_r | A], every subset T ⊆ [r] of basis rows defines
a distinct flat:
    F_T = { e_i : i ∈ T } ∪ { non-basis j : support(j) ⊆ T }
with rank(F_T) = |T|.  The lattice of flats is therefore isomorphic to the
Boolean lattice 2^[r], giving Whitney numbers of the second kind:
    w_k = C(r, k)   (number of flats of rank k)

Consequently:
  • The normalised profile w_k / |L(M)| = C(r,k) / 2^r is a Binomial(r, 1/2)
    distribution, symmetric about k* = r/2.
  • β affects only the attachment structure, not the rank r.  Since r is
    determined solely by discovery events (p_row = λ, independent of β),
    Whitney numbers are β-independent.
  • r varies between replicates as r ~ start_r + Binomial(n_steps, λ),
    centred at start_r + λ · n_steps = 10 + 0.05 · 3000 = 160.

Left panel : normalised Whitney number sequence w_k / |L| vs k, one curve per β
             (curves overlap — β-independence visible).
Right panel: peak rank k* = argmax_k w_k vs β (flat — confirms β-independence).

Saves: whitney_numbers.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import os

from core.engine import MatroidEngine

LAMBDA  = 0.05
BETAS   = [0.5, 1.0, 1.5, 2.0]
N_STEPS = 3000
START_R = 10
K       = 4
REPS    = 10
COLORS  = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
OUT     = os.path.dirname(__file__)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

peak_k_per_beta = []

print(f"\n{'β':>5}  {'r_mean':>8}  {'r_std':>7}  {'k*_mean':>8}")

for beta, color in zip(BETAS, COLORS):
    r_vals = []
    for _ in range(REPS):
        eng  = MatroidEngine(n_steps=N_STEPS, k_params=K, C=LAMBDA,
                             gamma=0.0, beta=beta, start_r=START_R)
        data = eng.run()
        r_vals.append(data["r"])

    r_mean = float(np.mean(r_vals))
    r_std  = float(np.std(r_vals))
    r_rep  = int(round(r_mean))   # representative r for plotting

    ks   = np.arange(r_rep + 1)
    wk   = np.array([comb(r_rep, int(k), exact=False) for k in ks])
    wk  /= wk.sum()

    ax1.plot(ks, wk, "-", color=color, lw=1.8,
             label=fr"$\beta={beta}$  ($r\approx{r_rep}$)", alpha=0.85)

    peak_k = [r // 2 for r in r_vals]
    peak_k_per_beta.append(peak_k)
    print(f"{beta:>5.1f}  {r_mean:>8.1f}  {r_std:>7.2f}  {np.mean(peak_k):>8.1f}")

ax1.set_xlabel("Rank $k$", fontsize=12)
ax1.set_ylabel(r"$w_k\,/\,|\mathcal{L}(M_t)|$", fontsize=12)
ax1.set_title(r"Normalised Whitney numbers ($\gamma=0$, $\lambda=0.05$, $k=4$)",
              fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: peak k* vs beta
peak_means = [np.mean(pk) for pk in peak_k_per_beta]
peak_stds  = [np.std(pk)  for pk in peak_k_per_beta]
ax2.errorbar(BETAS, peak_means, yerr=peak_stds, fmt="k-o", lw=2,
             markersize=7, capsize=4, label="mean ± std")
for b, m, c in zip(BETAS, peak_means, COLORS):
    ax2.scatter([b], [m], color=c, zorder=5, s=70)
ax2.set_xlabel(r"$\beta$", fontsize=12)
ax2.set_ylabel(r"Peak rank $k^* = \arg\max_k\, w_k$", fontsize=12)
ax2.set_title("Peak Whitney rank vs PA strength", fontsize=12)
ax2.set_ylim(0, max(peak_means) * 1.2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(OUT, "whitney_numbers.png")
fig.savefig(path, dpi=150)
print(f"\n  → {path}")
print("\nNote: w_k = C(r,k) for all β — lattice of flats ≅ Boolean 2^[r].")
print("      β-independence of Whitney numbers is by construction.")
