"""
XOR circuit chain mixing time τ vs matroid size n for fixed β=0.5 (below β*).

Parameters: γ=0, λ=0.05, α=0.3, β=0.5, start_r=10.
n ∈ {20,30,40,50,60,80}: number of attached PA columns (= chain state space size).
For each n: generate PA matroid with exactly n PA columns, compute spectral gap
δ of XOR chain within largest non-overlap component, report τ = 1/δ.
Average over REPS replicates.

Saves: experiments/circuit_chain/pa_xor_scaling.json  (cached)
       figures/xor_scaling.png
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ── Parameters ─────────────────────────────────────────────────────────────────
ALPHA   = 0.3
LAMBDA  = 0.05
GAMMA   = 0.0
BETA    = 0.5
START_R = 10
N_LIST  = [20, 30, 40, 50, 60, 80]   # target number of PA columns
REPS    = 30                           # replicates per n

OUT_DIR = os.path.dirname(__file__)
CACHE   = os.path.join(OUT_DIR, "pa_xor_scaling.json")
OUT_FIG = os.path.join(OUT_DIR, "..", "..", "figures", "xor_scaling.png")

PARAMS_KEY = dict(alpha=ALPHA, lam=LAMBDA, gamma=GAMMA, beta=BETA,
                  start_r=START_R, n_list=N_LIST, reps=REPS)


# ── PA growth until exactly n_target PA columns ────────────────────────────────
def run_pa_fixed_n(n_target, alpha, lam, gamma, beta, start_r, seed):
    rng       = np.random.default_rng(seed)
    curr_r    = start_r
    row_usage = np.ones(start_r, dtype=np.float64)
    supports  = []
    t = 0
    while len(supports) < n_target:
        t += 1
        p_row = min(1.0, lam * (t ** (-gamma)))
        if rng.random() < p_row:
            curr_r += 1
            row_usage = np.append(row_usage, 1.0)
        else:
            k   = max(1, min(int(np.floor(alpha * curr_r)), curr_r))
            w   = row_usage ** beta;  w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            supports.append(frozenset(int(x) for x in sel))
            for idx in sel:
                row_usage[int(idx)] += 1
    return supports, curr_r


# ── Spectral gap of XOR chain within largest non-overlap component ──────────────
def xor_tau(supports, r_final):
    n_att = len(supports)
    if n_att < 3:
        return float("nan"), float("nan"), n_att

    B = np.zeros((n_att, r_final), dtype=np.float32)
    for j, supp in enumerate(supports):
        for i in supp:
            B[j, i] = 1.0

    G         = (B @ B.T).round().astype(np.int32)
    nonoverlap = (G == 0)
    np.fill_diagonal(nonoverlap, False)

    n_comp, labels = connected_components(csr_matrix(nonoverlap), directed=False)
    comp_sizes     = np.bincount(labels)
    largest_comp   = int(np.argmax(comp_sizes))
    frac_largest   = float(comp_sizes[largest_comp]) / n_att

    mask  = labels == largest_comp
    idx   = np.where(mask)[0]
    n_sub = len(idx)
    if n_sub < 3:
        return float("nan"), frac_largest, n_att

    no_sub  = nonoverlap[np.ix_(idx, idx)]
    deg_sub = no_sub.sum(axis=1).astype(np.float64)

    P_sub = np.zeros((n_sub, n_sub), dtype=np.float64)
    P_sub[no_sub] = 1.0 / (2.0 * n_att)
    np.fill_diagonal(P_sub, 1.0 - deg_sub / (2.0 * n_att))

    eigvals = np.linalg.eigvalsh(P_sub)
    lam2    = float(eigvals[-2])
    delta   = 1.0 - lam2
    if delta < 1e-14:
        return float("nan"), frac_largest, n_att

    return 1.0 / delta, frac_largest, n_att   # τ = 1/δ


# ── Load or compute ────────────────────────────────────────────────────────────
blob = None
if os.path.exists(CACHE):
    with open(CACHE) as f:
        blob = json.load(f)
    if blob.get("params_key") != PARAMS_KEY:
        print("Cache differs — recomputing.")
        blob = None
    else:
        print(f"Loaded cache from {CACHE}")

if blob is None:
    print(f"Running XOR chain scaling  (β={BETA}, {REPS} reps per n) …")
    records = []
    for n in N_LIST:
        taus, fracs = [], []
        for rep in range(REPS):
            seed = int(BETA * 1e5) + n * 1000 + rep
            sup, rf = run_pa_fixed_n(n, ALPHA, LAMBDA, GAMMA, BETA, START_R, seed)
            tau, frac, _ = xor_tau(sup, rf)
            if not np.isnan(tau):
                taus.append(tau); fracs.append(frac)
        tau_mean = float(np.mean(taus)) if taus else float("nan")
        tau_std  = float(np.std(taus))  if taus else float("nan")
        records.append(dict(n=n, tau_mean=tau_mean, tau_std=tau_std,
                            frac_largest_mean=float(np.mean(fracs)) if fracs else float("nan"),
                            n_valid=len(taus)))
        print(f"  n={n:3d}  τ={tau_mean:.2f} ± {tau_std:.2f}"
              f"  (frac_largest={np.mean(fracs):.2f}, valid={len(taus)}/{REPS})")

    blob = {"params_key": PARAMS_KEY, "results": records}
    with open(CACHE, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"Cached → {CACHE}")

# ── Power-law fit τ ~ n^γ ──────────────────────────────────────────────────────
records = blob["results"]
ns      = np.array([r["n"]        for r in records], dtype=float)
taus    = np.array([r["tau_mean"] for r in records], dtype=float)
errs    = np.array([r["tau_std"]  for r in records], dtype=float)
valid   = ~np.isnan(taus)

slope, intercept, r_val, *_ = linregress(np.log(ns[valid]), np.log(taus[valid]))
gamma_fit = slope
C_fit     = np.exp(intercept)
print(f"\nPower-law fit: τ ~ {C_fit:.3f} · n^{gamma_fit:.3f}  "
      f"(R²={r_val**2:.3f}, γ {'<' if gamma_fit < 1 else '≥'} 1)")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))

# Reference lines anchored at geometric midpoint
n_mid   = np.exp(np.mean(np.log(ns[valid])))
tau_mid = C_fit * n_mid ** gamma_fit
n_ref   = np.linspace(ns[valid].min() * 0.85, ns[valid].max() * 1.2, 200)

for exp, ls, lbl in [(0.5, ":", r"$\sim n^{0.5}$"),
                     (1.0, "--", r"$\sim n^{1}$")]:
    c = tau_mid / n_mid ** exp
    ax.plot(n_ref, c * n_ref**exp, ls, color="#aaaaaa", lw=1.0, zorder=1, label=lbl)

# Data with error bars
ax.errorbar(ns[valid], taus[valid], yerr=errs[valid],
            fmt="o", color="#d73027", markersize=6,
            capsize=3, lw=1.2, elinewidth=0.9,
            markeredgecolor="white", markeredgewidth=0.5,
            zorder=3, label=fr"XOR chain ($\beta={BETA}$)")

# Fitted power law
ax.plot(n_ref, C_fit * n_ref**gamma_fit, "-", color="black", lw=1.8, zorder=2,
        label=fr"fit: $\tau \sim n^{{{gamma_fit:.2f}}}$")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"PA columns $n$", fontsize=12)
ax.set_ylabel(r"Mixing time $\tau = 1/\delta$", fontsize=12)
ax.set_title(fr"XOR chain mixing time vs matroid size"
             fr"  ($\beta={BETA},\ \alpha={ALPHA},\ \lambda={LAMBDA}$)",
             fontsize=10)
ax.legend(fontsize=9.5, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, which="both", lw=0.3, alpha=0.4)
ax.tick_params(labelsize=9)
plt.tight_layout()

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_FIG}")
