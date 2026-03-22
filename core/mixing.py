"""
Mixing time analysis for the circuit Markov chain.

Two methods:
  1. Spectral gap  — from eigenvalues of P
  2. TV distance curve — exact d_TV(t) = max_x ||P^t(x,·) - π||_TV
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.engine import MatroidEngine
from core.markov_chain import fundamental_circuits
from core.circuits import all_circuits
from core.stationary import build_transition_matrix, stationary_distribution


# ─────────────────────────────────────────────────────────────────────────────
# Spectral analysis
# ─────────────────────────────────────────────────────────────────────────────

def spectral_analysis(P):
    """
    Compute eigenvalue spectrum and spectral gap of transition matrix P.

    Returns
    -------
    gap       : spectral gap  1 - |λ₂|
    lambda2   : second largest eigenvalue magnitude
    eigenvals : all eigenvalues sorted by magnitude descending
    t_mix_bound : upper bound on t_mix(0.25) from spectral gap
    """
    eigenvals = np.linalg.eigvals(P)
    mags = np.sort(np.abs(eigenvals))[::-1]   # descending

    lambda1 = mags[0]   # should be ~1
    lambda2 = mags[1]
    gap = 1.0 - lambda2

    # Standard spectral bound: t_mix(ε) ≤ log(1/(ε·π_min)) / gap
    # Using ε=0.25 and a loose π_min=1/N
    N = P.shape[0]
    t_mix_bound = np.log(N / 0.25) / gap if gap > 1e-12 else np.inf

    return {
        "gap":          float(gap),
        "lambda1":      float(lambda1),
        "lambda2":      float(lambda2),
        "eigenvals":    eigenvals,
        "t_mix_bound":  float(t_mix_bound),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Exact TV distance curve
# ─────────────────────────────────────────────────────────────────────────────

def tv_distance_curve(P, pi, max_t=200, epsilon=0.25):
    """
    Compute the worst-case and mean TV distance to π over time.

    d_TV(t) = max_x  (1/2) Σ_y |P^t(x,y) - π(y)|

    Returns
    -------
    tv_max  : array of length max_t, worst-case TV distance at each step
    tv_mean : array of length max_t, mean TV distance over starting states
    t_mix   : first t where tv_max[t] < epsilon  (None if not reached)
    """
    N = len(pi)
    Pt = np.eye(N, dtype=np.float64)   # P^0 = identity
    pi_row = pi[np.newaxis, :]          # (1, N) broadcast

    tv_max  = np.zeros(max_t)
    tv_mean = np.zeros(max_t)
    t_mix   = None

    for t in range(max_t):
        Pt = Pt @ P
        diff = np.abs(Pt - pi_row)       # (N, N)
        tv   = 0.5 * diff.sum(axis=1)   # (N,) TV distance per starting state

        tv_max[t]  = tv.max()
        tv_mean[t] = tv.mean()

        if t_mix is None and tv_max[t] < epsilon:
            t_mix = t + 1   # 1-indexed step count

    return tv_max, tv_mean, t_mix


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline for one matroid
# ─────────────────────────────────────────────────────────────────────────────

def analyse_mixing(params, modes=("global", "adjacent"), max_t=300, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    engine = MatroidEngine(**params)
    result = engine.run()
    M, r, n = result["M"], result["r"], result["n"]

    all_c, _, _ = all_circuits(M, r, mode='global')
    circuits = sorted(all_c, key=sorted)
    N = len(circuits)
    print(f"  rank={r}, #elements={n}, #circuits={N}")

    out = {}
    for mode in modes:
        P  = build_transition_matrix(M, r, circuits, mode=mode)
        pi = stationary_distribution(P)
        sp = spectral_analysis(P)
        tv_max, tv_mean, t_mix = tv_distance_curve(P, pi, max_t=max_t)

        out[mode] = {
            "P": P, "pi": pi,
            "spectral": sp,
            "tv_max": tv_max,
            "tv_mean": tv_mean,
            "t_mix": t_mix,
            "N": N,
        }
        print(f"    [{mode:>8}]  gap={sp['gap']:.4f}  λ₂={sp['lambda2']:.4f}"
              f"  t_mix(0.25)={t_mix}  bound≤{sp['t_mix_bound']:.0f}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"global": "#1f77b4", "adjacent": "#d62728"}
LS     = {"global": "-",       "adjacent": "--"}


def plot_mixing(results_list, labels, out_path="markov-circuits/mixing.png"):
    """
    results_list : list of dicts from analyse_mixing (one per param set)
    labels       : matching list of string labels for the legend
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax_tv   = fig.add_subplot(gs[0, :])   # TV curve — full width
    ax_gap  = fig.add_subplot(gs[1, 0])   # spectral gap bar chart
    ax_eig  = fig.add_subplot(gs[1, 1])   # eigenvalue spectrum

    # ── TV distance curves ────────────────────────────────────────────────────
    for res, label in zip(results_list, labels):
        for mode in ("global", "adjacent"):
            d = res[mode]
            t = np.arange(1, len(d["tv_max"]) + 1)
            ax_tv.plot(t, d["tv_max"], color=COLORS[mode], ls=LS[mode],
                       lw=1.8, alpha=0.85,
                       label=f"{label}  [{mode}]  t_mix={d['t_mix']}")

    ax_tv.axhline(0.25, color="gray", lw=1.2, ls=":", label="ε = 0.25")
    ax_tv.set_xlabel("Steps  t", fontsize=12)
    ax_tv.set_ylabel("Worst-case TV distance", fontsize=12)
    ax_tv.set_title("Total variation distance to stationary distribution", fontsize=13, fontweight='bold')
    ax_tv.legend(fontsize=9, ncol=2)
    ax_tv.set_ylim(0, 1.05)
    ax_tv.grid(alpha=0.3)

    # ── Spectral gap bar chart ─────────────────────────────────────────────────
    x      = np.arange(len(labels))
    width  = 0.35
    gaps_g = [r["global"]["spectral"]["gap"]   for r in results_list]
    gaps_a = [r["adjacent"]["spectral"]["gap"] for r in results_list]

    ax_gap.bar(x - width/2, gaps_g, width, label="global",   color=COLORS["global"],   alpha=0.8)
    ax_gap.bar(x + width/2, gaps_a, width, label="adjacent", color=COLORS["adjacent"], alpha=0.8)
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(labels, fontsize=10)
    ax_gap.set_ylabel("Spectral gap  1 − |λ₂|", fontsize=11)
    ax_gap.set_title("Spectral gap by parameter set", fontsize=12, fontweight='bold')
    ax_gap.legend(fontsize=10)
    ax_gap.grid(axis='y', alpha=0.3)

    # ── Eigenvalue spectrum (last param set, both modes) ──────────────────────
    res = results_list[-1]
    label = labels[-1]
    for mode in ("global", "adjacent"):
        eigs = np.sort(np.abs(res[mode]["spectral"]["eigenvals"]))[::-1]
        ax_eig.plot(eigs, color=COLORS[mode], ls=LS[mode], lw=1.8,
                    label=f"{mode}  (gap={res[mode]['spectral']['gap']:.4f})")

    ax_eig.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax_eig.set_xlabel("Eigenvalue rank", fontsize=11)
    ax_eig.set_ylabel("|λ|", fontsize=11)
    ax_eig.set_title(f"Eigenvalue spectrum  ({label})", fontsize=12, fontweight='bold')
    ax_eig.legend(fontsize=10)
    ax_eig.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main: compare beta values (tractable circuit counts)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = dict(n_steps=20, k_params=2, C=0.1, gamma=0.0, start_r=2)

    configs = [
        ({**BASE, "beta": 0.0}, "β=0.0"),
        ({**BASE, "beta": 0.8}, "β=0.8"),
        ({**BASE, "beta": 1.2}, "β=1.2"),
    ]

    print("Mixing time analysis\n")
    results_list = []
    labels       = []
    for params, label in configs:
        print(f"  {label}")
        res = analyse_mixing(params, max_t=300)
        results_list.append(res)
        labels.append(label)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'label':>8}  {'mode':>10}  {'gap':>8}  {'λ₂':>8}  "
          f"{'t_mix(0.25)':>12}  {'bound':>8}")
    print("-" * 62)
    for res, label in zip(results_list, labels):
        for mode in ("global", "adjacent"):
            sp = res[mode]["spectral"]
            tm = res[mode]["t_mix"]
            print(f"{label:>8}  {mode:>10}  {sp['gap']:>8.4f}  "
                  f"{sp['lambda2']:>8.4f}  {str(tm):>12}  "
                  f"{sp['t_mix_bound']:>8.0f}")

    plot_mixing(results_list, labels)
