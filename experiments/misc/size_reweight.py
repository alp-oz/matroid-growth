"""
Check whether π(C) is well-determined by |C| alone.

If yes: run adjacent chain, reweight by 1/π̂(|C|) → effectively uniform,
        with no MH overhead and no mixing penalty.

Steps:
  1. Fit π vs size (exponential model: π(C) ≈ a * exp(-b*|C|))
  2. Measure within-size variance (CV per size class)
  3. Compare L1 to uniform: raw adjacent vs size-reweighted vs MH
  4. Repeat across matroid sizes to check if size-explanation holds asymptotically
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from core.engine import MatroidEngine
from core.markov_chain import fundamental_circuits
from core.circuits import all_circuits
from core.stationary import build_transition_matrix, stationary_distribution


def exponential_model(size, a, b):
    return a * np.exp(-b * size)


def analyse_size_bias(params, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    engine = MatroidEngine(**params)
    result = engine.run()
    M, r, n = result["M"], result["r"], result["n"]

    all_c, trunc, _ = all_circuits(M, r, mode='global')
    if trunc:
        return None
    circuits = sorted(all_c, key=sorted)
    N = len(circuits)

    P_adj  = build_transition_matrix(M, r, circuits, mode='adjacent')
    pi     = stationary_distribution(P_adj)
    sizes  = np.array([len(c) for c in circuits])
    pi_uni = np.ones(N) / N

    # ── 1. Fit exponential model ──────────────────────────────────────────────
    try:
        popt, _ = curve_fit(exponential_model, sizes, pi,
                            p0=[pi.max(), 0.5], maxfev=5000)
        pi_fit = exponential_model(sizes, *popt)
        pi_fit = np.clip(pi_fit, 1e-12, None)
        pi_fit /= pi_fit.sum()
    except Exception:
        pi_fit = None
        popt   = (None, None)

    # ── 2. Size-class statistics ──────────────────────────────────────────────
    unique_sizes = sorted(set(sizes))
    size_stats   = {}
    for sz in unique_sizes:
        mask = sizes == sz
        vals = pi[mask]
        size_stats[sz] = {
            "count":  int(mask.sum()),
            "mean":   float(vals.mean()),
            "std":    float(vals.std()),
            "cv":     float(vals.std() / vals.mean()) if vals.mean() > 0 else 0,
            "min":    float(vals.min()),
            "max":    float(vals.max()),
        }

    # ── 3. Size-reweighted distribution ──────────────────────────────────────
    # Weight each circuit by 1 / mean_π(|C|), then normalise
    mean_pi_by_size = np.array([size_stats[sz]["mean"] for sz in sizes])
    weights         = 1.0 / mean_pi_by_size
    pi_reweighted   = pi * weights
    pi_reweighted  /= pi_reweighted.sum()

    # ── 4. R² of size as predictor of log(π) ─────────────────────────────────
    log_pi   = np.log(pi + 1e-300)
    rho, _   = spearmanr(sizes, pi)
    ss_tot   = np.var(log_pi)
    # Mean log_pi per size as prediction
    pred_log = np.array([np.mean(log_pi[sizes == sz]) for sz in sizes])
    ss_res   = np.mean((log_pi - pred_log) ** 2)
    r2_size  = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # ── 5. L1 distances ───────────────────────────────────────────────────────
    l1_adj    = np.sum(np.abs(pi           - pi_uni))
    l1_rew    = np.sum(np.abs(pi_reweighted - pi_uni))
    l1_fit    = np.sum(np.abs(pi_fit        - pi_uni)) if pi_fit is not None else None

    return {
        "N": N, "r": r,
        "pi": pi, "sizes": sizes,
        "pi_reweighted": pi_reweighted,
        "pi_fit": pi_fit,
        "fit_params": popt,
        "size_stats": size_stats,
        "unique_sizes": unique_sizes,
        "rho": float(rho),
        "r2_size": float(r2_size),
        "l1_adj": float(l1_adj),
        "l1_rew": float(l1_rew),
        "l1_fit": float(l1_fit) if l1_fit is not None else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE  = dict(k_params=2, C=0.1, gamma=0.0, beta=0.8, start_r=2)
    steps = [10, 14, 18, 22, 26, 30]

    print(f"{'n_steps':>8} {'N':>6} {'r':>4}  "
          f"{'ρ(size,π)':>10} {'R²(log π|size)':>15}  "
          f"{'L1 adj':>8} {'L1 reweighted':>14} {'reduction':>10}")
    print("-" * 80)

    all_res = []
    for ns in steps:
        res = analyse_size_bias({**BASE, "n_steps": ns})
        if res is None:
            continue
        all_res.append((ns, res))
        red = 1 - res["l1_rew"] / res["l1_adj"]
        print(f"{ns:>8} {res['N']:>6} {res['r']:>4}  "
              f"{res['rho']:>10.4f} {res['r2_size']:>15.4f}  "
              f"{res['l1_adj']:>8.4f} {res['l1_rew']:>14.4f} {red:>9.1%}")

    # ── Within-size CV table for last (largest) instance ─────────────────────
    _, res_large = all_res[-1]
    print(f"\nWithin-size statistics (n_steps={steps[-1]}, N={res_large['N']}):")
    print(f"  {'size':>5}  {'count':>6}  {'mean π':>10}  {'std π':>10}  {'CV':>8}  {'min/mean':>9}  {'max/mean':>9}")
    print("  " + "-" * 60)
    for sz in res_large["unique_sizes"]:
        s = res_large["size_stats"][sz]
        if s["mean"] > 0:
            print(f"  {sz:>5}  {s['count']:>6}  {s['mean']:>10.6f}  "
                  f"{s['std']:>10.6f}  {s['cv']:>8.3f}  "
                  f"{s['min']/s['mean']:>9.3f}  {s['max']/s['mean']:>9.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # 1. π vs size with fit for largest instance
    ax1 = fig.add_subplot(gs[0, :])
    _, res = all_res[-1]
    sizes_arr = res["sizes"]
    pi_arr    = res["pi"]
    jitter    = np.random.uniform(-0.15, 0.15, len(sizes_arr))

    ax1.scatter(sizes_arr + jitter, pi_arr, s=15, alpha=0.4,
                color='#1f77b4', label='π(C) individual', zorder=2)

    # Mean per size
    for sz in res["unique_sizes"]:
        mask = sizes_arr == sz
        ax1.plot([sz - 0.25, sz + 0.25],
                 [res["size_stats"][sz]["mean"]] * 2,
                 color='#1f77b4', lw=2.5, zorder=3)

    # Exponential fit
    if res["pi_fit"] is not None:
        sz_range = np.linspace(sizes_arr.min(), sizes_arr.max(), 200)
        ax1.plot(sz_range,
                 exponential_model(sz_range, *res["fit_params"]) / res["pi_fit"].sum() * 1,
                 'r--', lw=2,
                 label=f'exp fit  a={res["fit_params"][0]:.4f}, b={res["fit_params"][1]:.3f}')

    ax1.set_yscale('log')
    ax1.set_xlabel("Circuit size |C|", fontsize=12)
    ax1.set_ylabel("π(C)  (log scale)", fontsize=12)
    ax1.set_title(f"π vs circuit size  (N={res['N']})  —  "
                  f"R²(log π | size) = {res['r2_size']:.4f}", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(True, which='both', alpha=0.3)

    # 2. L1 reduction across matroid sizes
    ax2 = fig.add_subplot(gs[1, 0])
    ns_vals  = [ns for ns, _ in all_res]
    N_vals   = [r["N"]     for _, r in all_res]
    l1_a     = [r["l1_adj"] for _, r in all_res]
    l1_r     = [r["l1_rew"] for _, r in all_res]
    logN     = np.log(N_vals)

    ax2.plot(logN, l1_a, 'o-', color='#d62728', lw=2, ms=7, label='adjacent (raw)')
    ax2.plot(logN, l1_r, 's-', color='#2ca02c', lw=2, ms=7, label='size-reweighted')
    ax2.axhline(0, color='gray', lw=1, ls=':')
    ax2.set_xlabel("log(#circuits)", fontsize=11)
    ax2.set_ylabel("L1 distance to uniform", fontsize=11)
    ax2.set_title("L1 to uniform: raw vs reweighted", fontsize=12, fontweight='bold')
    ax2.legend(); ax2.grid(alpha=0.3)

    # 3. R² and CV across sizes
    ax3 = fig.add_subplot(gs[1, 1])
    r2_vals = [r["r2_size"] for _, r in all_res]
    # Mean CV across size classes
    mean_cv = [np.mean([r["size_stats"][sz]["cv"]
                        for sz in r["unique_sizes"]]) for _, r in all_res]

    ax3_twin = ax3.twinx()
    ax3.plot(logN, r2_vals,  'o-', color='#9467bd', lw=2, ms=7, label='R²(log π | size)')
    ax3_twin.plot(logN, mean_cv, 's--', color='#ff7f0e', lw=2, ms=7, label='mean within-size CV')

    ax3.set_xlabel("log(#circuits)", fontsize=11)
    ax3.set_ylabel("R²", fontsize=11, color='#9467bd')
    ax3_twin.set_ylabel("Mean CV within size class", fontsize=11, color='#ff7f0e')
    ax3.set_title("How well does size predict π?", fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left',  fontsize=9)
    ax3_twin.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)

    fig.suptitle("Size as predictor of π — is reweighting sufficient?",
                 fontsize=14, fontweight='bold')
    out = "markov-circuits/size_reweight.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out}")
    plt.close()
