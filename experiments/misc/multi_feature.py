"""
Fit log π(C) on multiple circuit features and check if the residual L1
to uniform vanishes — making size-based reweighting unnecessary.

Model: log π(C) = β₀ + β₁·size + β₂·n_nonbasis + β₃·log(deg_adj) + ...
       fitted by OLS on log scale.

Reweighting: w(C) = 1/π̂(C),  π_rw(C) ∝ π(C)/π̂(C).
If π̂ ≈ π everywhere → π_rw ≈ uniform.

We use leave-one-out cross-validation to avoid overfitting on small instances.
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.engine import MatroidEngine
from core.markov_chain import fundamental_circuits, decompose_into_circuits
from core.circuits import all_circuits
from core.stationary import build_transition_matrix, stationary_distribution, circuit_features


# ─────────────────────────────────────────────────────────────────────────────
# OLS on log π  (no sklearn needed)
# ─────────────────────────────────────────────────────────────────────────────

def fit_log_model(X, log_pi):
    """
    OLS: log_pi = X β.
    X should include a column of ones for intercept.
    Returns β, R², residuals.
    """
    beta, res, rank, sv = np.linalg.lstsq(X, log_pi, rcond=None)
    pred    = X @ beta
    ss_tot  = np.var(log_pi) * len(log_pi)
    ss_res  = np.sum((log_pi - pred) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta, float(r2), pred


def reweight_to_uniform(pi, log_pi_pred):
    """Given fitted log π̂, reweight π(C)/π̂(C) → near uniform."""
    pi_hat     = np.exp(log_pi_pred)
    pi_hat    /= pi_hat.sum()
    pi_rw      = pi / (pi_hat + 1e-300)
    pi_rw      = np.clip(pi_rw, 0, None)
    pi_rw     /= pi_rw.sum()
    return pi_rw


def loo_r2(X, log_pi):
    """Leave-one-out R² to guard against overfitting."""
    n    = len(log_pi)
    pred = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        beta, _, _ = fit_log_model(X[mask], log_pi[mask])
        pred[i]    = X[i] @ beta
    ss_tot = np.var(log_pi) * n
    ss_res = np.sum((log_pi - pred) ** 2)
    return float(1 - ss_res / ss_tot)


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix builder
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SETS = {
    "size only":            ["size"],
    "size + n_nb":          ["size", "n_nonbasis"],
    "size + deg":           ["size", "deg_adj_log"],
    "size + n_nb + deg":    ["size", "n_nonbasis", "deg_adj_log"],
    "all":                  ["size", "n_nonbasis", "n_basis",
                             "deg_adj_log", "n_elig_adj"],
}


def build_feature_matrix(feats, feature_names):
    cols = []
    for name in feature_names:
        if name == "deg_adj_log":
            cols.append(np.log(feats["degree_adjacent"] + 1))
        elif name == "n_basis":
            cols.append(feats["size"] - feats["n_nonbasis"])
        elif name == "n_elig_adj":
            cols.append(feats["n_eligible_adjacent"])
        else:
            cols.append(feats[name])
    X = np.column_stack(cols)
    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis for one matroid
# ─────────────────────────────────────────────────────────────────────────────

def analyse(params, seed=42):
    random.seed(seed); np.random.seed(seed)

    engine = MatroidEngine(**params)
    result = engine.run()
    M, r, n = result["M"], result["r"], result["n"]

    all_c, trunc, _ = all_circuits(M, r, mode='global')
    if trunc: return None
    circuits = sorted(all_c, key=sorted)
    N        = len(circuits)

    P_adj = build_transition_matrix(M, r, circuits, mode='adjacent')
    pi    = stationary_distribution(P_adj)
    pi_uni = np.ones(N) / N
    log_pi = np.log(pi + 1e-300)

    feats = circuit_features(M, r, circuits)

    results = {"N": N, "r": r, "pi": pi, "feats": feats}

    print(f"\n  N={N}, rank={r}")
    print(f"  {'Feature set':<28}  {'R²(in-sample)':>14}  "
          f"{'R²(LOO)':>9}  {'L1 reweighted':>14}  {'reduction':>10}")
    print("  " + "-" * 80)

    for fname, flist in FEATURE_SETS.items():
        X          = build_feature_matrix(feats, flist)
        beta, r2, pred = fit_log_model(X, log_pi)
        r2_loo     = loo_r2(X, log_pi) if N <= 500 else float('nan')
        pi_rw      = reweight_to_uniform(pi, pred)
        l1_rw      = float(np.sum(np.abs(pi_rw - pi_uni)))
        l1_raw     = float(np.sum(np.abs(pi - pi_uni)))
        reduction  = 1 - l1_rw / l1_raw

        results[fname] = {
            "beta": beta, "r2": r2, "r2_loo": r2_loo,
            "l1_rw": l1_rw, "reduction": reduction,
            "pred": pred, "pi_rw": pi_rw,
        }

        r2_loo_str = f"{r2_loo:.4f}" if not np.isnan(r2_loo) else "  n/a "
        print(f"  {fname:<28}  {r2:>14.4f}  "
              f"{r2_loo_str:>9}  {l1_rw:>14.4f}  {reduction:>9.1%}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE  = dict(k_params=2, C=0.1, gamma=0.0, beta=0.8, start_r=2)
    steps = [10, 14, 18, 22, 26, 30]

    all_res = []
    for ns in steps:
        print(f"\nn_steps={ns}")
        res = analyse({**BASE, "n_steps": ns})
        if res:
            all_res.append((ns, res))

    # ── Scaling plot: L1 reduction vs log N for each feature set ─────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    logN   = np.array([np.log(r["N"]) for _, r in all_res])
    colors = ['#aec7e8', '#1f77b4', '#ffbb78', '#d62728', '#9467bd']

    # 1. L1 to uniform vs log N for all feature sets
    ax1 = fig.add_subplot(gs[0, :])
    for (fname, col) in zip(FEATURE_SETS.keys(), colors):
        l1s = [r[fname]["l1_rw"] for _, r in all_res]
        ax1.plot(logN, l1s, 'o-', color=col, lw=2, ms=7, label=fname)
    # Raw adjacent
    l1_raw = [np.sum(np.abs(r["pi"] - np.ones(r["N"])/r["N"])) for _, r in all_res]
    ax1.plot(logN, l1_raw, 's--', color='black', lw=2, ms=7, label='adjacent (no reweighting)')
    ax1.axhline(0, color='gray', lw=1, ls=':')
    ax1.set_xlabel("log(#circuits)", fontsize=12)
    ax1.set_ylabel("L1 distance to uniform", fontsize=12)
    ax1.set_title("Residual L1 after reweighting — does adding features help?",
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, ncol=3); ax1.grid(alpha=0.3)

    # 2. R² (LOO) vs log N
    ax2 = fig.add_subplot(gs[1, 0])
    for (fname, col) in zip(FEATURE_SETS.keys(), colors):
        r2s = [r[fname]["r2_loo"] if not np.isnan(r[fname]["r2_loo"]) else None
               for _, r in all_res]
        valid = [(lN, rv) for lN, rv in zip(logN, r2s) if rv is not None]
        if valid:
            xs, ys = zip(*valid)
            ax2.plot(xs, ys, 'o-', color=col, lw=2, ms=7, label=fname)
    ax2.set_xlabel("log(#circuits)", fontsize=11)
    ax2.set_ylabel("R² (leave-one-out)", fontsize=11)
    ax2.set_title("Predictive R² (LOO) vs matroid size", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # 3. Residual scatter for largest instance — best vs worst model
    ax3 = fig.add_subplot(gs[1, 1])
    _, res_large = all_res[-1]
    sizes_arr = res_large["feats"]["size"].astype(int)
    log_pi    = np.log(res_large["pi"] + 1e-300)

    pred_size = res_large["size only"]["pred"]
    pred_all  = res_large["all"]["pred"]

    ax3.scatter(pred_size, log_pi - pred_size, s=15, alpha=0.5,
                color='#aec7e8', label=f'size only  R²={res_large["size only"]["r2"]:.3f}')
    ax3.scatter(pred_all,  log_pi - pred_all,  s=15, alpha=0.5,
                color='#9467bd', label=f'all features  R²={res_large["all"]["r2"]:.3f}')
    ax3.axhline(0, color='gray', lw=1, ls='--')
    ax3.set_xlabel("Predicted log π", fontsize=11)
    ax3.set_ylabel("Residual  (log π − predicted)", fontsize=11)
    ax3.set_title(f"Residuals: size-only vs all features  (N={res_large['N']})",
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    out = "markov-circuits/multi_feature.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out}")
    plt.close()
