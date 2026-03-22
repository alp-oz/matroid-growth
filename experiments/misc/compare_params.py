"""
Run the full analysis pipeline for a list of MatroidEngine parameter sets
and print a consolidated comparison table.
"""
import numpy as np
import random
from scipy.stats import spearmanr

from core.engine import MatroidEngine
from core.markov_chain import fundamental_circuits, decompose_into_circuits
from core.circuits import all_circuits
from core.stationary import build_transition_matrix, stationary_distribution, circuit_features


FEAT_CIRCUIT_LIMIT = 2000   # skip expensive feature loop above this count

def run_one(params, seed=42):
    """Run the full pipeline for one parameter set. Returns a result dict."""
    random.seed(seed)
    np.random.seed(seed)

    engine = MatroidEngine(**params)
    result = engine.run()
    M, r, n = result["M"], result["r"], result["n"]

    all_c, truncated, _ = all_circuits(M, r, mode='global')
    if truncated:
        print(f"  WARNING: circuit BFS truncated for {params}")
    circuits = sorted(all_c, key=sorted)

    P_adj = build_transition_matrix(M, r, circuits, mode='adjacent')
    pi    = stationary_distribution(P_adj)

    # Feature correlations — skip for large circuit sets (too slow)
    skip_feats = len(circuits) > FEAT_CIRCUIT_LIMIT
    if not skip_feats:
        feats = circuit_features(M, r, circuits)
        sizes = feats["size"].astype(int)
        rho_size, _     = spearmanr(sizes, pi)
        rho_nonbasis, _ = spearmanr(feats["n_nonbasis"], pi)
        rho_deg, _      = spearmanr(feats["degree_adjacent"], pi)
        unique_sizes    = sorted(set(sizes))
        med_by_size     = {sz: np.median(pi[sizes == sz]) for sz in unique_sizes}
    else:
        # Fast size computation only (no decomposition calls)
        sizes        = np.array([len(c) for c in circuits])
        rho_size, _  = spearmanr(sizes, pi)
        rho_nonbasis = float('nan')
        rho_deg      = float('nan')
        unique_sizes = sorted(set(sizes.tolist()))
        med_by_size  = {sz: np.median(pi[sizes == sz]) for sz in unique_sizes}

    return {
        "params":        params,
        "r":             r,
        "n":             n,
        "n_circuits":    len(circuits),
        "truncated":     truncated,
        "skip_feats":    skip_feats,
        "pi":            pi,
        "sizes":         sizes,
        "entropy":       float(-np.sum(pi * np.log(pi + 1e-300))),
        "max_entropy":   float(np.log(len(circuits))),
        "pi_max":        float(pi.max()),
        "pi_min":        float(pi.min()),
        "rho_size":      float(rho_size),
        "rho_nonbasis":  float(rho_nonbasis),
        "rho_deg_adj":   float(rho_deg),
        "med_by_size":   med_by_size,
        "unique_sizes":  unique_sizes,
    }


def print_comparison(results):
    # ── Main summary table ────────────────────────────────────────────────────
    cols = ["beta", "k", "C", "rank r", "#elem", "#circuits",
            "entropy", "max H", "π max", "π min",
            "ρ(size)", "ρ(n_nb)", "ρ(deg_adj)"]

    header = f"{'beta':>5} {'k':>10} {'C':>5} {'r':>6} {'n':>6} {'#circ':>7}  " \
             f"{'H':>6} {'Hmax':>6}  {'π max':>8} {'π min':>10}  " \
             f"{'ρ size':>7} {'ρ n_nb':>7} {'ρ deg':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for r in results:
        p = r["params"]
        trunc = "†" if r["truncated"] else " "
        kl = r.get("k_label", k_label(p["k_params"]))
        print(
            f"{p['beta']:>5.2f} "
            f"{kl:>10} "
            f"{p['C']:>5.3f} "
            f"{r['r']:>6} "
            f"{r['n']:>6} "
            f"{r['n_circuits']:>6}{trunc}  "
            f"{r['entropy']:>6.3f} "
            f"{r['max_entropy']:>6.3f}  "
            f"{r['pi_max']:>8.5f} "
            f"{r['pi_min']:>10.7f}  "
            f"{r['rho_size']:>7.3f} "
            f"{r['rho_nonbasis']:>7.3f} "
            f"{r['rho_deg_adj']:>7.3f}"
        )

    print("=" * len(header))
    print("† = BFS circuit count truncated at cap\n")

    # ── Median π by circuit size ──────────────────────────────────────────────
    all_sizes = sorted({sz for r in results for sz in r["unique_sizes"]})
    print("Median π (adjacent) by circuit size:")
    size_header = f"{'beta':>5} {'k':>10}  " + "  ".join(f"sz={s:>2}" for s in all_sizes)
    print(size_header)
    print("-" * len(size_header))
    for r in results:
        p = r["params"]
        kl = r.get("k_label", k_label(p["k_params"]))
        row = f"{p['beta']:>5.2f} {kl:>10}  "
        for sz in all_sizes:
            med = r["med_by_size"].get(sz)
            if med is not None:
                row += f"{med:>8.5f}  "
            else:
                row += f"{'—':>8}  "
        print(row)


def k_label(k_params):
    """Human-readable label for k_params."""
    if isinstance(k_params, tuple) and k_params[0] == "poisson":
        return f"Pois({k_params[1]})"
    return str(k_params)


# ── Parameter grid ────────────────────────────────────────────────────────────
BASE = dict(n_steps=20, C=0.1, gamma=0.0, beta=0.8, start_r=2)

param_sets = [
    {**BASE, "k_params": 2},
    {**BASE, "k_params": 3},
    {**BASE, "k_params": 4},
    {**BASE, "k_params": ("poisson", 4)},
]

if __name__ == "__main__":
    print("Running pipeline for each parameter set (adjacent mode)...\n")
    results = []
    for params in param_sets:
        kl = k_label(params['k_params'])
        print(f"  beta={params['beta']:.1f}, k={kl}, C={params['C']} ...", end=" ", flush=True)
        res = run_one(params)
        res["k_label"] = kl
        print(f"rank={res['r']}, #circuits={res['n_circuits']}")
        results.append(res)

    print_comparison(results)
