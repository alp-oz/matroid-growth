"""
Empirical scaling study: how does mixing time grow as the matroid grows?

We vary n_steps (matroid size) and for each instance measure:
  - #circuits N
  - spectral gap of adjacent chain
  - spectral gap of MH chain (estimated from TV curve)
  - t_mix(0.25) for adjacent and MH chains

If t_mix(MH) = O(log N) or O(poly(rank)) → evidence for rapid mixing.
If t_mix(MH) = O(N^alpha) → slow mixing, MH not asymptotically viable.
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress

from core.engine import MatroidEngine
from core.markov_chain import MHCircuitChain, fundamental_circuits, decompose_into_circuits
from core.circuits import all_circuits
from core.stationary import build_transition_matrix, stationary_distribution
from core.mixing import spectral_analysis, tv_distance_curve


def build_mh_transition_matrix(M, r, circuits):
    """
    Build the exact transition matrix for the MH chain (alpha=0, uniform target).
    Entry P[i,j] = probability of moving from circuits[i] to circuits[j].
    """
    fc        = fundamental_circuits(M, r)
    non_basis = list(range(r, M.shape[1]))
    idx       = {c: i for i, c in enumerate(circuits)}
    N         = len(circuits)
    P         = np.zeros((N, N), dtype=np.float64)

    for i, C in enumerate(circuits):
        eligible_C = [j for j in non_basis if fc[j] & C]
        if not eligible_C:
            P[i, i] = 1.0
            continue

        n_elig_C = len(eligible_C)
        total_weight = 0.0
        proposals = {}   # (j, C_prop) -> unnormalised weight

        for j in eligible_C:
            sym_fwd = C ^ fc[j]
            if not sym_fwd:
                # Stay: add to self-loop
                proposals[(j, C)] = proposals.get((j, C), 0) + 1.0 / n_elig_C
                total_weight += 1.0 / n_elig_C
                continue

            parts_fwd = decompose_into_circuits(M, sym_fwd)
            if not parts_fwd:
                proposals[(j, C)] = proposals.get((j, C), 0) + 1.0 / n_elig_C
                total_weight += 1.0 / n_elig_C
                continue

            n_parts_fwd = len(parts_fwd)
            q_fwd = 1.0 / (n_elig_C * n_parts_fwd)

            for C_prop in parts_fwd:
                # Check reverse feasibility
                if not (fc[j] & C_prop):
                    # j not eligible from C_prop → reject, stay
                    proposals[(j, C)] = proposals.get((j, C), 0) + q_fwd
                    total_weight += q_fwd
                    continue

                sym_rev = C_prop ^ fc[j]
                if not sym_rev:
                    proposals[(j, C)] = proposals.get((j, C), 0) + q_fwd
                    total_weight += q_fwd
                    continue

                parts_rev = decompose_into_circuits(M, sym_rev)
                if C not in parts_rev:
                    proposals[(j, C)] = proposals.get((j, C), 0) + q_fwd
                    total_weight += q_fwd
                    continue

                eligible_prop = [jj for jj in non_basis if fc[jj] & C_prop]
                n_parts_rev   = len(parts_rev)
                q_rev = 1.0 / (len(eligible_prop) * n_parts_rev)

                ratio = q_rev / q_fwd   # uniform target: w(C')=w(C)=1
                alpha = min(1.0, ratio)

                move_prob = q_fwd * alpha
                stay_prob = q_fwd * (1.0 - alpha)

                k = idx.get(C_prop)
                if k is not None:
                    P[i, k] += move_prob
                P[i, i]   += stay_prob
                total_weight += q_fwd

        # Normalise (should already sum to 1)

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P /= row_sums
    return P


def run_scaling(params, max_t=500, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    engine = MatroidEngine(**params)
    result = engine.run()
    M, r, n = result["M"], result["r"], result["n"]

    all_c, truncated, _ = all_circuits(M, r, mode='global')
    if truncated:
        return None   # skip truncated cases
    circuits = sorted(all_c, key=sorted)
    N = len(circuits)

    # Adjacent chain
    P_adj  = build_transition_matrix(M, r, circuits, mode='adjacent')
    pi_adj = stationary_distribution(P_adj)
    sp_adj = spectral_analysis(P_adj)
    tv_adj, _, tmix_adj = tv_distance_curve(P_adj, pi_adj, max_t=max_t)

    # MH chain
    P_mh  = build_mh_transition_matrix(M, r, circuits)
    pi_mh = stationary_distribution(P_mh)
    sp_mh = spectral_analysis(P_mh)
    tv_mh, _, tmix_mh = tv_distance_curve(P_mh, pi_mh, max_t=max_t)

    return {
        "r": r, "n": n, "N": N,
        "gap_adj": sp_adj["gap"], "tmix_adj": tmix_adj,
        "gap_mh":  sp_mh["gap"],  "tmix_mh":  tmix_mh,
        "tv_adj": tv_adj, "tv_mh": tv_mh,
    }


# ── Parameter sweep: grow matroid by increasing n_steps ──────────────────────
if __name__ == "__main__":
    BASE   = dict(k_params=2, C=0.1, gamma=0.0, beta=0.8, start_r=2)
    steps  = [10, 14, 18, 22, 26, 30]

    print(f"{'n_steps':>8} {'rank':>6} {'#circ N':>9}  "
          f"{'gap_adj':>9} {'tmix_adj':>10}  "
          f"{'gap_mh':>8} {'tmix_mh':>9}")
    print("-" * 70)

    rows = []
    for ns in steps:
        params = {**BASE, "n_steps": ns}
        res = run_scaling(params)
        if res is None:
            print(f"{ns:>8}  (truncated — too many circuits)")
            continue
        rows.append(res)
        tmix_mh_str  = str(res["tmix_mh"])  if res["tmix_mh"]  else ">500"
        tmix_adj_str = str(res["tmix_adj"]) if res["tmix_adj"] else ">500"
        print(f"{ns:>8} {res['r']:>6} {res['N']:>9}  "
              f"{res['gap_adj']:>9.4f} {tmix_adj_str:>10}  "
              f"{res['gap_mh']:>8.4f} {tmix_mh_str:>9}")

    if not rows:
        print("No results.")
        exit()

    # ── Scaling plots ─────────────────────────────────────────────────────────
    Ns    = np.array([r["N"]        for r in rows])
    ranks = np.array([r["r"]        for r in rows])
    g_adj = np.array([r["gap_adj"]  for r in rows])
    g_mh  = np.array([r["gap_mh"]   for r in rows])
    tm_adj = np.array([r["tmix_adj"] if r["tmix_adj"] else 500 for r in rows])
    tm_mh  = np.array([r["tmix_mh"]  if r["tmix_mh"]  else 500 for r in rows])

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── 1. Spectral gap vs rank ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ranks, g_adj, 'o-', color='#1f77b4', lw=2, ms=7, label='adjacent')
    ax1.plot(ranks, g_mh,  's-', color='#d62728', lw=2, ms=7, label='MH (uniform)')
    ax1.set_xlabel("Rank r", fontsize=11)
    ax1.set_ylabel("Spectral gap  1−|λ₂|", fontsize=11)
    ax1.set_title("Spectral gap vs rank", fontsize=12, fontweight='bold')
    ax1.legend(); ax1.grid(alpha=0.3)

    # ── 2. t_mix vs log(#circuits) ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    logN = np.log(Ns)
    ax2.plot(logN, tm_adj, 'o-', color='#1f77b4', lw=2, ms=7, label='adjacent')
    ax2.plot(logN, tm_mh,  's-', color='#d62728', lw=2, ms=7, label='MH (uniform)')

    # Fit log trend for MH
    if len(rows) >= 3:
        slope, intercept, rv, _, _ = linregress(logN, tm_mh)
        xfit = np.linspace(logN.min(), logN.max(), 100)
        ax2.plot(xfit, slope * xfit + intercept, '--', color='#d62728', alpha=0.5,
                 label=f'linear fit (slope={slope:.2f})')

    ax2.set_xlabel("log(#circuits)", fontsize=11)
    ax2.set_ylabel("t_mix(0.25)", fontsize=11)
    ax2.set_title("Mixing time vs log(#circuits)\n(linear here = O(log N))",
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # ── 3. Gap vs log(N) ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(logN, g_adj, 'o-', color='#1f77b4', lw=2, ms=7, label='adjacent')
    ax3.plot(logN, g_mh,  's-', color='#d62728', lw=2, ms=7, label='MH')
    ax3.set_xlabel("log(#circuits)", fontsize=11)
    ax3.set_ylabel("Spectral gap", fontsize=11)
    ax3.set_title("Does gap stay bounded away from 0?", fontsize=12, fontweight='bold')
    ax3.legend(); ax3.grid(alpha=0.3)

    # ── 4. TV curves for smallest and largest instance ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    t_axis = np.arange(1, 501)
    for i, (row, ls) in enumerate([(rows[0], '-'), (rows[-1], '--')]):
        label_a = f"adjacent N={row['N']}"
        label_m = f"MH N={row['N']}"
        tv_a = row["tv_adj"]
        tv_m = row["tv_mh"]
        ax4.plot(t_axis[:len(tv_a)], tv_a, color='#1f77b4', ls=ls, lw=1.8, label=label_a)
        ax4.plot(t_axis[:len(tv_m)], tv_m, color='#d62728', ls=ls, lw=1.8, label=label_m)

    ax4.axhline(0.25, color='gray', lw=1, ls=':', label='ε=0.25')
    ax4.set_xlim(0, 60); ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Steps t", fontsize=11)
    ax4.set_ylabel("Worst-case TV distance", fontsize=11)
    ax4.set_title("TV curves: smallest vs largest matroid", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, ncol=2); ax4.grid(alpha=0.3)

    fig.suptitle("Asymptotic mixing: adjacent vs MH chain", fontsize=14, fontweight='bold')
    out = "markov-circuits/asymptotic_mixing.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out}")
    plt.close()
