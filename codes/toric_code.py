"""
Toric code as a binary matroid — Markov chain study.

The L×L toric code has:
  - 2L² qubits on torus edges  (L² horizontal + L² vertical)
  - L² X-type plaquette stabilizers, each acting on exactly 4 qubits

Matroid M[H_X]:
  - Ground set = 2L² qubits (columns of H_X)
  - Independent sets = linearly independent column subsets over GF(2)
  - Circuits = minimal linearly dependent column subsets
             = supports of minimum-weight codewords of the code ker(H_X)

Key facts:
  - rank(H_X) = L² - 1   (sum of all rows = 0 mod 2)
  - Minimum circuit size ≥ L  (a loop around the torus)
  - Compare to PA matroids where circuits can be very small (size ~ r+1)

The comparison illuminates the code quality ↔ chain regularity connection:
  - Better codes (large min distance, uniform circuit sizes) ↔ nearly uniform π
  - Worse codes (small circuits) ↔ highly non-uniform π biased toward small circuits
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.markov_chain import fundamental_circuits, decompose_into_circuits
from core.circuits import all_circuits
from core.stationary import build_transition_matrix, stationary_distribution
from core.mixing import spectral_analysis, tv_distance_curve
from core.engine import MatroidEngine


# ─────────────────────────────────────────────────────────────────────────────
# Toric code construction
# ─────────────────────────────────────────────────────────────────────────────

def build_toric_hx(L):
    """
    Build the X-stabilizer parity check matrix H_X for the L×L toric code.
    Returns H_X of shape (L², 2L²) over GF(2).

    Qubit labeling:
      horizontal edge (i,j): index i*L + j             for i,j ∈ {0,..,L-1}
      vertical   edge (i,j): index L² + i*L + j

    Plaquette (i,j) acts on:
      H(i,j),  H((i+1)%L, j),  V(i,j),  V(i, (j+1)%L)
    """
    n_qubits = 2 * L * L
    n_plaq   = L * L
    H = np.zeros((n_plaq, n_qubits), dtype=np.uint8)

    for i in range(L):
        for j in range(L):
            p = i * L + j  # plaquette index
            H[p, i * L + j]                        = 1  # H(i,j)
            H[p, ((i + 1) % L) * L + j]            = 1  # H((i+1)%L, j)
            H[p, L*L + i * L + j]                  = 1  # V(i,j)
            H[p, L*L + i * L + (j + 1) % L]        = 1  # V(i,(j+1)%L)

    return H


def build_toric_hz(L):
    """
    Build the Z-stabilizer parity check matrix H_Z for the L×L toric code.
    Returns H_Z of shape (L², 2L²) over GF(2).

    Same qubit labeling as build_toric_hx.

    Vertex (i,j) acts on the 4 edges meeting it:
      H(i,j),  H(i, (j-1)%L),  V(i,j),  V((i-1)%L, j)
    """
    n_qubits = 2 * L * L
    n_vert   = L * L
    H = np.zeros((n_vert, n_qubits), dtype=np.uint8)

    for i in range(L):
        for j in range(L):
            v = i * L + j  # vertex index
            H[v, i * L + j]                          = 1  # H(i,j)
            H[v, i * L + (j - 1) % L]                = 1  # H(i,(j-1)%L)
            H[v, L*L + i * L + j]                    = 1  # V(i,j)
            H[v, L*L + ((i - 1) % L) * L + j]        = 1  # V((i-1)%L,j)

    return H


def gf2_row_reduce(H):
    """
    GF(2) row reduction with column tracking.
    Returns (H_rref, pivot_cols, rank).
    """
    H = H.copy().astype(np.uint8)
    m, n = H.shape
    pivot_cols = []
    row = 0
    for col in range(n):
        pivot = None
        for rr in range(row, m):
            if H[rr, col]:
                pivot = rr
                break
        if pivot is None:
            continue
        H[[row, pivot]] = H[[pivot, row]]
        pivot_cols.append(col)
        for rr in range(m):
            if rr != row and H[rr, col]:
                H[rr] = (H[rr] + H[row]) % 2
        row += 1
    return H[:row], pivot_cols, row


def toric_to_matroid(L):
    """
    Convert L×L toric code H_X to binary matroid in standard [I | A] form.

    Returns
    -------
    M          : r × n  float64 matrix in [I | A] form
    r          : rank
    n          : number of columns (= 2L²)
    pivot_cols : original H_X column indices used as basis
    free_cols  : remaining column indices
    """
    H = build_toric_hx(L)
    m, n = H.shape

    H_rref, pivot_cols, r = gf2_row_reduce(H)

    pivot_set = set(pivot_cols)
    free_cols = [j for j in range(n) if j not in pivot_set]

    # Reorder: basis columns first, then free columns
    col_order = pivot_cols + free_cols
    M = H_rref[:, col_order].astype(np.float64)

    # Verify left block is identity
    assert np.allclose(M[:r, :r], np.eye(r)), "Left block not identity after reduction"

    return M, r, n, pivot_cols, free_cols


# ─────────────────────────────────────────────────────────────────────────────
# Single-instance analysis
# ─────────────────────────────────────────────────────────────────────────────

def circuit_count_only(L, max_circuits=50000, seed=42):
    """Count circuits and get size distribution without running the chain."""
    random.seed(seed); np.random.seed(seed)
    M, r, n, pivot_cols, free_cols = toric_to_matroid(L)
    all_c, trunc, _ = all_circuits(M, r, mode='global', max_circuits=max_circuits)
    N = len(all_c)
    sizes = np.array([len(c) for c in all_c]) if all_c else np.array([])
    return {
        "L": L, "r": r, "n": n, "N": N, "trunc": trunc,
        "sizes": sizes,
        "min_sz": int(sizes.min()) if N > 0 else None,
        "max_sz": int(sizes.max()) if N > 0 else None,
    }


def analyse_toric(L, max_t=500, max_circuits=5000, seed=42):
    random.seed(seed); np.random.seed(seed)

    M, r, n, pivot_cols, free_cols = toric_to_matroid(L)

    all_c, trunc, _ = all_circuits(M, r, mode='global', max_circuits=max_circuits)
    if trunc:
        return None

    circuits = sorted(all_c, key=sorted)
    N = len(circuits)
    if N == 0:
        return None

    sizes = np.array([len(c) for c in circuits])

    P_adj = build_transition_matrix(M, r, circuits, mode='adjacent')
    pi    = stationary_distribution(P_adj)
    sp    = spectral_analysis(P_adj)
    tv, _, tmix = tv_distance_curve(P_adj, pi, max_t=max_t)

    pi_uni = np.ones(N) / N
    l1_adj = float(np.sum(np.abs(pi - pi_uni)))

    return {
        "L": L, "r": r, "n": n, "N": N,
        "sizes": sizes,
        "min_sz": int(sizes.min()), "max_sz": int(sizes.max()),
        "gap": sp["gap"], "tmix": tmix,
        "pi": pi, "tv": tv,
        "l1": l1_adj,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)

    # ── 1. Toric code sweep: chain analysis for small L, count-only for larger ──
    print("Toric code — binary matroid analysis")
    print("=" * 80)
    print(f"{'L':>4}  {'r=L²-1':>7}  {'n=2L²':>6}  {'N circuits':>12}  "
          f"{'min|C|':>7}  {'max|C|':>7}  {'gap':>8}  "
          f"{'t_mix':>7}  {'L1 to unif':>11}")
    print("=" * 80)

    toric_rows = []
    for L in [2, 3, 4, 5, 6]:
        if L <= 3:
            res = analyse_toric(L)
            if res:
                toric_rows.append(res)
                tmix_s = str(res["tmix"]) if res["tmix"] else ">500"
                print(f"  {L:>3}  {res['r']:>7}  {res['n']:>6}  {res['N']:>12}  "
                      f"{res['min_sz']:>7}  {res['max_sz']:>7}  {res['gap']:>8.4f}  "
                      f"{tmix_s:>7}  {res['l1']:>11.4f}")
        else:
            # Circuit count only — chain matrix too large
            cnt_res = circuit_count_only(L)
            tag = "(all)" if not cnt_res["trunc"] else "(cap)"
            print(f"  {L:>3}  {cnt_res['r']:>7}  {cnt_res['n']:>6}  "
                  f"{cnt_res['N']:>10}{tag}  "
                  f"{cnt_res['min_sz']:>7}  {cnt_res['max_sz']:>7}"
                  f"   [chain too large to diagonalise]")

    # ── 2. Circuit size distribution for chain-analysed L ────────────────────
    for res in toric_rows:
        print(f"\nCircuit size distribution (L={res['L']}, r={res['r']}, N={res['N']}):")
        print(f"  {'size':>5}  {'count':>8}  {'fraction':>9}")
        for sz in sorted(set(res["sizes"].tolist())):
            cnt = int(np.sum(res["sizes"] == sz))
            print(f"  {sz:>5}  {cnt:>8}  {cnt/res['N']:>9.3f}")

    # ── 2b. Circuit size distribution for L=4 (count-only) ───────────────────
    print(f"\nCircuit size distribution (L=4, full enumeration, chain skipped):")
    cnt4 = circuit_count_only(4)
    print(f"  r={cnt4['r']}, n={cnt4['n']}, N={cnt4['N']}")
    print(f"  {'size':>5}  {'count':>8}  {'fraction':>9}")
    for sz in sorted(set(cnt4["sizes"].tolist())):
        cnt = int(np.sum(cnt4["sizes"] == sz))
        print(f"  {sz:>5}  {cnt:>8}  {cnt/cnt4['N']:>9.3f}")

    print("\nNote on circuit structure:")
    print("  L≥4: min|C|=4 = vertex stabilizers (star operators act on 4 edges).")
    print("  All sizes even — toric lattice is bipartite.")
    print("  Toric min distance (logical loops) = L, but circuits include stabilizers.")

    # ── 3. PA matroids for comparison ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PA matroids — same chain analysis (for comparison)")
    print("=" * 72)
    print(f"{'n_steps':>8}  {'r':>5}  {'N':>8}  "
          f"{'min|C|':>7}  {'max|C|':>7}  {'gap':>8}  "
          f"{'t_mix':>7}  {'L1 to unif':>11}")
    print("=" * 72)

    BASE = dict(k_params=2, C=0.1, gamma=0.0, beta=0.8, start_r=2)
    pa_steps = [10, 14, 18, 22, 26, 30]
    pa_rows = []

    for ns in pa_steps:
        random.seed(42); np.random.seed(42)
        engine = MatroidEngine(**{**BASE, "n_steps": ns})
        result = engine.run()
        M_pa, r_pa, n_pa = result["M"], result["r"], result["n"]

        all_c, trunc, _ = all_circuits(M_pa, r_pa, mode='global')
        if trunc:
            print(f"  n_steps={ns}: truncated")
            continue

        circuits_pa = sorted(all_c, key=sorted)
        N_pa = len(circuits_pa)
        if N_pa == 0:
            continue

        sizes_pa = np.array([len(c) for c in circuits_pa])
        P_pa     = build_transition_matrix(M_pa, r_pa, circuits_pa, mode='adjacent')
        pi_pa    = stationary_distribution(P_pa)
        sp_pa    = spectral_analysis(P_pa)
        tv_pa, _, tmix_pa = tv_distance_curve(P_pa, pi_pa, max_t=500)
        l1_pa    = float(np.sum(np.abs(pi_pa - np.ones(N_pa)/N_pa)))

        tmix_s = str(tmix_pa) if tmix_pa else ">500"
        print(f"  {ns:>7}  {r_pa:>5}  {N_pa:>8}  "
              f"{int(sizes_pa.min()):>7}  {int(sizes_pa.max()):>7}  "
              f"{sp_pa['gap']:>8.4f}  {tmix_s:>7}  {l1_pa:>11.4f}")

        pa_rows.append({
            "ns": ns, "r": r_pa, "N": N_pa,
            "sizes": sizes_pa, "gap": sp_pa["gap"],
            "tmix": tmix_pa, "pi": pi_pa, "tv": tv_pa, "l1": l1_pa,
        })

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # Panel 1: Circuit size histogram — toric vs PA (last instances)
    ax1 = fig.add_subplot(gs[0, :])
    if toric_rows:
        last_t = toric_rows[-1]
        ax1.hist(last_t["sizes"], bins=range(last_t["min_sz"], last_t["max_sz"] + 2),
                 alpha=0.6, color='#1f77b4', edgecolor='white',
                 label=f"Toric L={last_t['L']} (N={last_t['N']}, min|C|={last_t['min_sz']})")
    if pa_rows:
        last_pa = pa_rows[-1]
        ax1.hist(last_pa["sizes"],
                 bins=range(int(last_pa["sizes"].min()), int(last_pa["sizes"].max()) + 2),
                 alpha=0.6, color='#d62728', edgecolor='white',
                 label=f"PA n_steps={last_pa['ns']} (N={last_pa['N']}, min|C|={int(last_pa['sizes'].min())})")
    ax1.set_xlabel("Circuit size |C|", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Circuit size distribution: toric code vs PA matroid",
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Spectral gap comparison
    ax2 = fig.add_subplot(gs[1, 0])
    if toric_rows:
        Ls   = [r["L"]   for r in toric_rows]
        gaps_t = [r["gap"] for r in toric_rows]
        ax2.plot(Ls, gaps_t, 'o-', color='#1f77b4', lw=2, ms=8, label='Toric code')
    if pa_rows:
        logN_pa = np.log([r["N"] for r in pa_rows])
        gaps_pa = [r["gap"] for r in pa_rows]
        ax2b = ax2.twiny()
        ax2b.plot(logN_pa, gaps_pa, 's--', color='#d62728', lw=2, ms=7, label='PA matroid')
        ax2b.set_xlabel("log(N)  [PA]", fontsize=10, color='#d62728')
        ax2b.tick_params(axis='x', labelcolor='#d62728')
        ax2b.legend(loc='upper right', fontsize=9)
    ax2.set_xlabel("L  [Toric]", fontsize=11, color='#1f77b4')
    ax2.set_ylabel("Spectral gap", fontsize=11)
    ax2.set_title("Spectral gap", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9); ax2.grid(alpha=0.3)

    # Panel 3: L1 distance to uniform
    ax3 = fig.add_subplot(gs[1, 1])
    if toric_rows:
        Ns_t = [r["N"]  for r in toric_rows]
        l1s_t = [r["l1"] for r in toric_rows]
        ax3.plot(np.log(Ns_t), l1s_t, 'o-', color='#1f77b4', lw=2, ms=8, label='Toric code')
    if pa_rows:
        logN_pa = np.log([r["N"] for r in pa_rows])
        l1s_pa  = [r["l1"] for r in pa_rows]
        ax3.plot(logN_pa, l1s_pa, 's--', color='#d62728', lw=2, ms=7, label='PA matroid')
    ax3.axhline(0, color='gray', lw=1, ls=':')
    ax3.set_xlabel("log(N)", fontsize=11)
    ax3.set_ylabel("L1 distance to uniform", fontsize=11)
    ax3.set_title("Stationarity bias", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    # Panel 4: TV curves — toric largest vs PA largest
    ax4 = fig.add_subplot(gs[2, 0])
    t_ax = np.arange(1, 501)
    if toric_rows:
        last_t = toric_rows[-1]
        tv_t = last_t["tv"]
        ax4.plot(t_ax[:len(tv_t)], tv_t, color='#1f77b4', lw=2,
                 label=f"Toric L={last_t['L']}  t_mix={last_t['tmix']}")
    if pa_rows:
        last_pa = pa_rows[-1]
        tv_p = last_pa["tv"]
        ax4.plot(t_ax[:len(tv_p)], tv_p, color='#d62728', lw=2, ls='--',
                 label=f"PA n_steps={last_pa['ns']}  t_mix={last_pa['tmix']}")
    ax4.axhline(0.25, color='gray', lw=1, ls=':', label='ε=0.25')
    ax4.set_xlim(0, 100); ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Steps t", fontsize=11)
    ax4.set_ylabel("Worst-case TV distance", fontsize=11)
    ax4.set_title("TV mixing curves", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    # Panel 5: π vs circuit size scatter — toric
    ax5 = fig.add_subplot(gs[2, 1])
    if toric_rows:
        last_t = toric_rows[-1]
        jitter = np.random.uniform(-0.08, 0.08, len(last_t["sizes"]))
        ax5.scatter(last_t["sizes"] + jitter, last_t["pi"],
                    s=10, alpha=0.4, color='#1f77b4',
                    label=f'Toric L={last_t["L"]}')
        ax5.axhline(1.0 / last_t["N"], color='#1f77b4', lw=1.5, ls='--',
                    label=f'uniform 1/N (toric)')
    if pa_rows:
        last_pa = pa_rows[-1]
        jitter2 = np.random.uniform(-0.08, 0.08, len(last_pa["sizes"]))
        ax5.scatter(last_pa["sizes"] + jitter2, last_pa["pi"],
                    s=10, alpha=0.3, color='#d62728',
                    label=f'PA n_steps={last_pa["ns"]}')
        ax5.axhline(1.0 / last_pa["N"], color='#d62728', lw=1.5, ls='--',
                    label=f'uniform 1/N (PA)')
    ax5.set_xlabel("Circuit size |C|", fontsize=11)
    ax5.set_ylabel("π(C)", fontsize=11)
    ax5.set_title("Stationary distribution vs size", fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8, ncol=2); ax5.grid(alpha=0.3)

    fig.suptitle("Toric code vs PA matroid — Markov chain on circuits",
                 fontsize=14, fontweight='bold')
    out = "markov-circuits/toric_code.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out}")
    plt.close()
