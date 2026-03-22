"""
Markov chain on circuits: comparison across QECC families.

For each code we build the binary matroid M[H_X], enumerate circuits,
run the adjacent chain, and record:
  - N  : number of circuits
  - min|C| : smallest circuit (lower bound on d)
  - gap    : spectral gap
  - t_mix  : mixing time to TV < 0.25
  - L1     : L1 distance of π from uniform

Code families (all n ≈ 12–22 to keep N tractable):
  1. Weight-2 random LDPC   — bad codes, d = 2 with high probability
  2. Toric code L=2,3       — d = L, all-even circuits
  3. Bicycle codes           — CSS quasi-cyclic, d grows better than toric
  4. Weight-3 random LDPC   — asymptotically good family

Key question: does better code quality (larger min|C|/n) correlate with
larger spectral gap and smaller L1?
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from codes.toric_code import toric_to_matroid, gf2_row_reduce
from core.circuits import all_circuits
from core.stationary import build_transition_matrix, stationary_distribution
from core.mixing import spectral_analysis, tv_distance_curve


# ─────────────────────────────────────────────────────────────────────────────
# Code constructors  →  each returns an r×n float64 matrix in [I|A] form
# ─────────────────────────────────────────────────────────────────────────────

def h_to_matroid(H):
    """Row-reduce H (GF2) and return (M, r, n) in standard [I|A] form."""
    m, n = H.shape
    H_rref, pivot_cols, r = gf2_row_reduce(H)
    pivot_set  = set(pivot_cols)
    free_cols  = [j for j in range(n) if j not in pivot_set]
    col_order  = pivot_cols + free_cols
    M = H_rref[:, col_order].astype(np.float64)
    assert np.allclose(M[:r, :r], np.eye(r))
    return M, r, n


def random_ldpc_biregular(n, col_weight, row_weight, seed):
    """
    Random (col_weight, row_weight)-biregular LDPC parity check matrix.
    n  : number of qubits (columns).  Must have col_weight*n % row_weight == 0.
    Each column has exactly col_weight ones; each row has exactly row_weight ones.
    Returns H of shape (m, n) where m = col_weight * n // row_weight.

    Built by the standard permutation method:
      - Create col_weight copies of [0,..,n-1], concatenate → length T = col_weight*n
      - Shuffle, split into m groups of row_weight → m rows.
      - Retry if any row has a repeated column index (parallel edges).
    """
    assert (col_weight * n) % row_weight == 0
    m   = col_weight * n // row_weight
    rng = np.random.default_rng(seed)

    for attempt in range(200):
        arr = np.tile(np.arange(n), col_weight)
        rng.shuffle(arr)
        rows = arr.reshape(m, row_weight)
        # Check for duplicate columns within any row
        if all(len(set(row)) == row_weight for row in rows):
            H = np.zeros((m, n), dtype=np.uint8)
            for i, row in enumerate(rows):
                H[i, row] = 1
            return H
    # Fallback: return best attempt even with duplicates
    H = np.zeros((m, n), dtype=np.uint8)
    for i, row in enumerate(rows):
        H[i, row] = (H[i, row] + 1) % 2  # XOR to handle duplicates
    return H


def bicycle_code(r, a_exps, b_exps):
    """
    H_X = [A | B]  (r × 2r over GF2)
    A = Σ_{a in a_exps} P^a,   B = Σ_{b in b_exps} P^b
    P = r×r cyclic shift.
    """
    def circulant(exps):
        C = np.zeros((r, r), dtype=np.uint8)
        for e in exps:
            for i in range(r):
                C[i, (i + e) % r] = (C[i, (i + e) % r] + 1) % 2
        return C

    A = circulant(a_exps)
    B = circulant(b_exps)
    return np.hstack([A, B])


# ─────────────────────────────────────────────────────────────────────────────
# Single-code analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_code(label, M, r, n, max_circuits=6000, max_t=500, seed=42,
                 max_t_large=100, large_threshold=600):
    random.seed(seed); np.random.seed(seed)

    all_c, trunc, _ = all_circuits(M, r, mode='global', max_circuits=max_circuits)
    N = len(all_c)
    if trunc or N == 0:
        return None

    circuits = sorted(all_c, key=sorted)
    if len(circuits) < 2:
        return None
    sizes    = np.array([len(c) for c in circuits])

    actual_max_t = max_t_large if N >= large_threshold else max_t

    P    = build_transition_matrix(M, r, circuits, mode='adjacent')
    try:
        pi = stationary_distribution(P)
    except np.linalg.LinAlgError:
        # Chain is reducible (disconnected circuit graph) — use lstsq fallback
        N2 = P.shape[0]
        A  = (P.T - np.eye(N2)).astype(np.float64)
        A[-1, :] = 1.0
        b  = np.zeros(N2); b[-1] = 1.0
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.clip(pi, 0, None)
        s  = pi.sum()
        if s == 0: return None
        pi /= s
    sp   = spectral_analysis(P)
    tv, _, tmix = tv_distance_curve(P, pi, max_t=actual_max_t)

    pi_uni = np.ones(N) / N
    l1     = float(np.sum(np.abs(pi - pi_uni)))

    return {
        "label": label,
        "r": r, "n": n, "N": N,
        "sizes": sizes,
        "min_sz":  int(sizes.min()),
        "max_sz":  int(sizes.max()),
        "rel_min": float(sizes.min()) / n,   # min|C| / n  (normalised distance)
        "gap":  sp["gap"],
        "tmix": tmix,
        "pi":   pi,
        "tv":   tv,
        "l1":   l1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build all code instances
# ─────────────────────────────────────────────────────────────────────────────

def build_all_instances():
    instances = []

    # ── 1. Weight-2 biregular LDPC (bad codes, d≈2) ──────────────────────────
    # col_weight=2, row_weight=4: n=16 qubits, m=8 checks, each qubit in 2 checks
    for seed in range(3):
        H = random_ldpc_biregular(n=16, col_weight=2, row_weight=4, seed=seed + 10)
        M, r, n_ = h_to_matroid(H)
        label = f"w2-LDPC n=16 s={seed}"
        instances.append((label, M, r, n_, "w2-LDPC"))

    # ── 2. Toric code ─────────────────────────────────────────────────────────
    for L in [2, 3]:
        M, r, n, _, _ = toric_to_matroid(L)
        instances.append((f"Toric L={L}", M, r, n, "Toric"))

    # ── 3. Bicycle codes ──────────────────────────────────────────────────────
    # row weight 2: H = [I+P | I+P²]  →  each row has weight 4
    H = bicycle_code(6, [0, 1], [0, 2])
    M, r, n = h_to_matroid(H)
    instances.append(("Bicycle r=6 [0,1|0,2]", M, r, n, "Bicycle"))

    # row weight 3: H = [I+P+P² | I+P²+P³]
    H = bicycle_code(6, [0, 1, 2], [0, 2, 3])
    M, r, n = h_to_matroid(H)
    instances.append(("Bicycle r=6 w3", M, r, n, "Bicycle"))

    H = bicycle_code(9, [0, 1], [0, 4])
    M, r, n = h_to_matroid(H)
    instances.append(("Bicycle r=9 [0,1|0,4]", M, r, n, "Bicycle"))

    H = bicycle_code(9, [0, 1, 3], [0, 2, 5])
    M, r, n = h_to_matroid(H)
    instances.append(("Bicycle r=9 w3", M, r, n, "Bicycle"))

    # ── 4. Larger bicycle codes (better quality, higher min|C|/n) ────────────
    # r=12, n=24 (weight-2 rows only — N=1405, tractable)
    H = bicycle_code(12, [0, 1], [0, 5])
    M, r, n = h_to_matroid(H)
    instances.append(("Bicycle r=12 [0,1|0,5]", M, r, n, "Bicycle"))

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_COLORS = {
    "w2-LDPC":  "#d62728",   # red
    "Toric":    "#1f77b4",   # blue
    "Bicycle":  "#2ca02c",   # green
    "w3-LDPC":  "#9467bd",   # purple
}

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)

    instances = build_all_instances()

    print(f"{'label':<30}  {'r':>4}  {'n':>4}  {'N':>6}  "
          f"{'min|C|':>7}  {'min/n':>6}  {'gap':>8}  "
          f"{'t_mix':>7}  {'L1':>7}")
    print("-" * 90)

    results = []
    for label, M, r, n, family in instances:
        res = analyse_code(label, M, r, n)
        if res is None:
            print(f"  {label:<28}  — skipped (truncated or empty)")
            continue
        res["family"] = family
        results.append(res)
        tmix_s = str(res["tmix"]) if res["tmix"] else ">500"
        print(f"  {label:<28}  {r:>4}  {n:>4}  {res['N']:>6}  "
              f"{res['min_sz']:>7}  {res['rel_min']:>6.3f}  "
              f"{res['gap']:>8.4f}  {tmix_s:>7}  {res['l1']:>7.4f}")

    if not results:
        print("No results."); exit()

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    def scatter_by_family(ax, xs, ys, labels, families, xlabel, ylabel, title):
        seen = set()
        for x, y, lab, fam in zip(xs, ys, labels, families):
            col  = FAMILY_COLORS.get(fam, "gray")
            mkr  = {"w2-LDPC": "x", "Toric": "o",
                    "Bicycle": "s", "w3-LDPC": "^"}.get(fam, "D")
            lbl  = fam if fam not in seen else None
            ax.scatter(x, y, color=col, marker=mkr, s=80, zorder=3, label=lbl)
            seen.add(fam)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    rel_mins = [r["rel_min"] for r in results]
    gaps     = [r["gap"]     for r in results]
    l1s      = [r["l1"]      for r in results]
    tmixs    = [r["tmix"] if r["tmix"] else 500 for r in results]
    labels   = [r["label"]   for r in results]
    families = [r["family"]  for r in results]
    Ns       = [r["N"]       for r in results]

    # 1. Spectral gap vs min|C|/n
    ax1 = fig.add_subplot(gs[0, 0])
    scatter_by_family(ax1, rel_mins, gaps, labels, families,
                      "min|C| / n   (normalised distance lower bound)",
                      "Spectral gap",
                      "Gap vs code quality")

    # 2. L1 to uniform vs min|C|/n
    ax2 = fig.add_subplot(gs[0, 1])
    scatter_by_family(ax2, rel_mins, l1s, labels, families,
                      "min|C| / n",
                      "L1 distance to uniform",
                      "Stationarity bias vs code quality")

    # 3. t_mix vs min|C|/n
    ax3 = fig.add_subplot(gs[1, 0])
    scatter_by_family(ax3, rel_mins, tmixs, labels, families,
                      "min|C| / n",
                      "t_mix(0.25)",
                      "Mixing time vs code quality")

    # 4. TV curves — one per family (pick representative)
    ax4 = fig.add_subplot(gs[1, 1])
    t_ax = np.arange(1, 501)
    plotted = set()
    for res in results:
        fam = res["family"]
        if fam in plotted:
            continue
        col = FAMILY_COLORS.get(fam, "gray")
        tv  = res["tv"]
        tmix_s = str(res["tmix"]) if res["tmix"] else ">500"
        ax4.plot(t_ax[:len(tv)], tv, color=col, lw=2,
                 label=f"{fam}  (t_mix={tmix_s})")
        plotted.add(fam)
    ax4.axhline(0.25, color="gray", lw=1, ls=":", label="ε=0.25")
    ax4.set_xlim(0, 80); ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Steps t", fontsize=11)
    ax4.set_ylabel("Worst-case TV distance", fontsize=11)
    ax4.set_title("TV curves (one per family)", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    fig.suptitle("Markov chain on QECC circuits — gap, mixing, bias vs code quality",
                 fontsize=13, fontweight='bold')
    out = "markov-circuits/qecc_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out}")
    plt.close()

    # ── Summary: does better code → better mixing? ────────────────────────────
    print("\nSummary: sorted by min|C|/n  (ascending = worse codes first)")
    print(f"{'family':<12}  {'label':<30}  {'min/n':>6}  "
          f"{'gap':>8}  {'t_mix':>7}  {'L1':>7}")
    print("-" * 75)
    for res in sorted(results, key=lambda r: r["rel_min"]):
        tmix_s = str(res["tmix"]) if res["tmix"] else ">500"
        print(f"  {res['family']:<10}  {res['label']:<30}  "
              f"{res['rel_min']:>6.3f}  {res['gap']:>8.4f}  "
              f"{tmix_s:>7}  {res['l1']:>7.4f}")
