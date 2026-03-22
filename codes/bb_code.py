"""
Balanced Product (BP) / Bivariate Bicycle codes as binary matroids.

Construction (Breuckmann & Eberhardt 2021, IBM 2023):
  Given two polynomials  A(x,y), B(x,y)  in  Z_2[x,y] / (x^l - 1, y^m - 1):

    H_X = [ A | B ]        shape: (l·m) × (2·l·m)
    H_Z = [ B^T | A^T ]    shape: (l·m) × (2·l·m)

  CSS condition: H_X · H_Z^T = A·B^T + B·A^T = 0  (mod 2)
  — this holds when A and B are chosen so that a(x,y)·b̄(x,y) = b(x,y)·ā(x,y)
    in Z_2[x,y]/(x^l-1, y^m-1), where ā(x,y) = a(x^{-1}, y^{-1}).

  Parameters:
    n  = 2·l·m   (physical qubits)
    k  = 2·(k_A + k_B - l·m)  or computed from rank
    d  ≥ min(d_A, d_B)  (lower bound)

Relation to other constructions:
  - Bicycle code: l=1 or m=1  (1D cyclic group)
  - Toric code:   A = 1+x, B = 1+y  over Z_l×Z_m  (after re-parametrisation)
  - HGP code:     generalises to 2D group but with doubled qubit count

Known instances used here (small enough for exact chain analysis):
  1. BB(l=3, m=3): A = 1+x+y,  B = 1+x+y²    n=18, toy example
  2. BB(l=4, m=3): A = 1+x+y,  B = 1+x²+y    n=24
  3. BB(l=6, m=3): A = 1+x+y,  B = 1+x²+y    n=36
  4. BB(l=6, m=6): A = x³+y+y², B = y³+x+x²  n=72  [[72,12,6]] IBM code
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# ─────────────────────────────────────────────────────────────────────────────
# Bivariate bicycle code constructor
# ─────────────────────────────────────────────────────────────────────────────

def bb_code(l, m, a_shifts, b_shifts):
    """
    Build bivariate bicycle CSS code over Z_l × Z_m.

    a_shifts : list of (i, j) with A(x,y) = sum_{(i,j)} x^i · y^j
    b_shifts : list of (i, j) with B(x,y) = sum_{(i,j)} x^i · y^j

    Index convention:  (a, b) in Z_l × Z_m  →  flat index  a*m + b.

    Returns (H_X, H_Z) as uint8 arrays of shape (l*m, 2*l*m).
    """
    L = l * m

    def idx(a, b):
        return (a % l) * m + (b % m)

    def make_circulant(shifts):
        """L×L circulant matrix: M[row, col] = 1 if (row - col) is a shift."""
        M = np.zeros((L, L), dtype=np.uint8)
        for (di, dj) in shifts:
            for a in range(l):
                for b in range(m):
                    row = idx(a, b)
                    col = idx(a + di, b + dj)
                    M[row, col] = (M[row, col] + 1) % 2
        return M

    A = make_circulant(a_shifts)
    B = make_circulant(b_shifts)

    H_X = np.hstack([A, B])           # (L, 2L)
    H_Z = np.hstack([B.T % 2, A.T % 2])  # (L, 2L)

    # Verify CSS condition
    check = (H_X.astype(int) @ H_Z.T.astype(int)) % 2
    assert np.all(check == 0), \
        f"CSS condition H_X·H_Z^T ≠ 0 for BB({l},{m}) — choose compatible A,B."

    return H_X, H_Z


def gf2_rank(H):
    """GF(2) rank via Gaussian elimination."""
    H = H.copy().astype(np.uint8)
    m, n = H.shape
    rank = 0
    for col in range(n):
        pivot = next((r for r in range(rank, m) if H[r, col]), None)
        if pivot is None:
            continue
        H[[rank, pivot]] = H[[pivot, rank]]
        for r in range(m):
            if r != rank and H[r, col]:
                H[r] = (H[r] + H[rank]) % 2
        rank += 1
    return rank


def bb_params(l, m, a_shifts, b_shifts):
    """Return [[n, k, ?]] parameters of BB(l, m, A, B)."""
    H_X, H_Z = bb_code(l, m, a_shifts, b_shifts)
    n = H_X.shape[1]
    k = n - gf2_rank(H_X) - gf2_rank(H_Z)
    return {"n": n, "k": k, "H_X": H_X, "H_Z": H_Z}


# ─────────────────────────────────────────────────────────────────────────────
# Known polynomial pairs (CSS condition verified)
# ─────────────────────────────────────────────────────────────────────────────

def bb_instances():
    """
    Return a list of (label, l, m, a_shifts, b_shifts) tuples.

    All pairs are chosen so that A·B^T + B·A^T = 0 mod 2.
    Strategy: B = A^T  (adjoint).  Then A·B^T + B·A^T = A·A + A·A = 0.
    This always works and gives symmetric codes.
    """
    instances = []

    # ── 1. Symmetric pairs: B = adjoint of A ─────────────────────────────────
    # A(x,y) = 1 + x + y  →  A^†(x,y) = 1 + x^{-1} + y^{-1}
    # In Z_l×Z_m: x^{-1} = x^{l-1}, y^{-1} = y^{m-1}

    def adjoint(shifts, l, m):
        return [((-di) % l, (-dj) % m) for (di, dj) in shifts]

    def add_sym(label, l, m, a_shifts):
        b_shifts = adjoint(a_shifts, l, m)
        try:
            p = bb_params(l, m, a_shifts, b_shifts)
            instances.append((label, l, m, a_shifts, b_shifts,
                               p["n"], p["k"]))
        except AssertionError as e:
            print(f"  Skipping {label}: {e}")

    # Small: l=3, m=3  (n=18)
    add_sym("BB(3,3) A=1+x+y",    3, 3, [(0,0),(1,0),(0,1)])
    # l=4, m=3  (n=24)
    add_sym("BB(4,3) A=1+x+y",    4, 3, [(0,0),(1,0),(0,1)])
    add_sym("BB(4,3) A=1+x²+y",   4, 3, [(0,0),(2,0),(0,1)])
    # l=6, m=3  (n=36)
    add_sym("BB(6,3) A=1+x+y",    6, 3, [(0,0),(1,0),(0,1)])
    add_sym("BB(6,3) A=1+x²+y",   6, 3, [(0,0),(2,0),(0,1)])
    add_sym("BB(6,3) A=1+x³+y",   6, 3, [(0,0),(3,0),(0,1)])
    # l=4, m=4  (n=32)
    add_sym("BB(4,4) A=1+x+y",    4, 4, [(0,0),(1,0),(0,1)])
    add_sym("BB(4,4) A=1+x+y²",   4, 4, [(0,0),(1,0),(0,2)])
    # l=6, m=4  (n=48)
    add_sym("BB(6,4) A=1+x²+y",   6, 4, [(0,0),(2,0),(0,1)])

    # ── 2. IBM [[72,12,6]] code (A and B chosen by IBM, CSS verified) ─────────
    # A = x³ + y + y²,  B = y³ + x + x²  over Z_6 × Z_6
    # (These are NOT adjoints of each other — CSS needs explicit check)
    try:
        p = bb_params(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])
        instances.append(("BB(6,6) IBM [[72,12,6]]",
                          6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)],
                          p["n"], p["k"]))
    except AssertionError as e:
        print(f"  IBM [[72,12,6]] CSS check failed: {e}")

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from codes.qecc_comparison import h_to_matroid, analyse_code, bicycle_code
    from codes.hgp_code import rep, c523, hgp, hgp_params
    random.seed(42); np.random.seed(42)

    instances = bb_instances()

    print("Bivariate Bicycle (Balanced Product) codes — binary matroid analysis")
    print("=" * 90)
    print(f"{'label':<30}  {'n':>4}  {'k_css':>6}  {'r':>4}  {'N':>7}  "
          f"{'min|C|':>7}  {'min/n':>6}  {'gap':>7}  {'t_mix':>6}  {'L1':>7}")
    print("=" * 90)

    results = []
    for label, l, m, a_sh, b_sh, n_code, k_code in instances:
        H_X, _ = bb_code(l, m, a_sh, b_sh)
        M, r, n = h_to_matroid(H_X)
        res = analyse_code(label, M, r, n, max_circuits=8000)
        if res is None:
            print(f"  {label:<30}  n={n_code}  k={k_code}  — skipped")
            continue
        res["k_css"] = k_code
        res["l"] = l; res["m"] = m
        results.append(res)
        tmix_s = str(res["tmix"]) if res["tmix"] else ">500"
        print(f"  {res['label']:<30}  {n_code:>4}  {k_code:>6}  "
              f"{res['r']:>4}  {res['N']:>7}  {res['min_sz']:>7}  "
              f"{res['rel_min']:>6.3f}  {res['gap']:>7.4f}  "
              f"{tmix_s:>6}  {res['l1']:>7.4f}")

    if not results:
        print("No results."); exit()

    # ── Compare with HGP and bicycle ──────────────────────────────────────────
    print("\nReference codes:")
    ref = []
    for label, H1, H2 in [("HGP(C523,rep3)", c523(), rep(3)),
                           ("HGP(C523,rep4)", c523(), rep(4))]:
        from hgp_code import hgp_params
        p = hgp_params(H1, H2)
        M, r, n = h_to_matroid(p["H_X"])
        res = analyse_code(label, M, r, n, max_circuits=8000)
        if res:
            res["k_css"] = p["k"]; res["family"] = "HGP"
            ref.append(res)
            tmix_s = str(res["tmix"]) if res["tmix"] else ">500"
            print(f"  {label:<30}  {n:>4}  {p['k']:>6}  {r:>4}  "
                  f"{res['N']:>7}  {res['min_sz']:>7}  {res['rel_min']:>6.3f}  "
                  f"{res['gap']:>7.4f}  {tmix_s:>6}  {res['l1']:>7.4f}")

    H_b = bicycle_code(6, [0,1,2], [0,2,3])
    M_b, r_b, n_b = h_to_matroid(H_b)
    res_b = analyse_code("Bicycle r=6 w3", M_b, r_b, n_b)
    if res_b:
        res_b["k_css"] = 2; res_b["family"] = "Bicycle"
        ref.append(res_b)
        tmix_s = str(res_b["tmix"]) if res_b["tmix"] else ">500"
        print(f"  {'Bicycle r=6 w3':<30}  {n_b:>4}  {2:>6}  {r_b:>4}  "
              f"{res_b['N']:>7}  {res_b['min_sz']:>7}  {res_b['rel_min']:>6.3f}  "
              f"{res_b['gap']:>7.4f}  {tmix_s:>6}  {res_b['l1']:>7.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    for r in results:
        r["family"] = "BB"
    all_res = ref + results

    FAM_COL = {"BB": "#e74c3c", "HGP": "#9467bd", "Bicycle": "#2ca02c"}
    FAM_MKR = {"BB": "D", "HGP": "^", "Bicycle": "s"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax1, ax2), (ax3, ax4) = axes

    # Panel 1: gap vs n
    seen = set()
    for res in all_res:
        fam = res["family"]
        lbl = fam if fam not in seen else None
        ax1.scatter(res["n"], res["gap"],
                    color=FAM_COL[fam], marker=FAM_MKR[fam],
                    s=90, zorder=3, label=lbl)
        ax1.annotate(f"k={res['k_css']}",
                     xy=(res["n"], res["gap"]),
                     xytext=(4, 2), textcoords="offset points",
                     fontsize=7, color="gray")
        seen.add(fam)
    ax1.set_xlabel("n  (physical qubits)", fontsize=11)
    ax1.set_ylabel("Spectral gap δ", fontsize=11)
    ax1.set_title("Spectral gap vs n", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # Panel 2: L1 vs min|C|/n
    seen = set()
    for res in all_res:
        fam = res["family"]
        lbl = fam if fam not in seen else None
        ax2.scatter(res["rel_min"], res["l1"],
                    color=FAM_COL[fam], marker=FAM_MKR[fam],
                    s=90, zorder=3, label=lbl)
        seen.add(fam)
    ax2.set_xlabel("min|C| / n  (distance proxy)", fontsize=11)
    ax2.set_ylabel("L1 to uniform  (stationarity bias)", fontsize=11)
    ax2.set_title("Code quality: distance proxy vs stationarity bias",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # Panel 3: circuit size histogram for a few BB codes
    highlight = [r for r in results if r["N"] < 3000][:4]
    colors_h = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(highlight), 1)))
    for res, col in zip(highlight, colors_h):
        sizes = res["sizes"]
        unique = sorted(set(sizes.tolist()))
        frac = np.array([np.sum(sizes == sz) for sz in unique]) / res["N"]
        ax3.bar(unique, frac, alpha=0.55, color=col, edgecolor="white", width=0.6,
                label=f"{res['label']}  N={res['N']}, k={res['k_css']}")
    ax3.set_xlabel("Circuit size |C|", fontsize=11)
    ax3.set_ylabel("Fraction of circuits", fontsize=11)
    ax3.set_title("Circuit size distributions (BB codes)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    # Panel 4: TV curves
    t_ax = np.arange(1, 501)
    for res, col, ls in zip(
        all_res[:6],
        ["#e74c3c","#e74c3c","#9467bd","#9467bd","#2ca02c","#2ca02c"],
        ["-","--","-","--","-","--"]
    ):
        tv = res["tv"]
        tmix_s = str(res["tmix"]) if res["tmix"] else ">500"
        ax4.plot(t_ax[:len(tv)], tv, lw=2, color=col, ls=ls,
                 label=f"{res['label']}  t_mix={tmix_s}")
    ax4.axhline(0.25, color="gray", lw=1, ls=":", label="ε=0.25")
    ax4.set_xlim(0, 40); ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Steps t", fontsize=11)
    ax4.set_ylabel("Worst-case TV distance", fontsize=11)
    ax4.set_title("TV mixing curves", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=7, ncol=2); ax4.grid(alpha=0.3)

    fig.suptitle("Balanced Product (Bivariate Bicycle) codes — Markov chain on circuits",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "markov-circuits/bp_portrait.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()
