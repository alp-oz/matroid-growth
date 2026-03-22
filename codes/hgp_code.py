"""
Hypergraph Product (HGP) codes as binary matroids — Markov chain study.

Construction (Tillich-Zémor 2014):
  Given classical parity check matrices H1 (r1×n1) and H2 (r2×n2) over GF(2):

    H_X = [ H1 ⊗ I_{n2}  |  I_{r1} ⊗ H2^T ]   shape: (r1·n2) × (n1·n2 + r1·r2)
    H_Z = [ I_{n1} ⊗ H2  |  H1^T ⊗ I_{r2} ]   shape: (n1·r2) × (n1·n2 + r1·r2)

  CSS condition: H_X · H_Z^T = 0 (mod 2) — holds always by construction.

  Parameters:
    n     = n1·n2 + r1·r2   (physical qubits)
    k     = n - rank(H_X) - rank(H_Z)   (logical qubits)
    d     ≥ min(d1, d2)     (minimum distance lower bound)

Special cases:
  HGP(rep(L), rep(L))  = toric code on L×L torus  (k=2, d=L)
  HGP(H, H) symmetric  = "square" HGP code

Instances studied here (all with n ≤ 35 for tractable circuit enumeration):
  1. HGP(rep3, rep3)     — rectangular toric 3×3, baseline check
  2. HGP(rep3, rep4)     — rectangular toric 3×4
  3. HGP(C422, rep3)     — [4,2,2] × rep(3), k=4, d=2
  4. HGP(C523, rep3)     — [5,2,3] × rep(3), k=4, d=3
  5. HGP(C523, C523)     — [5,2,3] × [5,2,3], k>4, d=3, n=34
  6. HGP(rand, rand)     — random small LDPC × itself, several seeds
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from codes.toric_code import toric_to_matroid


# ─────────────────────────────────────────────────────────────────────────────
# Classical code constructors
# ─────────────────────────────────────────────────────────────────────────────

def rep(n):
    """Repetition code: (n-1)×n parity check. k=1, d=n."""
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


def c422():
    """[4,2,2] even-weight-pair code. H is 2×4, k=2, d=2."""
    return np.array([[1, 1, 0, 0],
                     [0, 0, 1, 1]], dtype=np.uint8)


def c523():
    """
    [5,2,3] code. H is 3×5, k=2, d=3.
    Generator: G = [[1,0,1,1,0],[0,1,0,1,1]]
    Parity check (verified H·G^T = 0 mod 2):
    """
    return np.array([[1, 0, 0, 1, 1],
                     [0, 1, 0, 0, 1],
                     [0, 0, 1, 1, 1]], dtype=np.uint8)


def random_ldpc_small(n, row_weight, n_checks, seed):
    """Random parity check: n_checks × n, each row has exactly row_weight ones."""
    rng = np.random.default_rng(seed)
    H = np.zeros((n_checks, n), dtype=np.uint8)
    for i in range(n_checks):
        cols = rng.choice(n, size=row_weight, replace=False)
        H[i, cols] = 1
    return H


# ─────────────────────────────────────────────────────────────────────────────
# HGP construction
# ─────────────────────────────────────────────────────────────────────────────

def hgp(H1, H2):
    """
    Build hypergraph product CSS code from H1 (r1×n1) and H2 (r2×n2).
    Returns (H_X, H_Z) over GF(2) as uint8 arrays.
    """
    r1, n1 = H1.shape
    r2, n2 = H2.shape

    H_X = np.hstack([
        np.kron(H1, np.eye(n2, dtype=np.uint8)),
        np.kron(np.eye(r1, dtype=np.uint8), H2.T)
    ]) % 2

    H_Z = np.hstack([
        np.kron(np.eye(n1, dtype=np.uint8), H2),
        np.kron(H1.T, np.eye(r2, dtype=np.uint8))
    ]) % 2

    # Verify CSS condition
    check = (H_X.astype(int) @ H_Z.T.astype(int)) % 2
    assert np.all(check == 0), "CSS condition H_X · H_Z^T ≠ 0!"

    return H_X, H_Z


def hgp_params(H1, H2):
    """Compute [[n, k, d≥]] parameters of HGP(H1,H2)."""
    r1, n1 = H1.shape
    r2, n2 = H2.shape
    H_X, H_Z = hgp(H1, H2)
    n = n1 * n2 + r1 * r2
    # k = n - rank(H_X) - rank(H_Z)  over GF(2)
    def gf2_rank(H):
        H = H.copy().astype(np.uint8)
        m, nc = H.shape
        rank = 0
        for col in range(nc):
            pivot = None
            for row in range(rank, m):
                if H[row, col]:
                    pivot = row; break
            if pivot is None: continue
            H[[rank, pivot]] = H[[pivot, rank]]
            for row in range(m):
                if row != rank and H[row, col]:
                    H[row] = (H[row] + H[rank]) % 2
            rank += 1
        return rank
    k = n - gf2_rank(H_X) - gf2_rank(H_Z)
    # d lower bound from classical codes
    d1_lb = n1 - gf2_rank(H1) + 1   # crude lb: Singleton-ish, not tight
    # Better: just note min(d1, d2) as lb, compute from rep codes exactly
    return {"n": n, "k": k, "H_X": H_X, "H_Z": H_Z}


# ─────────────────────────────────────────────────────────────────────────────
# Build instances
# ─────────────────────────────────────────────────────────────────────────────

def build_hgp_instances():
    instances = []

    def add(label, H1, H2):
        p = hgp_params(H1, H2)
        M, r, n = h_to_matroid(p["H_X"])
        instances.append({
            "label": label,
            "H1": H1, "H2": H2,
            "M": M, "r": r, "n": n,
            "k_css": p["k"],
        })

    # ── 1. Baseline: toric variants via repetition codes ─────────────────────
    add("HGP(rep3,rep3)",  rep(3), rep(3))   # = toric 3×3
    add("HGP(rep3,rep4)",  rep(3), rep(4))   # = rectangular toric 3×4
    add("HGP(rep4,rep4)",  rep(4), rep(4))   # = toric 4×4

    # ── 2. k>1: [4,2,2] code ─────────────────────────────────────────────────
    add("HGP(C422,rep3)",  c422(), rep(3))
    add("HGP(C422,C422)",  c422(), c422())

    # ── 3. k>1, d≥3: [5,2,3] code ────────────────────────────────────────────
    add("HGP(C523,rep3)",  c523(), rep(3))
    add("HGP(C523,rep4)",  c523(), rep(4))
    add("HGP(C523,C523)",  c523(), c523())   # n=34, might be large

    # ── 4. Random small LDPC × itself ────────────────────────────────────────
    for seed in range(3):
        H = random_ldpc_small(n=5, row_weight=3, n_checks=3, seed=seed)
        add(f"HGP(rand5,rand5) s={seed}", H, H)

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from codes.qecc_comparison import h_to_matroid, analyse_code, bicycle_code
    random.seed(42); np.random.seed(42)

    instances = build_hgp_instances()

    print("Hypergraph Product codes — binary matroid chain analysis")
    print("=" * 85)
    print(f"{'label':<26}  {'n':>4}  {'k_css':>6}  {'r':>4}  {'N':>7}  "
          f"{'min|C|':>7}  {'min/n':>6}  {'gap':>7}  {'t_mix':>6}  {'L1':>7}")
    print("=" * 85)

    results = []
    for inst in instances:
        res = analyse_code(inst["label"], inst["M"], inst["r"], inst["n"],
                           max_circuits=6000)
        if res is None:
            print(f"  {inst['label']:<26}  n={inst['n']}  k={inst['k_css']}"
                  f"  — skipped (no circuits / truncated)")
            continue
        res["k_css"] = inst["k_css"]
        results.append(res)
        tmix_s = str(res["tmix"]) if res["tmix"] else ">500"
        print(f"  {res['label']:<26}  {inst['n']:>4}  {inst['k_css']:>6}  "
              f"{res['r']:>4}  {res['N']:>7}  {res['min_sz']:>7}  "
              f"{res['rel_min']:>6.3f}  {res['gap']:>7.4f}  "
              f"{tmix_s:>6}  {res['l1']:>7.4f}")

    # ── Summary sorted by min|C|/n ────────────────────────────────────────────
    print("\nSorted by min|C|/n (ascending = worse distance):")
    print(f"  {'label':<26}  {'k_css':>6}  {'min/n':>6}  "
          f"{'gap':>7}  {'t_mix':>6}  {'L1':>7}")
    print("  " + "-" * 65)
    for res in sorted(results, key=lambda x: x["rel_min"]):
        tmix_s = str(res["tmix"]) if res["tmix"] else ">500"
        print(f"  {res['label']:<26}  {res['k_css']:>6}  {res['rel_min']:>6.3f}  "
              f"{res['gap']:>7.4f}  {tmix_s:>6}  {res['l1']:>7.4f}")

    # ── Circuit size distributions ────────────────────────────────────────────
    print()
    for res in results:
        sizes = res["sizes"]
        unique = sorted(set(sizes.tolist()))
        dist = {sz: int(np.sum(sizes == sz)) for sz in unique}
        print(f"  {res['label']}  (N={res['N']}, k_css={res['k_css']}): "
              f"sizes {dist}")

    # ══════════════════════════════════════════════════════════════════════════
    # Figure: HGP portrait + comparison with toric and bicycle
    # ══════════════════════════════════════════════════════════════════════════
    if not results:
        print("No results to plot."); exit()

    # Add reference codes for comparison
    print("\nAdding toric L=3 and bicycle r=6 w3 for comparison...")
    from analysis import all_circuits
    from stationary import build_transition_matrix, stationary_distribution
    from mixing import spectral_analysis, tv_distance_curve

    ref_codes = []
    # Toric L=3
    M_t, r_t, n_t, _, _ = toric_to_matroid(3)
    res_t = analyse_code("Toric L=3", M_t, r_t, n_t)
    if res_t:
        res_t["k_css"] = 2; res_t["family"] = "Toric"
        ref_codes.append(res_t)
    # Bicycle r=6 w3
    H_b = bicycle_code(6, [0,1,2], [0,2,3])
    M_b, r_b, n_b = h_to_matroid(H_b)
    res_b = analyse_code("Bicycle r=6 w3", M_b, r_b, n_b)
    if res_b:
        res_b["k_css"] = 2; res_b["family"] = "Bicycle"
        ref_codes.append(res_b)

    # Tag HGP results by subtype
    for res in results:
        if "rep" in res["label"] and "C" not in res["label"] and "rand" not in res["label"]:
            res["family"] = "HGP-toric"
        elif "rand" in res["label"]:
            res["family"] = "HGP-rand"
        else:
            res["family"] = "HGP-LDPC"

    all_res = ref_codes + results

    FAM_COL = {
        "Toric":    "#1f77b4",
        "Bicycle":  "#2ca02c",
        "HGP-toric":"#ff7f0e",
        "HGP-LDPC": "#9467bd",
        "HGP-rand": "#8c564b",
    }
    FAM_MKR = {
        "Toric": "o", "Bicycle": "s",
        "HGP-toric": "D", "HGP-LDPC": "^", "HGP-rand": "x",
    }

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, hspace=0.48, wspace=0.35)

    # Panel 1: Circuit size histogram — one per interesting HGP instance + refs
    ax1 = fig.add_subplot(gs[0, :])
    highlight = (["HGP(rep3,rep3)", "HGP(C523,rep3)", "HGP(C523,C523)"]
                 + ["Toric L=3", "Bicycle r=6 w3"])
    colors_h = ["#ff7f0e", "#9467bd", "#d62728", "#1f77b4", "#2ca02c"]
    for label, col in zip(highlight, colors_h):
        found = next((r for r in all_res if r["label"] == label), None)
        if not found: continue
        sizes = found["sizes"]
        unique = sorted(set(sizes.tolist()))
        frac = np.array([np.sum(sizes == sz) for sz in unique]) / found["N"]
        ax1.bar(unique, frac, alpha=0.5, color=col, edgecolor="white", width=0.7,
                label=f"{label}  (N={found['N']}, k={found['k_css']}, "
                      f"min|C|={found['min_sz']})")
    ax1.set_xlabel("Circuit size |C|", fontsize=11)
    ax1.set_ylabel("Fraction of circuits", fontsize=11)
    ax1.set_title("Circuit size distributions — HGP codes vs reference",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(axis="y", alpha=0.3)

    # Panel 2: π vs |C| scatter for HGP(C523,rep3) and toric L=3
    ax2 = fig.add_subplot(gs[1, 0])
    for label, col in [("HGP(C523,rep3)", "#9467bd"), ("Toric L=3", "#1f77b4"),
                        ("Bicycle r=6 w3", "#2ca02c")]:
        found = next((r for r in all_res if r["label"] == label), None)
        if not found: continue
        jit = np.random.uniform(-0.1, 0.1, len(found["sizes"]))
        ax2.scatter(found["sizes"] + jit, found["pi"],
                    s=10, alpha=0.45, color=col, label=label)
        ax2.axhline(1.0/found["N"], color=col, lw=1, ls="--", alpha=0.6)
    ax2.set_xlabel("Circuit size |C|", fontsize=11)
    ax2.set_ylabel("π(C)", fontsize=11)
    ax2.set_title("Stationary distribution vs circuit size",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Panel 3: L1 vs min|C|/n — all instances coloured by family
    ax3 = fig.add_subplot(gs[1, 1])
    seen = set()
    for res in all_res:
        fam = res.get("family", "HGP-LDPC")
        col = FAM_COL.get(fam, "gray")
        mkr = FAM_MKR.get(fam, "o")
        lbl = fam if fam not in seen else None
        ax3.scatter(res["rel_min"], res["l1"], color=col, marker=mkr,
                    s=80, zorder=3, label=lbl)
        seen.add(fam)
    ax3.set_xlabel("min|C| / n", fontsize=11)
    ax3.set_ylabel("L1 distance to uniform", fontsize=11)
    ax3.set_title("Stationarity bias vs code quality",
                  fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    # Panel 4: gap vs n
    ax4 = fig.add_subplot(gs[2, 0])
    seen = set()
    for res in all_res:
        fam = res.get("family", "HGP-LDPC")
        col = FAM_COL.get(fam, "gray")
        mkr = FAM_MKR.get(fam, "o")
        lbl = fam if fam not in seen else None
        ax4.scatter(res["n"], res["gap"], color=col, marker=mkr,
                    s=80, zorder=3, label=lbl)
        seen.add(fam)
    ax4.set_xlabel("n  (physical qubits)", fontsize=11)
    ax4.set_ylabel("Spectral gap", fontsize=11)
    ax4.set_title("Spectral gap vs n", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    # Panel 5: TV curves — rep-toric, LDPC-HGP, reference
    ax5 = fig.add_subplot(gs[2, 1])
    t_ax = np.arange(1, 501)
    for label, col in [("HGP(rep3,rep3)", "#ff7f0e"),
                        ("HGP(C523,rep3)", "#9467bd"),
                        ("Toric L=3",     "#1f77b4"),
                        ("Bicycle r=6 w3","#2ca02c")]:
        found = next((r for r in all_res if r["label"] == label), None)
        if not found: continue
        tv = found["tv"]
        tmix_s = str(found["tmix"]) if found["tmix"] else ">500"
        ax5.plot(t_ax[:len(tv)], tv, lw=2, color=col,
                 label=f"{label}  t_mix={tmix_s}")
    ax5.axhline(0.25, color="gray", lw=1, ls=":", label="ε=0.25")
    ax5.set_xlim(0, 30); ax5.set_ylim(0, 1.05)
    ax5.set_xlabel("Steps t", fontsize=11)
    ax5.set_ylabel("Worst-case TV distance", fontsize=11)
    ax5.set_title("TV mixing curves", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    fig.suptitle("Hypergraph Product codes — Markov chain on circuits",
                 fontsize=14, fontweight="bold")
    out = "markov-circuits/hgp_portrait.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()
