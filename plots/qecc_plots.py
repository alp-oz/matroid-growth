"""
Two figures:

Figure 1 — QECC portrait (analogous to PA plots.py):
  Panel 1: Circuit size histograms for representative codes from each family
  Panel 2: π(C) vs circuit size scatter
  Panel 3: Mean π per size class normalised by uniform (1/N)
  Panel 4: TV distance curves
  Panel 5: Spectral gap vs n across all QECCs
  Panel 6: L1 to uniform vs min|C|/n (code quality scatter)

Figure 2 — PA vs QECC comparison:
  Panel 1: Circuit size histogram PA vs QECC (two contrasting instances)
  Panel 2: π vs |C| scatter PA vs QECC
  Panel 3: L1 to uniform vs n: PA family and QECC family
  Panel 4: t_mix vs n: PA vs QECC
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.engine import MatroidEngine
from core.circuits import all_circuits
from core.stationary import build_transition_matrix, stationary_distribution
from core.mixing import spectral_analysis, tv_distance_curve
from codes.qecc_comparison import (
    bicycle_code, h_to_matroid, random_ldpc_biregular, analyse_code
)
from codes.toric_code import toric_to_matroid


# ─────────────────────────────────────────────────────────────────────────────
# Build QECC instances  (same set as qecc_comparison.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_qecc_instances():
    instances = []

    # w2-LDPC
    for seed in range(3):
        H = random_ldpc_biregular(n=16, col_weight=2, row_weight=4, seed=seed + 10)
        M, r, n = h_to_matroid(H)
        instances.append((f"w2-LDPC s={seed}", M, r, n, "w2-LDPC"))

    # Toric
    for L in [2, 3]:
        M, r, n, _, _ = toric_to_matroid(L)
        instances.append((f"Toric L={L}", M, r, n, "Toric"))

    # Bicycle
    for label, exps in [("[0,1|0,2]", ([0,1],[0,2])),
                         ("w3",        ([0,1,2],[0,2,3]))]:
        H = bicycle_code(6, *exps)
        M, r, n = h_to_matroid(H)
        instances.append((f"Bicycle r=6 {label}", M, r, n, "Bicycle"))

    for label, exps in [("[0,1|0,4]", ([0,1],[0,4])),
                         ("w3",        ([0,1,3],[0,2,5]))]:
        H = bicycle_code(9, *exps)
        M, r, n = h_to_matroid(H)
        instances.append((f"Bicycle r=9 {label}", M, r, n, "Bicycle"))

    H = bicycle_code(12, [0,1], [0,5])
    M, r, n = h_to_matroid(H)
    instances.append(("Bicycle r=12 w2", M, r, n, "Bicycle"))

    H = bicycle_code(12, [0,1,4], [0,2,7])
    M, r, n = h_to_matroid(H)
    instances.append(("Bicycle r=12 w3", M, r, n, "Bicycle"))

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_COLORS = {
    "w2-LDPC":  "#d62728",
    "Toric":    "#1f77b4",
    "Bicycle":  "#2ca02c",
}
FAMILY_MARKERS = {
    "w2-LDPC": "x",
    "Toric":   "o",
    "Bicycle": "s",
}

# Pick one representative per family for detailed panels
REPRESENTATIVES = {
    "w2-LDPC": "w2-LDPC s=0",
    "Toric":   "Toric L=3",
    "Bicycle": "Bicycle r=6 w3",   # highest min|C|/n
}

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)

    print("Running QECC analysis...")
    instances = build_qecc_instances()
    results = []
    for label, M, r, n, family in instances:
        res = analyse_code(label, M, r, n)
        if res:
            res["family"] = family
            results.append(res)
            print(f"  {label:<32} N={res['N']:>5}  gap={res['gap']:.3f}"
                  f"  t_mix={res['tmix']}  L1={res['l1']:.3f}")

    # Pick representatives
    rep = {r["label"]: r for r in results}

    # ── PA matroids ────────────────────────────────────────────────────────
    print("\nRunning PA analysis...")
    BASE = dict(k_params=2, C=0.1, gamma=0.0, beta=0.8, start_r=2)
    pa_steps = [10, 14, 18, 22, 26, 30]
    pa_results = []
    for ns in pa_steps:
        random.seed(42); np.random.seed(42)
        engine = MatroidEngine(**{**BASE, "n_steps": ns})
        result = engine.run()
        M_pa, r_pa, n_pa = result["M"], result["r"], result["n"]
        all_c, trunc, _ = all_circuits(M_pa, r_pa, mode='global')
        if trunc or len(all_c) < 2:
            continue
        circuits_pa = sorted(all_c, key=sorted)
        N_pa = len(circuits_pa)
        sizes_pa = np.array([len(c) for c in circuits_pa])
        P_pa  = build_transition_matrix(M_pa, r_pa, circuits_pa, mode='adjacent')
        pi_pa = stationary_distribution(P_pa)
        sp_pa = spectral_analysis(P_pa)
        tv_pa, _, tmix_pa = tv_distance_curve(P_pa, pi_pa, max_t=500)
        l1_pa = float(np.sum(np.abs(pi_pa - np.ones(N_pa)/N_pa)))
        pa_results.append({
            "ns": ns, "r": r_pa, "n": n_pa, "N": N_pa,
            "sizes": sizes_pa, "gap": sp_pa["gap"],
            "tmix": tmix_pa, "pi": pi_pa, "tv": tv_pa, "l1": l1_pa,
        })
        print(f"  PA n_steps={ns:<3}  N={N_pa:>5}  gap={sp_pa['gap']:.3f}"
              f"  t_mix={tmix_pa}  L1={l1_pa:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — QECC portrait
    # ══════════════════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(18, 14))
    gs1  = gridspec.GridSpec(3, 2, hspace=0.48, wspace=0.35)

    # ── Panel 1: Circuit size histograms (representatives, normalised) ────────
    ax1 = fig1.add_subplot(gs1[0, 0])
    rep_labels = list(REPRESENTATIVES.values())
    for fam, rlabel in REPRESENTATIVES.items():
        if rlabel not in rep: continue
        r = rep[rlabel]
        sizes = r["sizes"]
        col  = FAMILY_COLORS[fam]
        unique = sorted(set(sizes.tolist()))
        counts = np.array([np.sum(sizes == sz) for sz in unique]) / r["N"]
        ax1.bar(unique, counts, alpha=0.55, color=col, edgecolor="white",
                label=f"{rlabel}  (N={r['N']}, min|C|={r['min_sz']})",
                width=0.8)
    ax1.set_xlabel("Circuit size |C|", fontsize=11)
    ax1.set_ylabel("Fraction of circuits", fontsize=11)
    ax1.set_title("Circuit size distribution\n(one representative per family)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: π(C) vs |C| scatter (representatives) ───────────────────────
    ax2 = fig1.add_subplot(gs1[0, 1])
    for fam, rlabel in REPRESENTATIVES.items():
        if rlabel not in rep: continue
        r = rep[rlabel]
        col = FAMILY_COLORS[fam]
        jitter = np.random.uniform(-0.12, 0.12, len(r["sizes"]))
        ax2.scatter(r["sizes"] + jitter, r["pi"],
                    s=10, alpha=0.45, color=col, label=rlabel)
        ax2.axhline(1.0/r["N"], color=col, lw=1, ls="--", alpha=0.6)
    ax2.set_xlabel("Circuit size |C|", fontsize=11)
    ax2.set_ylabel("π(C)", fontsize=11)
    ax2.set_title("Stationary distribution vs circuit size",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ── Panel 3: Mean π per size class, normalised by 1/N ────────────────────
    ax3 = fig1.add_subplot(gs1[1, 0])
    for fam, rlabel in REPRESENTATIVES.items():
        if rlabel not in rep: continue
        r = rep[rlabel]
        col = FAMILY_COLORS[fam]
        sizes = r["sizes"]
        unique = sorted(set(sizes.tolist()))
        means  = np.array([r["pi"][sizes == sz].mean() for sz in unique])
        norm   = means * r["N"]   # divide by 1/N → 1.0 = uniform
        ax3.plot(unique, norm, "o-", color=col, lw=2, ms=7, label=rlabel)
    ax3.axhline(1.0, color="gray", lw=1.2, ls=":", label="uniform (=1)")
    ax3.set_xlabel("Circuit size |C|", fontsize=11)
    ax3.set_ylabel("Mean π(C) / (1/N)", fontsize=11)
    ax3.set_title("Relative stationary weight per size class\n(1 = uniform)",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # ── Panel 4: TV distance curves (representatives) ─────────────────────────
    ax4 = fig1.add_subplot(gs1[1, 1])
    t_ax = np.arange(1, 501)
    for fam, rlabel in REPRESENTATIVES.items():
        if rlabel not in rep: continue
        r  = rep[rlabel]
        col = FAMILY_COLORS[fam]
        tv = r["tv"]
        tmix_s = str(r["tmix"]) if r["tmix"] else ">500"
        ax4.plot(t_ax[:len(tv)], tv, color=col, lw=2,
                 label=f"{rlabel}  t_mix={tmix_s}")
    ax4.axhline(0.25, color="gray", lw=1, ls=":", label="ε=0.25")
    ax4.set_xlim(0, 30); ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Steps t", fontsize=11)
    ax4.set_ylabel("Worst-case TV distance", fontsize=11)
    ax4.set_title("TV mixing curves", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # ── Panel 5: Spectral gap vs n ────────────────────────────────────────────
    ax5 = fig1.add_subplot(gs1[2, 0])
    seen = set()
    for r in results:
        fam = r["family"]
        col = FAMILY_COLORS[fam]
        mkr = FAMILY_MARKERS[fam]
        lbl = fam if fam not in seen else None
        ax5.scatter(r["n"], r["gap"], color=col, marker=mkr, s=70, zorder=3,
                    label=lbl)
        seen.add(fam)
    ax5.set_xlabel("n  (physical qubits)", fontsize=11)
    ax5.set_ylabel("Spectral gap", fontsize=11)
    ax5.set_title("Spectral gap vs n across all QECCs",
                  fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

    # ── Panel 6: L1 vs min|C|/n ───────────────────────────────────────────────
    ax6 = fig1.add_subplot(gs1[2, 1])
    seen = set()
    for r in results:
        fam = r["family"]
        col = FAMILY_COLORS[fam]
        mkr = FAMILY_MARKERS[fam]
        lbl = fam if fam not in seen else None
        ax6.scatter(r["rel_min"], r["l1"], color=col, marker=mkr, s=70,
                    zorder=3, label=lbl)
        seen.add(fam)
    ax6.set_xlabel("min|C| / n   (normalised distance lower bound)", fontsize=11)
    ax6.set_ylabel("L1 distance to uniform", fontsize=11)
    ax6.set_title("Stationarity bias vs code quality",
                  fontsize=11, fontweight="bold")
    ax6.legend(fontsize=9); ax6.grid(alpha=0.3)

    fig1.suptitle("QECC circuits — Markov chain portrait", fontsize=14,
                  fontweight="bold")
    out1 = "markov-circuits/qecc_portrait.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out1}")
    plt.close(fig1)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — PA vs QECC comparison
    # ══════════════════════════════════════════════════════════════════════════
    # Pick contrasting pair: PA n_steps=26 (N≈371, n≈unknown) vs Bicycle r=9 w3
    pa_mid  = next((p for p in pa_results if p["ns"] == 26), pa_results[-1])
    pa_large = pa_results[-1]

    qecc_mid  = rep.get("Bicycle r=6 w3")   or results[0]
    qecc_large = rep.get("Bicycle r=12 w3") or rep.get("Bicycle r=9 w3") or results[-1]

    fig2 = plt.figure(figsize=(16, 12))
    gs2  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    PA_COL   = "#ff7f0e"
    QECC_COL = "#2ca02c"

    # ── Panel 1: Circuit size histogram — PA vs best QECC ────────────────────
    ax1b = fig2.add_subplot(gs2[0, 0])
    # PA (largest)
    pa = pa_large
    pa_unique = sorted(set(pa["sizes"].tolist()))
    pa_frac   = np.array([np.sum(pa["sizes"] == s) for s in pa_unique]) / pa["N"]
    ax1b.bar(pa_unique, pa_frac, alpha=0.6, color=PA_COL, edgecolor="white",
             label=f"PA n_steps={pa['ns']}  N={pa['N']}  min|C|={int(pa['sizes'].min())}")

    # QECC (best)
    qe = qecc_large
    qe_unique = sorted(set(qe["sizes"].tolist()))
    qe_frac   = np.array([np.sum(qe["sizes"] == s) for s in qe_unique]) / qe["N"]
    ax1b.bar(qe_unique, qe_frac, alpha=0.6, color=QECC_COL, edgecolor="white",
             label=f"{qe['label']}  N={qe['N']}  min|C|={qe['min_sz']}")

    ax1b.set_xlabel("Circuit size |C|", fontsize=11)
    ax1b.set_ylabel("Fraction of circuits", fontsize=11)
    ax1b.set_title("Circuit size distribution\nPA vs QECC", fontsize=11, fontweight="bold")
    ax1b.legend(fontsize=8); ax1b.grid(axis="y", alpha=0.3)

    # ── Panel 2: π vs |C| scatter — PA vs QECC ───────────────────────────────
    ax2b = fig2.add_subplot(gs2[0, 1])
    pa = pa_mid
    jit_pa = np.random.uniform(-0.1, 0.1, len(pa["sizes"]))
    ax2b.scatter(pa["sizes"] + jit_pa, pa["pi"],
                 s=8, alpha=0.35, color=PA_COL,
                 label=f"PA n_steps={pa['ns']}  N={pa['N']}")
    ax2b.axhline(1.0/pa["N"], color=PA_COL, lw=1.2, ls="--", alpha=0.7,
                 label=f"uniform (PA)")

    qe = qecc_mid
    jit_qe = np.random.uniform(-0.1, 0.1, len(qe["sizes"]))
    ax2b.scatter(qe["sizes"] + jit_qe, qe["pi"],
                 s=8, alpha=0.45, color=QECC_COL,
                 label=f"{qe['label']}  N={qe['N']}")
    ax2b.axhline(1.0/qe["N"], color=QECC_COL, lw=1.2, ls="--", alpha=0.7,
                 label=f"uniform (QECC)")

    ax2b.set_xlabel("Circuit size |C|", fontsize=11)
    ax2b.set_ylabel("π(C)", fontsize=11)
    ax2b.set_title("Stationary distribution vs size\nPA vs QECC",
                   fontsize=11, fontweight="bold")
    ax2b.legend(fontsize=8); ax2b.grid(alpha=0.3)

    # ── Panel 3: L1 to uniform vs n ──────────────────────────────────────────
    ax3b = fig2.add_subplot(gs2[1, 0])
    # PA
    pa_ns  = [p["n"]  for p in pa_results]
    pa_l1  = [p["l1"] for p in pa_results]
    ax3b.plot(pa_ns, pa_l1, "o-", color=PA_COL, lw=2, ms=8, label="PA matroids")
    # QECC — one per family, take mean across seeds
    from collections import defaultdict
    fam_l1 = defaultdict(list)
    fam_n  = defaultdict(list)
    for r in results:
        fam_l1[r["family"]].append(r["l1"])
        fam_n [r["family"]].append(r["n"])
    for fam in ["w2-LDPC", "Toric", "Bicycle"]:
        if fam not in fam_n: continue
        ns_f = fam_n[fam]; l1_f = fam_l1[fam]
        order = np.argsort(ns_f)
        ax3b.plot(np.array(ns_f)[order], np.array(l1_f)[order],
                  FAMILY_MARKERS[fam] + "--", color=FAMILY_COLORS[fam],
                  lw=1.5, ms=7, label=fam)
    ax3b.set_xlabel("n  (physical qubits)", fontsize=11)
    ax3b.set_ylabel("L1 distance to uniform", fontsize=11)
    ax3b.set_title("L1 bias vs n\n(higher = more biased away from uniform)",
                   fontsize=11, fontweight="bold")
    ax3b.legend(fontsize=9); ax3b.grid(alpha=0.3)

    # ── Panel 4: t_mix vs n ───────────────────────────────────────────────────
    ax4b = fig2.add_subplot(gs2[1, 1])
    pa_tm = [p["tmix"] if p["tmix"] else 500 for p in pa_results]
    ax4b.plot(pa_ns, pa_tm, "o-", color=PA_COL, lw=2, ms=8, label="PA matroids")

    # Add t_mix/n = 1 reference line
    n_range = np.linspace(min(pa_ns + [r["n"] for r in results]),
                          max(pa_ns + [r["n"] for r in results]), 50)
    ax4b.plot(n_range, n_range, "k:", lw=1, alpha=0.5, label="t_mix = n (reference)")

    fam_tm = defaultdict(list)
    for r in results:
        fam_tm[r["family"]].append(r["tmix"] if r["tmix"] else 500)
    for fam in ["w2-LDPC", "Toric", "Bicycle"]:
        if fam not in fam_n: continue
        ns_f = fam_n[fam]; tm_f = fam_tm[fam]
        order = np.argsort(ns_f)
        ax4b.plot(np.array(ns_f)[order], np.array(tm_f)[order],
                  FAMILY_MARKERS[fam] + "--", color=FAMILY_COLORS[fam],
                  lw=1.5, ms=7, label=fam)
    ax4b.set_xlabel("n  (physical qubits)", fontsize=11)
    ax4b.set_ylabel("t_mix(0.25)", fontsize=11)
    ax4b.set_title("Mixing time vs n\n(below t=n line → poly(n) mixing)",
                   fontsize=11, fontweight="bold")
    ax4b.legend(fontsize=9); ax4b.grid(alpha=0.3)

    fig2.suptitle("PA matroids vs QECC families — Markov chain comparison",
                  fontsize=14, fontweight="bold")
    out2 = "markov-circuits/qecc_vs_pa.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")
    plt.close(fig2)
