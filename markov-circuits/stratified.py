"""
Stratified MH chain: target distribution uniform within each size class,
natural (adjacent chain) weights across size classes.

π_strat(C) = π_size(|C|) / n_{|C|}

where π_size(k) = Σ_{C:|C|=k} π_adj(C)  and  n_k = #{circuits of size k}.

MH acceptance ratio:
  same-size move   (|C'| = |C|):  α = min(1, q_rev / q_fwd)
  cross-size move  (|C'| ≠ |C|):  α = min(1, (n_{|C|}/n_{|C'|}) * q_rev/q_fwd)

Also checks within-class irreducibility and compares mixing times.
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

from engine import MatroidEngine
from markov_chain import (MarkovChainCircuits, MHCircuitChain,
                           fundamental_circuits, decompose_into_circuits)
from analysis import all_circuits
from stationary import build_transition_matrix, stationary_distribution
from mixing import spectral_analysis, tv_distance_curve


# ─────────────────────────────────────────────────────────────────────────────
# Stratified MH chain
# ─────────────────────────────────────────────────────────────────────────────

class MHStratifiedChain:
    """
    MH chain targeting π_strat(C) = π_size(|C|) / n_{|C|}.

    Uniform within each size class; natural cross-class weights preserved.
    Only within-class imbalance is corrected → higher acceptance rate.
    """

    def __init__(self, M, r, circuits):
        self.M         = M
        self.r         = r
        self.n         = M.shape[1]
        self.fc        = fundamental_circuits(M, r)
        self.non_basis = list(range(r, self.n))

        # Count circuits per size class
        sizes          = np.array([len(c) for c in circuits])
        unique_sizes   = sorted(set(sizes.tolist()))
        self.n_by_size = {sz: int(np.sum(sizes == sz)) for sz in unique_sizes}

    def step(self, C):
        eligible_C = [j for j in self.non_basis if self.fc[j] & C]
        if not eligible_C:
            return C

        j       = random.choice(eligible_C)
        sym_fwd = C ^ self.fc[j]

        if not sym_fwd:
            return C

        parts_fwd = decompose_into_circuits(self.M, sym_fwd)
        if not parts_fwd:
            return C

        C_prop = random.choice(parts_fwd)

        # Reverse feasibility
        if not (self.fc[j] & C_prop):
            return C

        sym_rev = C_prop ^ self.fc[j]
        if not sym_rev:
            return C

        parts_rev = decompose_into_circuits(self.M, sym_rev)
        if C not in parts_rev:
            return C

        eligible_prop = [jj for jj in self.non_basis if self.fc[jj] & C_prop]
        q_fwd = 1.0 / (len(eligible_C)    * len(parts_fwd))
        q_rev = 1.0 / (len(eligible_prop) * len(parts_rev))

        # Stratified target correction
        n_C    = self.n_by_size.get(len(C),      1)
        n_Cp   = self.n_by_size.get(len(C_prop), 1)
        # π_strat(C') / π_strat(C) = (π_size(|C'|)/n_{|C'|}) / (π_size(|C|)/n_{|C|})
        # Cross-class factor: n_{|C|} / n_{|C'|}  (π_size cancels only if |C'|=|C|)
        # For same-size: ratio = 1, so α = min(1, q_rev/q_fwd)  ← pure within-class correction
        # For cross-size: ratio = n_C / n_Cp  (approximate; ignores π_size ratio)
        cross_factor = n_C / n_Cp

        ratio = cross_factor * q_rev / q_fwd

        if random.random() < min(1.0, ratio):
            return C_prop
        return C

    def run(self, n_steps, start=None):
        if start is None:
            j     = random.choice(self.non_basis)
            state = self.fc[j]
        else:
            state = start
        traj = [state]
        for _ in range(n_steps):
            state = self.step(state)
            traj.append(state)
        return traj


# ─────────────────────────────────────────────────────────────────────────────
# Within-class irreducibility check
# ─────────────────────────────────────────────────────────────────────────────

def within_class_irreducibility(M, r, circuits, mode='adjacent'):
    """
    For each size class, check whether all circuits are reachable from
    each other using ONLY same-size transitions.

    Returns dict: size -> (n_circuits, n_components, component_sizes)
    """
    from analysis import reachable_from

    fc        = fundamental_circuits(M, r)
    non_basis = list(range(r, M.shape[1]))

    sizes      = np.array([len(c) for c in circuits])
    unique_sz  = sorted(set(sizes.tolist()))
    results    = {}

    for sz in unique_sz:
        class_circuits = [c for c in circuits if len(c) == sz]
        class_set      = set(class_circuits)
        visited_all    = set()
        components     = []

        for start in class_circuits:
            if start in visited_all:
                continue

            # BFS restricted to same-size circuits
            comp    = set()
            queue   = deque([start])
            comp.add(start)

            while queue:
                C = queue.popleft()
                for j in non_basis:
                    if mode == 'adjacent' and not (fc[j] & C):
                        continue
                    sym_diff = C ^ fc[j]
                    if not sym_diff:
                        continue
                    for C_next in decompose_into_circuits(M, sym_diff):
                        if C_next in class_set and C_next not in comp:
                            comp.add(C_next)
                            queue.append(C_next)

            visited_all |= comp
            components.append(len(comp))

        results[sz] = {
            "n_circuits":   len(class_circuits),
            "n_components": len(components),
            "comp_sizes":   sorted(components, reverse=True),
            "irreducible":  len(components) == 1,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Build exact transition matrix for stratified chain
# ─────────────────────────────────────────────────────────────────────────────

def build_stratified_matrix(M, r, circuits):
    fc        = fundamental_circuits(M, r)
    non_basis = list(range(r, M.shape[1]))
    idx       = {c: i for i, c in enumerate(circuits)}
    N         = len(circuits)
    sizes     = np.array([len(c) for c in circuits])
    n_by_size = {int(sz): int(np.sum(sizes == sz)) for sz in set(sizes.tolist())}

    P = np.zeros((N, N), dtype=np.float64)

    for i, C in enumerate(circuits):
        eligible_C = [j for j in non_basis if fc[j] & C]
        if not eligible_C:
            P[i, i] = 1.0
            continue

        n_elig_C = len(eligible_C)

        for j in eligible_C:
            sym_fwd = C ^ fc[j]
            if not sym_fwd:
                P[i, i] += 1.0 / n_elig_C
                continue

            parts_fwd = decompose_into_circuits(M, sym_fwd)
            if not parts_fwd:
                P[i, i] += 1.0 / n_elig_C
                continue

            n_parts_fwd = len(parts_fwd)
            q_fwd = 1.0 / (n_elig_C * n_parts_fwd)

            for C_prop in parts_fwd:
                if not (fc[j] & C_prop):
                    P[i, i] += q_fwd; continue

                sym_rev = C_prop ^ fc[j]
                if not sym_rev:
                    P[i, i] += q_fwd; continue

                parts_rev = decompose_into_circuits(M, sym_rev)
                if C not in parts_rev:
                    P[i, i] += q_fwd; continue

                eligible_prop = [jj for jj in non_basis if fc[jj] & C_prop]
                q_rev = 1.0 / (len(eligible_prop) * len(parts_rev))

                n_C  = n_by_size.get(len(C),      1)
                n_Cp = n_by_size.get(len(C_prop), 1)
                ratio = (n_C / n_Cp) * (q_rev / q_fwd)
                alpha = min(1.0, ratio)

                k = idx.get(C_prop)
                if k is not None:
                    P[i, k] += q_fwd * alpha
                P[i, i]    += q_fwd * (1.0 - alpha)

    P /= P.sum(axis=1, keepdims=True)
    return P


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from asymptotic import build_mh_transition_matrix

    random.seed(42); np.random.seed(42)

    BASE  = dict(k_params=2, C=0.1, gamma=0.0, beta=0.8, start_r=2)
    steps = [10, 14, 18, 22, 26, 30]

    print("=" * 72)
    print(f"{'n_steps':>8} {'N':>6}  {'chain':>12}  {'gap':>8}  "
          f"{'t_mix(0.25)':>12}  {'accept%':>9}")
    print("=" * 72)

    all_rows = []
    for ns in steps:
        random.seed(42); np.random.seed(42)
        params = {**BASE, "n_steps": ns}
        engine = MatroidEngine(**params)
        result = engine.run()
        M, r, n = result["M"], result["r"], result["n"]

        all_c, trunc, _ = all_circuits(M, r, mode='global')
        if trunc: continue
        circuits = sorted(all_c, key=sorted)
        N = len(circuits)

        # Build all three transition matrices
        P_adj  = build_transition_matrix(M, r, circuits, mode='adjacent')
        P_mh   = build_mh_transition_matrix(M, r, circuits)
        P_st   = build_stratified_matrix(M, r, circuits)

        row = {"ns": ns, "N": N, "r": r}
        for label, P in [("adjacent", P_adj), ("MH-uniform", P_mh),
                         ("MH-strat", P_st)]:
            pi  = stationary_distribution(P)
            sp  = spectral_analysis(P)
            tv, _, tmix = tv_distance_curve(P, pi, max_t=500)

            # Acceptance rate: 1 - P[i,i] averaged
            accept = 1.0 - np.diag(P).mean()

            tmix_s = str(tmix) if tmix else ">500"
            print(f"{ns:>8} {N:>6}  {label:>12}  {sp['gap']:>8.4f}  "
                  f"{tmix_s:>12}  {accept:>8.1%}")

            row[label] = {"gap": sp["gap"], "tmix": tmix,
                          "accept": accept, "tv": tv, "pi": pi}

        all_rows.append(row)
        print()

    # ── Within-class irreducibility for last instance ─────────────────────────
    random.seed(42); np.random.seed(42)
    params = {**BASE, "n_steps": 30}
    engine = MatroidEngine(**params)
    result = engine.run()
    M30, r30, _ = result["M"], result["r"], result["n"]
    all_c30, _, _ = all_circuits(M30, r30, mode='global')
    circuits30 = sorted(all_c30, key=sorted)

    print("\nWithin-class irreducibility (adjacent mode, n_steps=30):")
    wc = within_class_irreducibility(M30, r30, circuits30)
    print(f"  {'size':>5}  {'#circuits':>10}  {'#components':>12}  "
          f"{'irreducible':>12}  component sizes")
    for sz, info in wc.items():
        print(f"  {sz:>5}  {info['n_circuits']:>10}  "
              f"{info['n_components']:>12}  {str(info['irreducible']):>12}  "
              f"{info['comp_sizes']}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    COLS = {"adjacent": "#1f77b4", "MH-uniform": "#d62728", "MH-strat": "#2ca02c"}

    # 1. Spectral gap vs log N
    ax1 = fig.add_subplot(gs[0, 0])
    logN = np.log([r["N"] for r in all_rows])
    for label, col in COLS.items():
        gaps = [r[label]["gap"] for r in all_rows]
        ax1.plot(logN, gaps, 'o-', color=col, lw=2, ms=7, label=label)
    ax1.set_xlabel("log(#circuits)", fontsize=11)
    ax1.set_ylabel("Spectral gap", fontsize=11)
    ax1.set_title("Spectral gap vs matroid size", fontsize=12, fontweight='bold')
    ax1.legend(); ax1.grid(alpha=0.3)

    # 2. t_mix vs log N
    ax2 = fig.add_subplot(gs[0, 1])
    for label, col in COLS.items():
        tmixs = [r[label]["tmix"] if r[label]["tmix"] else 500 for r in all_rows]
        ax2.plot(logN, tmixs, 'o-', color=col, lw=2, ms=7, label=label)
    ax2.set_xlabel("log(#circuits)", fontsize=11)
    ax2.set_ylabel("t_mix(0.25)", fontsize=11)
    ax2.set_title("Mixing time vs matroid size", fontsize=12, fontweight='bold')
    ax2.legend(); ax2.grid(alpha=0.3)

    # 3. Acceptance rate vs log N
    ax3 = fig.add_subplot(gs[1, 0])
    for label, col in COLS.items():
        if label == "adjacent": continue
        accepts = [r[label]["accept"] for r in all_rows]
        ax3.plot(logN, accepts, 'o-', color=col, lw=2, ms=7, label=label)
    ax3.set_xlabel("log(#circuits)", fontsize=11)
    ax3.set_ylabel("Acceptance rate", fontsize=11)
    ax3.set_title("MH acceptance rate vs matroid size", fontsize=12, fontweight='bold')
    ax3.legend(); ax3.grid(alpha=0.3)

    # 4. TV curves for largest instance
    ax4 = fig.add_subplot(gs[1, 1])
    t_ax = np.arange(1, 501)
    last = all_rows[-1]
    for label, col in COLS.items():
        tv = last[label]["tv"]
        ax4.plot(t_ax[:len(tv)], tv, color=col, lw=2, label=f"{label}  t_mix={last[label]['tmix']}")
    ax4.axhline(0.25, color='gray', lw=1, ls=':', label='ε=0.25')
    ax4.set_xlim(0, 80); ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Steps t", fontsize=11)
    ax4.set_ylabel("Worst-case TV distance", fontsize=11)
    ax4.set_title(f"TV curves (N={last['N']})", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    fig.suptitle("Adjacent  vs  MH-uniform  vs  MH-stratified",
                 fontsize=14, fontweight='bold')
    out = "markov-circuits/stratified.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out}")
    plt.close()
