"""
Circuit intersection graph with dynamic k_t = floor(α·r_t).

γ=0, λ=0.05, α=0.3, n_steps=1000, start_r=10, 5 replicates.
β ∈ {0.5, 1.0, 1.5}.

Circuit intersection graph:
  Vertices = non-basis elements (fundamental circuits).
  Edge (i,j): circuits share at least one basis row.

Adjacency built via clique-cover: for each basis row, all circuits
containing that row form a clique.  Uses scipy sparse + numpy meshgrid
(same approach as pa_circuit_percolation.py — handles large r).

Reports: n_circuits, largest component fraction, mean degree.
Plots:   log-log degree distribution for each β (mean over replicates ± std),
         plus largest component fraction vs replicate.

Saves: circuit_graph_dynk.png
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import os

LAMBDA  = 0.05
GAMMA   = 0.0
ALPHA   = 0.3
BETAS   = [0.5, 1.0, 1.5]
N_STEPS = 1000
START_R = 10
REPS    = 5
COLORS  = ["#2980b9", "#27ae60", "#e74c3c"]
OUT     = os.path.dirname(__file__)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_dynamic_k(n_steps, alpha, C, gamma, beta, start_r, seed):
    rng       = np.random.default_rng(seed)
    curr_r    = start_r
    row_usage = np.ones(start_r, dtype=np.float64)
    supports  = []
    for t in range(1, n_steps + 1):
        p_row = min(1.0, C * (t ** (-gamma)))
        if rng.random() < p_row:
            curr_r += 1
            row_usage = np.append(row_usage, 1.0)
        else:
            k = max(1, int(np.floor(alpha * curr_r)))
            k = min(k, curr_r)
            w = row_usage ** beta;  w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            supports.append(sorted(int(x) for x in sel))
            for idx in sel:
                row_usage[int(idx)] += 1
    return curr_r, supports


# ── Circuit intersection graph ────────────────────────────────────────────────

def circuit_graph_metrics(columns):
    """
    Build circuit intersection graph via clique-cover and return:
      lf     — largest component fraction
      degrees — degree array
    """
    n = len(columns)
    if n < 2:
        return None, None

    row_to_circs = defaultdict(list)
    for j, col in enumerate(columns):
        for row in col:
            row_to_circs[row].append(j)

    src_list, dst_list = [], []
    for circ_list in row_to_circs.values():
        if len(circ_list) < 2:
            continue
        arr = np.array(circ_list, dtype=np.int32)
        ii, jj = np.meshgrid(arr, arr, indexing='ij')
        mask = ii < jj
        src_list.append(ii[mask].ravel())
        dst_list.append(jj[mask].ravel())

    if not src_list:
        return 1.0 / n, np.zeros(n, dtype=np.int32)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    all_r = np.concatenate([src, dst])
    all_c = np.concatenate([dst, src])
    adj = csr_matrix(
        (np.ones(len(all_r), dtype=np.int8), (all_r, all_c)),
        shape=(n, n))
    adj = (adj > 0).astype(np.int8)

    n_comp, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    lf = sizes.max() / n

    degrees = np.array(adj.sum(axis=1)).ravel().astype(np.int32)
    return lf, degrees


# ── Collect data ──────────────────────────────────────────────────────────────

results = {beta: [] for beta in BETAS}

print(f"\n{'β':>5}  {'rep':>4}  {'r':>5}  {'n_circ':>8}  "
      f"{'lf':>6}  {'mean_deg':>9}  {'max_deg':>8}")
print("─" * 55)

for beta in BETAS:
    for rep in range(REPS):
        seed = int(beta * 100) * 1000 + rep
        r, cols = run_dynamic_k(N_STEPS, ALPHA, LAMBDA, GAMMA,
                                beta, START_R, seed)
        lf, degs = circuit_graph_metrics(cols)
        n_circ = len(cols)
        results[beta].append(dict(r=r, n_circ=n_circ, lf=lf, degs=degs))
        print(f"{beta:>5.1f}  {rep:>4d}  {r:>5d}  {n_circ:>8d}  "
              f"{lf:>6.3f}  {np.mean(degs):>9.1f}  {degs.max():>8d}")
    print()


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'β':>5}  {'n_circ':>8}  {'lf_mean':>8}  {'mean_deg':>9}  {'max_deg':>8}")
print("─" * 45)
for beta in BETAS:
    reps = results[beta]
    nc   = np.mean([d["n_circ"]        for d in reps])
    lf   = np.mean([d["lf"]            for d in reps])
    md   = np.mean([np.mean(d["degs"]) for d in reps])
    mx   = np.mean([d["degs"].max()    for d in reps])
    print(f"{beta:>5.1f}  {nc:>8.0f}  {lf:>8.3f}  {md:>9.1f}  {mx:>8.0f}")


# ── Plot: degree distribution ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax1, ax2  = axes

for beta, color in zip(BETAS, COLORS):
    reps = results[beta]

    # Pool all degree arrays across replicates, compute histogram per rep
    all_degs = [d["degs"] for d in reps]
    max_deg  = max(d.max() for d in all_degs)
    bins     = np.logspace(0, np.log10(max_deg + 1), 40)

    # Plot each replicate lightly
    for degs in all_degs:
        counts, edges = np.histogram(degs, bins=bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        mask = counts > 0
        ax1.loglog(mids[mask], counts[mask], '.', color=color, alpha=0.2, ms=4)

    # Mean CCDF (complementary CDF) — more robust on log-log
    ax2_data = []
    for degs in all_degs:
        sorted_d = np.sort(degs)
        ccdf     = 1.0 - np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2_data.append((sorted_d, ccdf))

    # Overlay mean histogram
    counts_all = []
    for degs in all_degs:
        c, _ = np.histogram(degs, bins=bins)
        counts_all.append(c)
    mean_c = np.mean(counts_all, axis=0)
    std_c  = np.std(counts_all,  axis=0)
    mids   = 0.5 * (bins[:-1] + bins[1:])
    mask   = mean_c > 0
    ax1.loglog(mids[mask], mean_c[mask], '-', color=color, lw=2,
               label=fr"$\beta={beta}$  "
                     fr"$(\bar{{d}}={np.mean([np.mean(d['degs']) for d in reps]):.0f})$")

    # CCDF on ax2
    for degs in all_degs:
        sd = np.sort(degs)
        cc = 1.0 - np.arange(1, len(sd)+1) / len(sd)
        ax2.loglog(sd, cc, color=color, alpha=0.25, lw=1)
    # Mean CCDF: interpolate onto common grid
    sd_all = np.sort(np.concatenate([d["degs"] for d in reps]))
    cc_all = 1.0 - np.arange(1, len(sd_all)+1) / len(sd_all)
    ax2.loglog(sd_all, cc_all, color=color, lw=2,
               label=fr"$\beta={beta}$")

ax1.set_xlabel("Degree $d$", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_title("Degree distribution (histogram)", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, which="both", alpha=0.2)

ax2.set_xlabel("Degree $d$", fontsize=12)
ax2.set_ylabel(r"$P(D \geq d)$  (CCDF)", fontsize=12)
ax2.set_title("Degree CCDF", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, which="both", alpha=0.2)

fig.suptitle(
    fr"Circuit intersection graph: $\gamma=0$, $\lambda=0.05$, "
    fr"$\alpha=0.3$, $k_t=\lfloor\alpha r_t\rfloor$, $n_{{\rm steps}}=1000$",
    fontsize=12)
plt.tight_layout()
path = os.path.join(OUT, "circuit_graph_dynk.png")
fig.savefig(path, dpi=150)
print(f"\n  → {path}")
