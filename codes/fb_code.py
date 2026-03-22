"""
Fiber Bundle CSS codes as binary matroids.

Construction (Hastings, Haah, O'Brien 2021 — simplified):
  Base graph  G_B = cycle C_r  (vertices 0..r-1, edge i = (i → (i+1) mod r))
  Fiber graph G_F = cycle C_s  (vertices 0..s-1, edge j = (j → (j+1) mod s))
  Connection  σ = {k_i}_{i=0}^{r-1}  where k_i ∈ Z_s is the cyclic shift
              applied to fiber vertices when crossing base edge i.

Cell complex:
  0-cells (vertices): V_B × V_F                     — r·s total
  1-cells (qubits):
    horizontal  H[i,v] at base edge i, fiber vertex v — r·s total  (index: i*s+v)
    vertical    V[u,j] at base vertex u, fiber edge j  — r·s total  (index: r*s+u*s+j)
  2-cells (faces):    E_B × E_F                      — r·s total

CSS code:
  H_X[face, qubit] = 1 if qubit is on boundary of face  (∂_2)
  H_Z[vertex, qubit] = 1 if qubit is incident to vertex  (∂_1^T)
  CSS condition: H_X · H_Z^T = 0  ⟺  ∂_1 ∘ ∂_2 = 0  (holds by construction)

Parameters:
  n = 2·r·s  (physical qubits)
  k = computed from ranks of H_X, H_Z
  d ≥ min(r, s)  (for identity connection: equals toric code)

Special cases:
  All k_i = 0               →  toric code on C_r × C_s  (baseline)
  All k_i = k (same shift)  →  bivariate bicycle code
  Mixed / random k_i        →  genuine fiber bundle (non-abelian structure)
"""

import numpy as np
import random


def fb_code(r, s, shifts):
    """
    Build fiber bundle CSS code from cycle C_r (base) and C_s (fiber).

    shifts : list of length r, shifts[i] = cyclic shift k_i ∈ {0,...,s-1}
             applied at base edge i.

    Returns (H_X, H_Z) as uint8 arrays of shape (r*s, 2*r*s).
    """
    # ── Qubit indices ─────────────────────────────────────────────────────────
    def h_idx(i, v):
        """Horizontal qubit: base edge i, fiber vertex v."""
        return i * s + v

    def v_idx(u, j):
        """Vertical qubit: base vertex u, fiber edge j."""
        return r * s + u * s + j

    n_qubits = 2 * r * s
    n_checks = r * s   # same for X and Z

    H_X = np.zeros((n_checks, n_qubits), dtype=np.uint8)
    H_Z = np.zeros((n_checks, n_qubits), dtype=np.uint8)

    # ── H_X: face boundaries (∂_2) ───────────────────────────────────────────
    # Face (i, j): base edge i = (a → (a+1)%r), fiber edge j = (j → (j+1)%s)
    # Boundary (mod 2):
    #   1. H[i, j]           horizontal at (base-edge i, fiber-vertex j)
    #   2. H[i, (j+1)%s]     horizontal at (base-edge i, fiber-vertex (j+1)%s)
    #   3. V[a, j]            vertical at (base-vertex a, fiber-edge j)
    #   4. V[(a+1)%r, (j+k_i)%s]  vertical at (base-vertex (a+1)%r,
    #                               twisted fiber-edge (j + shift[i]) % s)
    for i in range(r):
        a = i                    # tail of base edge i
        k = shifts[i]            # cyclic shift at base edge i
        for j in range(s):
            face = i * s + j
            H_X[face, h_idx(i, j)]               = 1  # bottom horizontal
            H_X[face, h_idx(i, (j + 1) % s)]     = 1  # top horizontal
            H_X[face, v_idx(a, j)]               = 1  # left vertical
            H_X[face, v_idx((a + 1) % r, (j + k) % s)] = 1  # right vertical (twisted)
    H_X %= 2

    # ── H_Z: vertex coboundaries (∂_1^T) ─────────────────────────────────────
    # Vertex (u, v): incident qubits are:
    #   Horizontal: H[u, v]  (u,v is the tail)
    #               H[(u-1)%r, (v - k_{u-1}) % s]  (u,v is the head: traversing
    #               base edge (u-1) from fiber vertex (v-k_{u-1}) shifts by k_{u-1})
    #   Vertical:   V[u, (v-1)%s]  and  V[u, v]  (standard fiber edges)
    for u in range(r):
        for v in range(s):
            vtx = u * s + v
            k_prev = shifts[(u - 1) % r]          # shift at incoming base edge
            H_Z[vtx, h_idx((u - 1) % r, (v - k_prev) % s)] = 1  # twisted incoming
            H_Z[vtx, h_idx(u, v)]                             = 1  # outgoing
            H_Z[vtx, v_idx(u, (v - 1) % s)]                  = 1
            H_Z[vtx, v_idx(u, v)]                             = 1
    H_Z %= 2

    # ── CSS check ─────────────────────────────────────────────────────────────
    check = (H_X.astype(int) @ H_Z.T.astype(int)) % 2
    assert np.all(check == 0), "CSS condition H_X · H_Z^T ≠ 0!"

    return H_X, H_Z


def gf2_rank(H):
    H = H.copy().astype(np.uint8)
    m, n = H.shape
    rank = 0
    for col in range(n):
        pivot = next((row for row in range(rank, m) if H[row, col]), None)
        if pivot is None:
            continue
        H[[rank, pivot]] = H[[pivot, rank]]
        for row in range(m):
            if row != rank and H[row, col]:
                H[row] = (H[row] + H[rank]) % 2
        rank += 1
    return rank


def fb_params(r, s, shifts):
    """Return [[n, k]] parameters of the fiber bundle code."""
    H_X, H_Z = fb_code(r, s, shifts)
    n = H_X.shape[1]
    k = n - gf2_rank(H_X) - gf2_rank(H_Z)
    return {"n": n, "k": k, "H_X": H_X, "H_Z": H_Z}


def fb_instances():
    """
    Return list of (label, r, s, shifts) tuples for study.

    Three families:
      1. Identity (all shifts=0)  →  toric code baseline
      2. Uniform shift            →  bivariate bicycle variant
      3. Alternating / random     →  genuine fiber bundle (non-abelian)
    """
    instances = []

    # ── Baseline: identity connection = toric code ────────────────────────────
    for r, s in [(3, 3), (4, 4), (5, 5)]:
        instances.append((f"FB-toric({r},{s})", r, s, [0] * r))

    # ── Uniform shift k=1: bivariate bicycle variant ──────────────────────────
    for r, s in [(3, 3), (4, 4), (5, 5), (6, 6)]:
        instances.append((f"FB-uniform({r},{s}) k=1", r, s, [1] * r))

    # ── Alternating shifts: non-abelian structure ─────────────────────────────
    # Pattern: [0, 1, 0, 1, ...] — alternates between two shifts
    for r, s in [(4, 4), (5, 5), (6, 6), (8, 6)]:
        shifts = [i % 2 for i in range(r)]
        instances.append((f"FB-alt({r},{s}) [0,1,...]", r, s, shifts))

    # ── Random shifts (seeded): genuine non-abelian ───────────────────────────
    rng = np.random.default_rng(42)
    for r, s in [(4, 4), (5, 5), (6, 6)]:
        shifts = rng.integers(0, s, size=r).tolist()
        instances.append((f"FB-rand({r},{s}) seed=42", r, s, shifts))

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Main: print parameters for all instances
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("Fiber Bundle code parameters")
    print("=" * 65)
    print(f"  {'label':<35}  {'n':>4}  {'k':>4}  {'shifts'}")
    print("=" * 65)

    rows = []
    for label, r, s, shifts in fb_instances():
        p = fb_params(r, s, shifts)
        print(f"  {label:<35}  {p['n']:>4}  {p['k']:>4}  {shifts}")
        rows.append({"label": label, "r": r, "s": s,
                     "n": p["n"], "k": p["k"], "shifts": shifts})

    out = "markov-circuits/fb_params.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved → {out}")
