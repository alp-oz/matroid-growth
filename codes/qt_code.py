"""
Quantum Tanner codes from left-right Cayley complexes.

Construction (Leverrier & Zémor 2022, simplified):
  Group G with symmetric left generators A and right generators B.

  Left-right Cayley complex:
    Vertices : G
    Left  edges : (g, a)  connecting g  →  g·a    for g ∈ G, a ∈ A
    Right edges : (g, b)  connecting g  →  b·g    for g ∈ G, b ∈ B
    Faces       : (g, a, b) — the square g, g·a, b·g, b·g·a

  CSS code with qubits = edges:
    H_X = ∂_2  : face boundary  (each face row has 4 nonzero entries)
    H_Z = ∂_1^T: vertex coboundary  (each vertex row = all incident edges,
                 both incoming and outgoing)

  CSS condition H_X · H_Z^T = 0 holds by ∂_1 ∘ ∂_2 = 0 (chain complex identity).

  n = |G|·(|A| + |B|)    qubits
  k = n - rank(H_X) - rank(H_Z)

Group:
  G = PSL(2,p):  2×2 matrices over F_p with det=1, modulo ±I.
  PSL(2,3) ≅ A_4 : order 12  (weak expansion, small n, good for sanity checks)
  PSL(2,5) ≅ A_5 : order 60  (good expansion, standard Quantum Tanner group)

Generators:
  s = [[1,1],[0,1]] (parabolic element, order p)
  t = [[0,p-1],[1,0]] (Weyl element, order 2 or 4)
  Use symmetric sets A = {s, s^{-1}} or A = {s, s^{-1}, t}.
"""

import numpy as np
from itertools import product as iproduct


# ─────────────────────────────────────────────────────────────────────────────
# PSL(2, p) group arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def mat_mul_mod(A, B, p):
    return np.array(A, dtype=int) @ np.array(B, dtype=int) % p


def mat_inv_mod(A, p):
    a, b, c, d = int(A[0,0]), int(A[0,1]), int(A[1,0]), int(A[1,1])
    det_inv = pow((a*d - b*c) % p, -1, p)
    return np.array([[d, -b], [-c, a]], dtype=int) * det_inv % p


def mat_key(A, p):
    return tuple(int(x) % p for x in A.flatten())


def psl2_elements(p):
    """Enumerate all |PSL(2,p)| elements as canonical 2×2 arrays over F_p."""
    seen = set()
    elements = []
    for a, b, c, d in iproduct(range(p), repeat=4):
        if (a*d - b*c) % p != 1:
            continue
        M   = np.array([[a, b], [c, d]], dtype=int)
        k1  = mat_key(M, p)
        k2  = mat_key((-M) % p, p)
        canon = min(k1, k2)
        if canon not in seen:
            seen.add(canon)
            rep = M if k1 == canon else (-M) % p
            elements.append(rep)
    return elements


def build_psl2(p):
    """
    Build PSL(2,p): element list + index map + multiplication table.
    Returns (elems, elem_index, mul_table).
    """
    elems = psl2_elements(p)
    elem_index = {}
    for idx, M in enumerate(elems):
        elem_index[mat_key(M, p)]       = idx
        elem_index[mat_key((-M) % p, p)] = idx

    G = len(elems)
    mul_table = np.zeros((G, G), dtype=np.int32)
    for i, A in enumerate(elems):
        for j, B in enumerate(elems):
            C   = mat_mul_mod(A, B, p)
            k   = mat_key(C, p)
            mul_table[i, j] = elem_index[k]
    return elems, elem_index, mul_table


def find_gen_index(mat, p, elem_index):
    """Return index of mat in PSL(2,p), or None if not found."""
    k = mat_key(mat % p, p)
    if k in elem_index:
        return elem_index[k]
    k2 = mat_key((-mat) % p, p)
    return elem_index.get(k2)


# ─────────────────────────────────────────────────────────────────────────────
# Quantum Tanner CSS code
# ─────────────────────────────────────────────────────────────────────────────

def qt_code(mul_table, A_idx, B_idx):
    """
    Build Quantum Tanner CSS code from left-right Cayley complex.

    Parameters
    ----------
    mul_table : (|G|, |G|) int array   — group multiplication table
    A_idx     : list of ints           — left  generator indices (symmetric: a⁻¹ ∈ A_idx)
    B_idx     : list of ints           — right generator indices (symmetric: b⁻¹ ∈ B_idx)

    Qubit layout:
      Left  qubits l_idx(g, ai) = g*nA + ai       for g ∈ G, ai ∈ 0..nA-1
      Right qubits r_idx(g, bi) = n_L + g*nB + bi for g ∈ G, bi ∈ 0..nB-1

    Left  edge (g, a):  tail = g,  head = g·a  = mul_table[g, a]
    Right edge (g, b):  tail = g,  head = b·g  = mul_table[b, g]

    H_X = ∂_2 (face boundary):
      Face (g, ai, bi) has corners g, g·a, b·g, b·g·a.
      Boundary edges: l_idx(g, ai), l_idx(b·g, ai), r_idx(g, bi), r_idx(g·a, bi).

    H_Z = ∂_1^T (vertex coboundary):
      Edge is incident to BOTH its tail and head vertex.
      l_idx(g, ai): incident to vertices g  and  g·a
      r_idx(g, bi): incident to vertices g  and  b·g
    """
    G  = mul_table.shape[0]
    nA = len(A_idx)
    nB = len(B_idx)
    n_L = G * nA
    n_R = G * nB
    n   = n_L + n_R

    def l_idx(g, ai): return g * nA + ai
    def r_idx(g, bi): return n_L + g * nB + bi

    # ── H_X: one row per face (g, ai, bi) ────────────────────────────────────
    H_X = np.zeros((G * nA * nB, n), dtype=np.uint8)
    for g in range(G):
        for ai, a in enumerate(A_idx):
            ga  = mul_table[g, a]          # g·a  (head of left edge)
            for bi, b in enumerate(B_idx):
                bg  = mul_table[b, g]      # b·g  (head of right edge)
                face = (g * nA + ai) * nB + bi
                H_X[face, l_idx(g,  ai)] ^= 1   # left edge tail g
                H_X[face, l_idx(bg, ai)] ^= 1   # left edge tail b·g
                H_X[face, r_idx(g,  bi)] ^= 1   # right edge tail g
                H_X[face, r_idx(ga, bi)] ^= 1   # right edge tail g·a
    H_X %= 2

    # ── H_Z: one row per vertex, both tail and head incidences ───────────────
    H_Z = np.zeros((G, n), dtype=np.uint8)
    for g in range(G):
        for ai, a in enumerate(A_idx):
            ga = mul_table[g, a]
            H_Z[g,  l_idx(g,  ai)] ^= 1   # g  is tail of left edge (g, a)
            H_Z[ga, l_idx(g,  ai)] ^= 1   # ga is head of left edge (g, a)
        for bi, b in enumerate(B_idx):
            bg = mul_table[b, g]
            H_Z[g,  r_idx(g,  bi)] ^= 1   # g  is tail of right edge (g, b)
            H_Z[bg, r_idx(g,  bi)] ^= 1   # bg is head of right edge (g, b)
    H_Z %= 2

    # ── CSS check ─────────────────────────────────────────────────────────────
    check = (H_X.astype(int) @ H_Z.T.astype(int)) % 2
    assert np.all(check == 0), \
        f"CSS condition H_X·H_Z^T ≠ 0!  max={check.max()}"

    return H_X, H_Z


def gf2_rank(H):
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


def qt_params(mul_table, A_idx, B_idx):
    H_X, H_Z = qt_code(mul_table, A_idx, B_idx)
    n = H_X.shape[1]
    k = n - gf2_rank(H_X) - gf2_rank(H_Z)
    return {"n": n, "k": k, "H_X": H_X, "H_Z": H_Z}


# ─────────────────────────────────────────────────────────────────────────────
# Instance builder
# ─────────────────────────────────────────────────────────────────────────────

def qt_instances():
    """
    Return list of (label, H_X, H_Z, n, k) instances.
    Uses PSL(2,3) (|G|=12) and PSL(2,5) (|G|=60) with symmetric generator sets.
    """
    instances = []

    for p in [3, 5]:
        elems, elem_index, mul_table = build_psl2(p)
        G = len(elems)
        print(f"\nPSL(2,{p}): |G|={G}")

        # Standard generators
        s_mat    = np.array([[1, 1], [0, 1]])
        t_mat    = np.array([[0, p-1], [1, 0]])
        s_inv_mat = mat_inv_mod(s_mat, p)

        s_idx    = find_gen_index(s_mat, p, elem_index)
        t_idx    = find_gen_index(t_mat, p, elem_index)
        si_idx   = find_gen_index(s_inv_mat, p, elem_index)

        if None in (s_idx, t_idx, si_idx):
            print(f"  Could not find generators, skipping.")
            continue

        # t has order 2, so t^{-1} = t (or t^{-1} is also in elem_index)
        t_inv_idx = find_gen_index(mat_inv_mod(t_mat, p), p, elem_index)

        gen_sets = [
            ([s_idx, si_idx], [s_idx, si_idx], f"|A|=2,|B|=2"),
            ([s_idx, si_idx, t_idx], [s_idx, si_idx, t_idx], f"|A|=3,|B|=3"),
        ]
        if t_inv_idx is not None and t_inv_idx != t_idx:
            gen_sets.insert(0, ([s_idx, si_idx], [t_idx, t_inv_idx], f"|A|=2,|B|=2(t)"))

        for A_idx, B_idx, glab in gen_sets:
            label = f"QT-PSL(2,{p}) {glab}"
            n_expected = G * (len(A_idx) + len(B_idx))
            if n_expected > 500:
                print(f"  {label:<38}  n={n_expected} — too large, skip")
                continue
            try:
                res = qt_params(mul_table, A_idx, B_idx)
                n, k = res["n"], res["k"]
                instances.append((label, res["H_X"], res["H_Z"], n, k))
                print(f"  {label:<38}  n={n:>4}  k={k:>3}  ✓")
            except AssertionError as e:
                print(f"  {label:<38}  CSS failed: {e}")

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("Quantum Tanner code parameters  (left-right Cayley complex)")
    print("=" * 65)

    instances = qt_instances()

    print("\n" + "=" * 65)
    print(f"  {'label':<38}  {'n':>5}  {'k':>4}")
    print("=" * 65)
    rows = []
    for label, H_X, H_Z, n, k in instances:
        print(f"  {label:<38}  {n:>5}  {k:>4}")
        rows.append({"label": label, "n": n, "k": k})

    out = "markov-circuits/qt_params.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved → {out}")
