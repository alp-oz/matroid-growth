"""
coset_descent_walk_survey.py — Chain 4 survey across all CSS code families.

Runs Chain 4 (Biased Codeword Walk) with q=0.3 on all instances and reports:
  d_Z, A_logical(z), T_cover, T_converge, coverage

Results are saved incrementally to chain4_results.json.
Re-running skips any instance already present in the file (matched by label).

Instances (ordered cheapest first):
  Toric L=2, L=3, L=4, L=5
  FB-toric(4,4), FB-uniform(4,4), FB-alt(4,4)[0101]
  FB-alt(4,4)[0011], FB-alt(4,4)[0123], FB-alt(4,4)[0202], FB-large(6,4)[010101]
  BB(3,3)
  IBM [[72,12,6]]  (T=2M — takes ~10 min)
"""

import sys
import os
import json
import numpy as np
from chains.codeword_walk import preprocess
from chains.coset_descent_chain import run_chain4
from codes.toric_code import build_toric_hx, build_toric_hz
from codes.fb_code import fb_code
from codes.bb_code import bb_code

OUT_JSON = os.path.join(os.path.dirname(__file__), "chain4_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# Instance list
# ─────────────────────────────────────────────────────────────────────────────

def get_instances():
    instances = []

    def add(label, family, H_X, H_Z, T_max=200000, stability_window=10000):
        ok = np.all((H_X.astype(int) @ H_Z.T.astype(int)) % 2 == 0)
        if not ok:
            print(f"  [{label}] CSS condition FAILED — skipping")
            return
        instances.append({
            "label": label, "family": family,
            "n": int(H_X.shape[1]),
            "H_X": H_X, "H_Z": H_Z,
            "T_max": T_max, "stability_window": stability_window,
        })

    # Toric L=2..5
    for L in [2, 3, 4, 5]:
        add(f"Toric L={L}", "Toric",
            build_toric_hx(L), build_toric_hz(L))

    # Fiber bundle variants
    fb_variants = [
        ("FB-toric(4,4)",         "FB", 4, 4, [0, 0, 0, 0]),
        ("FB-uniform(4,4)",       "FB", 4, 4, [1, 1, 1, 1]),
        ("FB-alt(4,4)[0101]",     "FB", 4, 4, [0, 1, 0, 1]),
        ("FB-alt(4,4)[0011]",     "FB", 4, 4, [0, 0, 1, 1]),
        ("FB-alt(4,4)[0123]",     "FB", 4, 4, [0, 1, 2, 3]),
        ("FB-alt(4,4)[0202]",     "FB", 4, 4, [0, 2, 0, 2]),
        ("FB-large(6,4)[010101]", "FB", 6, 4, [0, 1, 0, 1, 0, 1]),
    ]
    for label, family, r, s, shifts in fb_variants:
        try:
            H_X, H_Z = fb_code(r=r, s=s, shifts=shifts)
            add(label, family, H_X, H_Z)
        except Exception as e:
            print(f"  [{label}] construction failed: {e}")

    # BB(3,3)
    try:
        H_X, H_Z = bb_code(3, 3, [(0,0),(1,0),(0,1)], [(0,0),(2,0),(0,2)])
        add("BB(3,3)", "BB", H_X, H_Z)
    except Exception as e:
        print(f"  [BB(3,3)] construction failed: {e}")

    # IBM [[72,12,6]] — long run
    try:
        H_X, H_Z = bb_code(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])
        add("IBM [[72,12,6]]", "BB", H_X, H_Z,
            T_max=2000000, stability_window=30000)
    except Exception as e:
        print(f"  [IBM [[72,12,6]]] construction failed: {e}")

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_done(path):
    if not os.path.exists(path):
        return {}, []
    with open(path) as f:
        rows = json.load(f)
    return {r["label"]: r for r in rows}, rows


def save_all(path, rows):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def fmt_A(A):
    return "  ".join(f"z^{w}x{c}" for w, c in sorted(A.items()))

def fmt_tc(v):
    return str(v) if v != -1 else ">T_max"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Chain 4 — coset descent walk survey  (q=0.3, seed=42)")
    print("=" * 70)

    done_map, all_rows = load_done(OUT_JSON)
    if done_map:
        print(f"Loaded {len(done_map)} cached results from {OUT_JSON}")

    instances = get_instances()

    for inst in instances:
        label = inst["label"]
        if label in done_map:
            print(f"  {label:<30}  [cached]")
            continue

        T_max = inst["T_max"]
        print(f"  {label:<30}  n={inst['n']}  T_max={T_max//1000}k ...",
              end=" ", flush=True)

        res = run_chain4(inst["H_X"], inst["H_Z"],
                         q=0.3, T_max=T_max,
                         stability_window=inst["stability_window"],
                         seed=42)

        row = {
            "label":      label,
            "family":     inst["family"],
            "n":          inst["n"],
            "k_logical":  res["k_logical"],
            "d_Z":        res["d_Z"],
            "A_logical_z": {str(k): v for k, v in res["A_logical_z"].items()},
            "T_cover":    res["T_cover"],
            "T_converge": res["T_converge"],
            "coverage":   round(res["coverage"], 4),
            "n_visited":  res["n_visited"],
            "n_cosets":   res["n_cosets"],
            "q":          res["q"],
        }
        all_rows.append(row)
        done_map[label] = row
        save_all(OUT_JSON, all_rows)

        if res["k_logical"] == 0:
            print("k_logical=0 — degenerate")
        else:
            print(f"d_Z={res['d_Z']}  k={res['k_logical']}  "
                  f"cov={res['coverage']:.0%}  "
                  f"T_cover={fmt_tc(res['T_cover'])}  "
                  f"T_conv={fmt_tc(res['T_converge'])}")
            print(f"    A_logical: {fmt_A(res['A_logical_z'])}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"{'Code':<26} {'n':>4} {'k':>3} {'cosets':>7} {'d_Z':>5}  {'A_logical(z)'}")
    print("-" * 90)
    for row in all_rows:
        A_str = fmt_A({int(k): v for k, v in row["A_logical_z"].items()})
        dz = str(row["d_Z"]) if row["d_Z"] is not None else "?"
        k  = str(row["k_logical"]) if row["k_logical"] > 0 else "degen"
        print(f"{row['label']:<26} {row['n']:>4} {k:>3} "
              f"{row['n_cosets']:>7} {dz:>5}  {A_str}")

    # ── Structural checks ─────────────────────────────────────────────────────
    print()
    print("Structural checks:")
    for L in [2, 3, 4, 5]:
        r = done_map.get(f"Toric L={L}")
        if r and r["d_Z"]:
            d = r["d_Z"]
            A = {int(k): v for k, v in r["A_logical_z"].items()}
            pattern = (A == {d: 2, 2*d: 1})
            print(f"  Toric L={L}: d_Z={d}  A={A}  "
                  f"2z^d+z^{{2d}}: {'YES' if pattern else 'NO'}")

    fb_tor = done_map.get("FB-toric(4,4)")
    toric4 = done_map.get("Toric L=4")
    if fb_tor and toric4:
        match = (fb_tor["d_Z"] == toric4["d_Z"] and
                 fb_tor["A_logical_z"] == toric4["A_logical_z"])
        print(f"  FB-toric(4,4) == Toric L=4: {'YES' if match else 'NO'}")

    bb33 = done_map.get("BB(3,3)")
    if bb33 and bb33["coverage"] == 1.0:
        A = {int(k): v for k, v in bb33["A_logical_z"].items()}
        uniform = len(A) == 1
        print(f"  BB(3,3): d_Z={bb33['d_Z']}  A={A}  "
              f"{'perfectly uniform' if uniform else 'non-uniform'}")

    print(f"\nResults saved to {OUT_JSON}")
