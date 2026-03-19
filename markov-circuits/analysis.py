import numpy as np
import random
from collections import deque, Counter
from engine import MatroidEngine
from markov_chain import (
    MarkovChainCircuits,
    fundamental_circuits,
    decompose_into_circuits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reachability: build transition graph via BFS
# ─────────────────────────────────────────────────────────────────────────────

def reachable_from(M, r, start_circuit, mode, max_circuits=5000):
    """
    BFS over the circuit graph to find all circuits reachable from
    start_circuit under the given mode.

    An edge C -> C' exists if C' is one of the possible outcomes of a single
    transition step from C (i.e. C' appears in the decomposition of C ^ fc[j]
    for some eligible j).

    Note: since C ^ fc[j] ^ fc[j] = C, all edges are symmetric, so
    reachability = connected component.

    Parameters
    ----------
    max_circuits : stop BFS early if this many circuits are found (cap for
                   large matroids where the circuit set is exponential).

    Returns
    -------
    visited   : set of frozensets (all reachable circuits including start)
    truncated : bool, True if BFS was stopped early
    """
    fc = fundamental_circuits(M, r)
    non_basis = list(range(r, M.shape[1]))

    visited = set()
    queue = deque([start_circuit])
    visited.add(start_circuit)
    truncated = False

    while queue:
        if len(visited) >= max_circuits:
            truncated = True
            break

        C = queue.popleft()

        for j in non_basis:
            if mode == 'adjacent' and not (fc[j] & C):
                continue

            sym_diff = C ^ fc[j]
            if not sym_diff:
                continue

            for C_next in decompose_into_circuits(M, sym_diff):
                if C_next not in visited:
                    visited.add(C_next)
                    queue.append(C_next)

    return visited, truncated


def check_irreducibility(M, r, mode, n_spot_checks=10, max_circuits=5000, seed=None):
    """
    Check whether the chain is irreducible within its connected component.

    Strategy:
      1. Pick a random starting circuit; BFS to find its connected component S.
      2. Spot-check: for n_spot_checks random circuits in S, verify their
         component has the same size.
      If all agree → likely irreducible within S.

    Parameters
    ----------
    max_circuits : BFS cap per run (see reachable_from).
    """
    if seed is not None:
        random.seed(seed)

    fc = fundamental_circuits(M, r)
    non_basis = list(range(r, M.shape[1]))

    start = fc[random.choice(non_basis)]
    component, truncated = reachable_from(M, r, start, mode, max_circuits)
    component_size = len(component)

    print(f"  Starting circuit   : {sorted(start)}")
    print(f"  Component size     : {component_size} circuits"
          + (" (BFS truncated at cap)" if truncated else ""))

    if truncated:
        print(f"  Irreducible (est.) : unknown — BFS hit cap of {max_circuits}")
        return {
            "component_size": component_size,
            "irreducible": None,
            "truncated": True,
            "start_circuit": start,
            "component": component,
        }

    # Spot-check
    samples = random.sample(list(component), min(n_spot_checks, component_size))
    mismatches = 0
    for c in samples:
        sub_comp, _ = reachable_from(M, r, c, mode, max_circuits)
        if len(sub_comp) != component_size:
            mismatches += 1

    irreducible = (mismatches == 0)
    print(f"  Spot-checks        : {len(samples)} sampled")
    print(f"  Mismatches         : {mismatches}")
    print(f"  Irreducible (est.) : {irreducible}")

    return {
        "component_size": component_size,
        "irreducible": irreducible,
        "truncated": False,
        "start_circuit": start,
        "component": component,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Element frequency along a trajectory
# ─────────────────────────────────────────────────────────────────────────────

def element_frequency(trajectory):
    """
    Count how often each element appears across all circuits in the trajectory.

    Returns a Counter: element_index -> visit count.
    """
    freq = Counter()
    for circuit in trajectory:
        freq.update(circuit)
    return freq


def print_frequency(freq, r, top_k=20):
    """Pretty-print element frequencies, labeling basis vs non-basis."""
    print(f"\n  {'Element':>8}  {'Type':>10}  {'Count':>8}")
    print("  " + "-" * 30)
    for elem, count in freq.most_common(top_k):
        kind = "basis" if elem < r else "non-basis"
        print(f"  {elem:>8}  {kind:>10}  {count:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Total circuit count: BFS from every fundamental circuit
# ─────────────────────────────────────────────────────────────────────────────

def all_circuits(M, r, mode, max_circuits=50000):
    """
    Find ALL circuits reachable from ANY fundamental circuit under the given mode.

    Runs BFS seeded by every fundamental circuit and unions the results.
    In global mode this should equal the complete set of circuits of the matroid
    (since every circuit is reachable from at least one fundamental circuit).

    Returns
    -------
    all_found  : set of all discovered circuits (frozensets)
    truncated  : bool
    components : list of (start_fc, component_size) — one entry per fc
                 that started a new component
    """
    fc = fundamental_circuits(M, r)
    non_basis = list(range(r, M.shape[1]))

    all_found = set()
    components = []
    truncated = False

    for j in non_basis:
        start = fc[j]
        if start in all_found:
            continue   # already covered by a previous BFS

        comp, trunc = reachable_from(M, r, start, mode, max_circuits)
        new = comp - all_found
        all_found |= comp
        components.append((sorted(start), len(comp)))
        if trunc:
            truncated = True

    return all_found, truncated, components


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Small matroid so BFS over circuits is tractable
    engine = MatroidEngine(n_steps=30, k_params=2, C=0.1, beta=0.8, start_r=2)
    result = engine.run()
    M, r, n = result["M"], result["r"], result["n"]
    print(f"Matroid: rank={r}, #elements={n}, #non-basis={n - r}\n")

    for mode in ("global", "adjacent"):
        print(f"{'='*55}")
        print(f"  Mode: {mode}")
        print(f"{'='*55}")

        # --- Irreducibility ---
        info = check_irreducibility(M, r, mode, n_spot_checks=20, seed=42)

        # --- Total circuit count ---
        print(f"\n  Counting ALL circuits (seeding BFS from every fc)...")
        all_c, trunc, comps = all_circuits(M, r, mode)
        print(f"  Total circuits found : {len(all_c)}"
              + (" (truncated)" if trunc else ""))
        print(f"  Distinct components  : {len(comps)}")
        if len(comps) > 1:
            for fc_start, sz in comps:
                print(f"    component from {fc_start}: size {sz}")

        # --- Element frequency ---
        chain = MarkovChainCircuits(M, r, mode=mode)
        start = info["start_circuit"]
        traj = chain.run(n_steps=5000, start=start)

        freq = element_frequency(traj)
        print(f"\n  Top-20 most visited elements (over {len(traj)} states):")
        print_frequency(freq, r, top_k=20)
        print()
