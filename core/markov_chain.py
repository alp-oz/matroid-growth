import numpy as np
import random
from core.engine import MatroidEngine


# ─────────────────────────────────────────────────────────────────────────────
# GF(2) circuit utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_circuit_in_set(M, S_list):
    """
    Find a circuit contained in S_list using GF(2) Gaussian elimination.

    Tracks dependencies via a coefficient vector so that when a column reduces
    to zero, the set of original columns responsible is returned as a circuit.

    Returns a frozenset of element indices, or None if S_list is independent.
    """
    cols = list(S_list)
    n = len(cols)
    if n == 0:
        return None

    R = M.shape[0]
    reduced = {}   # pivot_row -> (reduced_vector, dependency_vector)

    for j in range(n):
        v = M[:, cols[j]].astype(int).copy()
        d = np.zeros(n, dtype=int)
        d[j] = 1

        # Reduce v using existing pivots
        for row in sorted(reduced):
            if v[row] == 1:
                rv, rd = reduced[row]
                v = (v + rv) % 2
                d = (d + rd) % 2

        # Find pivot row
        pivot = next((row for row in range(R) if v[row] == 1), None)

        if pivot is None:
            # v reduced to zero: dependency found
            return frozenset(cols[i] for i in range(n) if d[i] == 1)

        reduced[pivot] = (v, d)

    return None  # S_list is independent


def decompose_into_circuits(M, S):
    """
    Decompose S (a set known to be in the cycle space) into disjoint circuits.

    Repeatedly finds a circuit within the remaining set and removes it via
    symmetric difference until nothing remains.

    Returns a list of frozensets, each a circuit of the matroid.
    """
    circuits = []
    remaining = set(S)

    while remaining:
        C = find_circuit_in_set(M, list(remaining))
        if C is None:
            break  # remaining is independent — should not happen if S is valid
        circuits.append(C)
        remaining ^= C   # symmetric difference: remove C from remaining

    return circuits


# ─────────────────────────────────────────────────────────────────────────────
# Matroid: extract fundamental circuits from [I | A] matrix
# ─────────────────────────────────────────────────────────────────────────────

def fundamental_circuits(M, r):
    """
    Return a dict mapping each non-basis element index j (>= r) to its
    fundamental circuit: {j} union {i in 0..r-1 : M[i,j] == 1}.
    """
    n = M.shape[1]
    return {
        j: frozenset([j] + [i for i in range(r) if M[i, j] == 1])
        for j in range(r, n)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markov chain on circuits
# ─────────────────────────────────────────────────────────────────────────────

class MarkovChainCircuits:
    """
    Markov chain on all circuits of a binary matroid.

    State: a frozenset of element indices representing a circuit.

    Two transition modes
    -------------------
    'global'   (Direction 1):
        Pick a uniformly random non-basis column j.
        XOR current circuit with circuit(j).
        Decompose into disjoint circuits, pick one uniformly.

    'adjacent' (Direction 2):
        Pick a uniformly random non-basis column j whose fundamental circuit
        shares at least one element with the current state.
        XOR, decompose, pick one uniformly.

    Parameters
    ----------
    M    : binary matrix in [I | A] form (r x n), dtype int
    r    : rank (number of basis / identity columns)
    mode : 'global' or 'adjacent'
    """

    def __init__(self, M, r, mode='global'):
        self.M = M
        self.r = r
        self.n = M.shape[1]
        self.mode = mode
        self.fc = fundamental_circuits(M, r)          # j -> frozenset
        self.non_basis = list(range(r, self.n))

    def _transition(self, current):
        if self.mode == 'global':
            j = random.choice(self.non_basis)
            other = self.fc[j]

        elif self.mode == 'adjacent':
            candidates = [j for j in self.non_basis if self.fc[j] & current]
            if not candidates:
                return current   # no adjacent circuit; stay
            j = random.choice(candidates)
            other = self.fc[j]

        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

        sym_diff = current ^ other   # symmetric difference

        if not sym_diff:
            return current   # same circuit; stay

        parts = decompose_into_circuits(self.M, sym_diff)

        if not parts:
            return current

        return random.choice(parts)

    def run(self, n_steps, start=None):
        """
        Run the chain for n_steps.

        start: a frozenset (circuit) or None to start from a random
               fundamental circuit.

        Returns list of frozensets (trajectory including start state).
        """
        if start is None:
            j = random.choice(self.non_basis)
            state = self.fc[j]
        else:
            state = start

        trajectory = [state]
        for _ in range(n_steps):
            state = self._transition(state)
            trajectory.append(state)

        return trajectory


# ─────────────────────────────────────────────────────────────────────────────
# Metropolis-Hastings corrected chain (targets uniform distribution)
# ─────────────────────────────────────────────────────────────────────────────

class MHCircuitChain:
    """
    Metropolis-Hastings correction of the adjacent chain targeting the
    uniform distribution over all circuits.

    Proposal: identical to adjacent mode — pick eligible j, XOR, pick one
    component C' uniformly.

    Acceptance: min(1, q_reverse / q_forward) where
        q_forward  = 1 / (|eligible(C)|  * |decomp(C  ^ fc[j])|)
        q_reverse  = 1 / (|eligible(C')| * |decomp(C' ^ fc[j])|)
                     (= 0 if j not eligible from C', or C not in decomp)

    By detailed balance this gives stationary distribution π(C) = 1/N
    (uniform over all circuits).

    Optionally target π(C) ∝ |C|^alpha instead of uniform (alpha=0).
    alpha > 0 upweights large circuits further; alpha=0 is uniform.
    """

    def __init__(self, M, r, alpha=0):
        self.M         = M
        self.r         = r
        self.n         = M.shape[1]
        self.fc        = fundamental_circuits(M, r)
        self.non_basis = list(range(r, self.n))
        self.alpha     = alpha   # 0 = uniform, >0 = favour larger circuits

    def _weight(self, C):
        return len(C) ** self.alpha if self.alpha != 0 else 1.0

    def step(self, C):
        eligible_C = [j for j in self.non_basis if self.fc[j] & C]
        if not eligible_C:
            return C

        j        = random.choice(eligible_C)
        sym_fwd  = C ^ self.fc[j]

        if not sym_fwd:
            return C   # same circuit; stay

        parts_fwd = decompose_into_circuits(self.M, sym_fwd)
        if not parts_fwd:
            return C

        C_prop = random.choice(parts_fwd)

        # ── Reverse move feasibility ──────────────────────────────────────────
        # j must be eligible from C_prop
        if not (self.fc[j] & C_prop):
            return C   # j not eligible from C_prop → reject

        sym_rev = C_prop ^ self.fc[j]

        if not sym_rev:
            return C   # C_prop ^ fc[j] = ∅ → stay at C_prop, can't reach C

        parts_rev = decompose_into_circuits(self.M, sym_rev)

        if C not in parts_rev:
            return C   # C not reachable from C_prop via j → reject

        # ── MH acceptance ratio ───────────────────────────────────────────────
        eligible_prop = [jj for jj in self.non_basis if self.fc[jj] & C_prop]

        # q_forward  = 1 / (|eligible_C|  * |parts_fwd|)
        # q_reverse  = 1 / (|eligible_prop| * |parts_rev|)
        # ratio = (w(C_prop) * q_reverse) / (w(C) * q_forward)
        #       = (w(C_prop) * |eligible_C| * |parts_fwd|)
        #         / (w(C) * |eligible_prop| * |parts_rev|)

        ratio = (self._weight(C_prop) * len(eligible_C)   * len(parts_fwd)) / \
                (self._weight(C)      * len(eligible_prop) * len(parts_rev))

        if random.random() < min(1.0, ratio):
            return C_prop
        return C   # rejected; stay

    def run(self, n_steps, start=None):
        if start is None:
            j     = random.choice(self.non_basis)
            state = self.fc[j]
        else:
            state = start

        trajectory = [state]
        for _ in range(n_steps):
            state = self.step(state)
            trajectory.append(state)
        return trajectory


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    engine = MatroidEngine(n_steps=100, k_params=2, C=0.05, beta=0.8, start_r=3)
    result = engine.run()
    M, r, n = result["M"], result["r"], result["n"]

    print(f"Matroid: rank={r}, #elements={n}, #non-basis={n - r}")

    for mode in ("global", "adjacent"):
        chain = MarkovChainCircuits(M, r, mode=mode)
        traj = chain.run(n_steps=20)
        print(f"\nMode: {mode}")
        print(f"  Start : {sorted(traj[0])}")
        for i, c in enumerate(traj[1:], 1):
            print(f"  Step {i:2d}: {sorted(c)}")
