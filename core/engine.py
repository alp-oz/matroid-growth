import numpy as np
import random


class MatroidEngine:
    """
    Binary matroid generator via preferential attachment.

    Produces a matrix in clean [I | A] form:
      - Basis elements: labeled 0..r-1, one per discovery event
      - Non-basis elements: labeled r..n-1, one per attachment event

    At each step:
      Discovery  (prob p_row = C * t^-gamma):
          new independent element — expands basis by 1.
      Attachment (prob 1 - p_row):
          new element whose support in the basis is drawn by preferential
          attachment (weighted by usage^beta). May form circuits.

    Parameters
    ----------
    n_steps  : total number of steps
    k_params : int or ("poisson", lam) — #basis elements each attachment picks
    C        : scales discovery probability
    gamma    : decay exponent for discovery rate
    beta     : preferential attachment strength (0=uniform, 1=linear PA)
    start_r  : initial number of basis elements
    """

    def __init__(self, n_steps=3000, k_params=2, C=0.05, gamma=0.0, beta=0.8, start_r=3):
        self.n_steps = n_steps
        self.k_params = k_params
        self.C = C
        self.gamma = gamma
        self.beta = beta
        self.curr_r = start_r
        self.attachment_supports = []       # list of sorted row-index lists
        self.row_usage = np.ones(start_r, dtype=float)

    def run(self):
        for t in range(1, self.n_steps + 1):
            p_row = min(1.0, self.C * (t ** (-self.gamma)))

            if random.random() < p_row:
                # Discovery: new independent (basis) element
                self.curr_r += 1
                self.row_usage = np.append(self.row_usage, 1.0)
            else:
                # Attachment: new element, support sampled by preferential attachment
                if isinstance(self.k_params, tuple) and self.k_params[0] == "poisson":
                    k = np.random.poisson(self.k_params[1])
                else:
                    k = self.k_params

                k = max(1, min(k, self.curr_r))

                weights = (self.row_usage ** self.beta).astype(np.float64)
                p = weights / weights.sum()
                p /= p.sum()

                selected = np.random.choice(self.curr_r, size=k, replace=False, p=p)
                self.attachment_supports.append(sorted(int(x) for x in selected))
                for idx in selected:
                    self.row_usage[int(idx)] += 1

        r = self.curr_r
        n_att = len(self.attachment_supports)
        n = r + n_att

        # Build [I | A]: rows = basis elements, cols 0..r-1 = identity, cols r..n-1 = attachment
        M = np.zeros((r, n), dtype=np.int8)
        M[:, :r] = np.eye(r, dtype=np.int8)
        for j, support in enumerate(self.attachment_supports):
            for i in support:
                M[i, r + j] = 1

        return {
            "M": M,
            "r": r,                               # rank = #basis elements
            "R_final": r,                         # alias for legacy scripts
            "n": n,                               # total #elements
            "attachment_supports": self.attachment_supports,
            "columns": self.attachment_supports,  # alias for legacy scripts
            "row_usage": self.row_usage,
        }
