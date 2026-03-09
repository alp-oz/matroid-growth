import numpy as np
import random

class MatroidEngine:
    def __init__(self, n_steps=3000, k_params=4, C=0.0001, gamma=0.0, beta=0.8, start_r=7):
        self.n_steps = n_steps
        self.k_params = k_params
        self.C = C
        self.gamma = gamma
        self.beta = beta
        self.curr_r = start_r
        self.columns = []
        # Explicit initialization of active rows
        self.row_usage = np.ones(start_r, dtype=float)

    def run(self):
        # We use t to represent a single action (Step)
        for t in range(1, self.n_steps + 1):
            p_row = min(1.0, self.C * (t**(-self.gamma)))
            
            # EXCLUSIVE: Every step is EITHER a Row OR a Column
            if random.random() < p_row:
                # OPTION A: Discovery (Add 1 Row)
                self.curr_r += 1
                self.row_usage = np.append(self.row_usage, 1.0)
            else:
                # OPTION B: Attachment (Add 1 Column)
                if isinstance(self.k_params, tuple) and self.k_params[0] == "poisson":
                    k = np.random.poisson(self.k_params[1])
                else:
                    k = self.k_params
                
                k = max(1, min(k, self.curr_r))
                
                # Calculate weights with high-precision normalization
                # This prevents the "Rich-get-Richer" effect from being artificially high
                weights = (self.row_usage ** self.beta).astype(np.float64)
                
                sum_w = np.sum(weights)
                if sum_w <= 0:
                    p = np.ones(self.curr_r) / self.curr_r
                else:
                    p = weights / sum_w
                    
                p /= p.sum() # Secondary normalization for float drift
                
                # Pick unique row indices
                selected = np.random.choice(self.curr_r, size=k, replace=False, p=p)
                
                # Store as a sorted list of integers
                self.columns.append(sorted([int(x) for x in selected]))
                
                # Update usage counts
                for r_idx in selected:
                    self.row_usage[int(r_idx)] += 1
                    
        # Return row_usage to fix the KeyError in batch_runner.py
        return {
            "columns": self.columns, 
            "R_final": self.curr_r,
            "row_usage": self.row_usage
        }