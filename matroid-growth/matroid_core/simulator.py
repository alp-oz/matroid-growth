import numpy as np

class MatroidSimulator:
    def __init__(self, n_steps=5000, k=3, C=0.1, gamma=0.0, beta=1.0, swaps=0):
        self.n = n_steps
        self.k = k
        self.C = C
        self.gamma = gamma
        self.beta = beta
        self.swaps = swaps
        
        self.curr_r = 7 
        self.col_to_rows = []
        # Degrees represent the "popularity" of each row
        self.row_degrees = np.ones(n_steps + 100, dtype=float)

    def _get_alpha(self, t):
        # Implementation of a(n) = C * n^(-gamma)
        # We clip it at 1.0 because probability cannot exceed 1
        return min(1.0, self.C * (t**(-self.gamma)))

    def run(self):
        for t in range(1, self.n + 1):
            # 1. Growth Phase
            if np.random.rand() < self._get_alpha(t):
                self.curr_r += 1
            else:
                # 2. Attachment
                # If beta=0, this is Uniform. If beta=1, this is BA.
                weights = self.row_degrees[:self.curr_r] ** self.beta
                p = weights / weights.sum()
                
                # Pick k distinct rows based on weights
                num_to_pick = min(self.k, self.curr_r)
                selected = np.random.choice(self.curr_r, size=num_to_pick, 
                                           replace=False, p=p)
                
                self.col_to_rows.append(list(selected))
                for r_idx in selected:
                    self.row_degrees[r_idx] += 1

            # 3. Refinement Phase
            if self.swaps > 0 and len(self.col_to_rows) > 0:
                self._refine()
        return self

    def _refine(self):
        # Logic remains the same, moving 1s toward hubs
        for _ in range(self.swaps):
            if not self.col_to_rows: break
            c_idx = np.random.randint(len(self.col_to_rows))
            rows_in_c = self.col_to_rows[c_idx]
            if len(rows_in_c) < 2: continue
            
            weak_r = rows_in_c[np.argmin(self.row_degrees[rows_in_c])]
            weights = self.row_degrees[:self.curr_r] ** self.beta
            target_r = np.random.choice(self.curr_r, p=weights/weights.sum())
            
            if target_r not in rows_in_c:
                self.col_to_rows[c_idx].remove(weak_r)
                self.col_to_rows[c_idx].append(target_r)
                self.row_degrees[weak_r] -= 1
                self.row_degrees[target_r] += 1