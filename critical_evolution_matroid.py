"""
MODEL: Evolution of Random Representable Matroids (Altschuler-Hanany 2024)
FILENAME: critical_evolution_matroid.py
---------------------------------------------------------
MATHEMATICAL SUMMARY:
1. PROCESS: Sequential addition of random vectors v_i from GF(q)^d.
2. CIRCUIT TRACKING: Identifying the 'Hitting Time' tau_1, the first 
   index n where the set {v_1, ..., v_n} becomes dependent.
3. CONNECTIVITY: The process tracks when the matroid becomes 
   colinearly connected.
4. CRITICAL PARAMETER: Monitoring the ratio n/q^(d-k).
---------------------------------------------------------
"""
# %%
import numpy as np

class CriticalEvolutionMatroid:
    # region Initialization
    def __init__(self, q, d):
        self.q = q
        self.d = d
        self.columns = []
        self.n = 0
        self.first_circuit_at = None
    # endregion

    # region Linear Algebra Oracle
    def check_independence(self, new_vector):
        """Checks if the new vector is independent of ALL previous columns"""
        if not self.columns:
            return any(x != 0 for x in new_vector)
            
        # Form matrix to check rank
        test_matrix = np.array(self.columns + [new_vector])
        r = self._get_gf_rank(test_matrix)
        return r == len(self.columns) + 1

    def _get_gf_rank(self, matrix):
        """GF(q) Gaussian elimination to find rank"""
        mat = matrix.copy().astype(int)
        rows, cols = mat.shape
        pivot_row = 0
        for j in range(cols):
            if pivot_row < rows:
                pivots = np.where(mat[pivot_row:, j] % self.q != 0)[0]
                if len(pivots) > 0:
                    i = pivots[0] + pivot_row
                    mat[[pivot_row, i]] = mat[[i, pivot_row]]
                    inv = pow(int(mat[pivot_row, j]), self.q - 2, self.q)
                    mat[pivot_row] = (mat[pivot_row] * inv) % self.q
                    for k in range(rows):
                        if k != pivot_row:
                            mat[k] = (mat[k] - mat[k, j] * mat[pivot_row]) % self.q
                    pivot_row += 1
        return pivot_row
    # endregion

    # region Evolution Step
    def step(self):
        """Adds one random vector and checks for the critical transition"""
        self.n += 1
        v_t = np.random.randint(0, self.q, size=self.d).tolist()
        
        is_indep = self.check_independence(v_t)
        
        if not is_indep and self.first_circuit_at is None:
            self.first_circuit_at = self.n
            
        self.columns.append(v_t)
        return v_t, is_indep
    # endregion

# --- EXECUTION ---
if __name__ == "__main__":
    # Parameters following the paper's logic
    Q_FIELD = 2
    D_DIM = 5
    
    process = CriticalEvolutionMatroid(Q_FIELD, D_DIM)
    
    print(f"Starting Evolution in GF({Q_FIELD})^{D_DIM}...")
    print(f"{'n':<5} | {'Vector':<15} | {'Independent?':<12} | {'State'}")
    print("-" * 55)

    for _ in range(20):
        vec, ind = process.step()
        state = "FREE" if process.first_circuit_at is None else "DEPENDENT"
        print(f"{process.n:<5} | {str(vec):<15} | {str(ind):<12} | {state}")
        
        if process.n == process.first_circuit_at:
            print(f">>> CRITICAL POINT: First Circuit formed at n={process.n}")

    final_r = process._get_gf_rank(np.array(process.columns))
    print(f"\nFinal Rank: {final_r}/{D_DIM}")