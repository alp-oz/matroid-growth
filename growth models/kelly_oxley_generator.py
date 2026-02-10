"""
MODEL: Kelly-Oxley Random Representable Matroid
FILENAME: kelly_oxley_generator.py
---------------------------------------------------------
MATHEMATICAL SUMMARY:
1. FIELD (q): The matrix is defined over GF(q). q must be a prime.
2. DIMENSION (d): Fixed number of rows.
3. GROWTH (n): The number of columns grows toward infinity.
4. GENERATION: Each column is a random vector v in GF(q)^d.
---------------------------------------------------------
"""
# %%
import numpy as np

class KellyOxleyMatroid:
    def __init__(self, q, d):
        # --- MODIFY THESE PARAMETERS EASILY ---
        self.q = q  # Field order (must be prime for this basic implementation)
        self.d = d  # Fixed rank/dimension (rows)
        # ---------------------------------------
        
        self.matrix = np.empty((self.d, 0), dtype=int)
        self.n = 0

    def add_columns(self, num_to_add=1):
        """Generates and appends random columns from GF(q)^d"""
        # Generate random integers in the range [0, q-1]
        new_cols = np.random.randint(0, self.q, size=(self.d, num_to_add))
        self.matrix = np.hstack([self.matrix, new_cols])
        self.n += num_to_add

    def get_current_rank(self):
        """Calculates rank over GF(q) using Gaussian Elimination"""
        if self.n == 0: return 0
        
        # We work on a copy to preserve the original matrix
        mat = self.matrix.copy().astype(float)
        rows, cols = mat.shape
        pivot_row = 0
        
        for j in range(cols):
            if pivot_row < rows:
                # Find pivot in column j
                pivot = np.where(mat[pivot_row:, j] % self.q != 0)[0]
                if len(pivot) > 0:
                    i = pivot[0] + pivot_row
                    # Swap rows
                    mat[[pivot_row, i]] = mat[[i, pivot_row]]
                    
                    # Modular Inverse for GF(q)
                    # For GF(2), this is always 1. For others, we find it.
                    val = int(mat[pivot_row, j]) % self.q
                    inv = self._primitive_inverse(val, self.q)
                    
                    mat[pivot_row] = (mat[pivot_row] * inv) % self.q
                    
                    for k in range(rows):
                        if k != pivot_row:
                            factor = mat[k, j]
                            mat[k] = (mat[k] - factor * mat[pivot_row]) % self.q
                    pivot_row += 1
                    
        return pivot_row

    def _primitive_inverse(self, a, p):
        """Fermat's Little Theorem for modular inverse: a^(p-2) % p"""
        return pow(a, p - 2, p) if a % p != 0 else 0

# --- EXECUTION ---
if __name__ == "__main__":
    # Define your parameters here
    Q_FIELD = 3   # Try 2, 3, 5, 7...
    D_RANK = 4    # The fixed dimension
    N_GROWTH = 10 # How many columns to add
    
    model = KellyOxleyMatroid(q=Q_FIELD, d=D_RANK)
    
    print(f"Generating Matroid over GF({Q_FIELD}) with Dimension {D_RANK}")
    print("-" * 50)
    
    for i in range(1, N_GROWTH + 1):
        model.add_columns(1)
        current_r = model.get_current_rank()
        print(f"n = {model.n:<3} | Rank = {current_r}/{D_RANK}")