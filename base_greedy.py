"""
MODEL: Edmonds' Greedy Algorithm (1971)
APPLICATION: Binary Linear Matroid
---------------------------------------------------------
MATHEMATICAL SUMMARY:
1. GROUND SET (E): A collection of 'n' vectors in GF(2).
2. INDEPENDENCE: Defined by linear independence over XOR.
3. RANK (r): The dimension of the span of the Ground Set.
4. GREEDY LOGIC: Sort elements by weight and append to Basis (B) 
   if and only if independence is maintained.

This is the 'Ancestor' model. It is a static, one-pass approach.
---------------------------------------------------------
"""
# %%
import numpy as np

class BinaryMatroid:
    def __init__(self, n_elements, dimension):
        """
        INITIALIZATION PROCESS:
        - n_elements (n): Size of the Ground Set.
        - dimension: The bit-length of vectors (Max theoretical rank).
        - self.ground_set: Randomly generated n x dim matrix of bits.
        """
        self.n = n_elements
        self.dim = dimension
        
        # Step 1: Generate the 'Universe' of the matroid
        # We use a random distribution of 0s and 1s.
        self.ground_set = np.random.randint(0, 2, size=(n_elements, dimension))
        
        # Step 2: Initialize empty structures for the result
        self.basis = []
        self.basis_indices = []
        self.rank = 0

    def is_independent(self, potential_basis):
        """
        ORACLE: Uses Gaussian Elimination via XOR to check 
        if the new set is linearly independent in GF(2).
        """
        if not potential_basis: return True
        
        matrix = np.array(potential_basis)
        rows, cols = matrix.shape
        pivot_row = 0
        
        for j in range(cols):
            if pivot_row < rows:
                for i in range(pivot_row, rows):
                    if matrix[i, j] == 1:
                        # Row Swap
                        matrix[[pivot_row, i]] = matrix[[i, pivot_row]]
                        # XOR-based reduction
                        for k in range(rows):
                            if k != pivot_row and matrix[k, j] == 1:
                                matrix[k] = np.bitwise_xor(matrix[k], matrix[pivot_row])
                        pivot_row += 1
                        break
        return pivot_row == len(potential_basis)

    def run_greedy(self, weights=None):
        """
        EXECUTION:
        1. Assign/Sort weights to prioritize elements.
        2. Iterate through elements.
        3. Lock independent elements into B.
        """
        if weights is None:
            indices = range(self.n)
        else:
            indices = np.argsort(weights)[::-1]

        self.basis = []
        self.basis_indices = []
        
        for i in indices:
            element = self.ground_set[i]
            if self.is_independent(self.basis + [element.tolist()]):
                self.basis.append(element.tolist())
                self.basis_indices.append(i)
        
        self.rank = len(self.basis)
        return self.basis

    def display_matroid_summary(self):
        print("\n" + "="*40)
        print(f"MATROID SUMMARY (n={self.n}, max_r={self.dim})")
        print("="*40)
        print("\nGround Set Matrix (E):")
        print(self.ground_set)
        print(f"\nBasis (B) indices: {sorted(self.basis_indices)}")
        print(f"Final Rank (r): {self.rank}")
        print("="*40)

# --- CONFIGURATION ---
N_ELEMENTS = 8 
MAX_RANK = 4    

matroid = BinaryMatroid(N_ELEMENTS, MAX_RANK)
weights = np.random.uniform(0, 100, N_ELEMENTS)
matroid.run_greedy(weights)
matroid.display_matroid_summary()
