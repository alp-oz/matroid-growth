"""
MODEL: Lawler's Exchange Logic (1975)
FILENAME: lawler_circuit_map.py
---------------------------------------------------------
GREEN DOCUMENTATION:
1. IDENTIFYING THE CIRCUIT: This code finds the unique subset of 
   the Basis (B) that spans a new element (y).
2. THE SWAP LIST: Every element in the returned circuit is a 
   mathematically valid candidate to be replaced by y.
3. ALGEBRAIC BASIS: Uses row reduction in GF(2) to solve for 
   the coefficients of dependency.
---------------------------------------------------------
"""
# %% 
import numpy as np

class CircuitMatroid:
    def __init__(self, n_elements, dimension):
        self.n = n_elements
        self.dim = dimension
        # The Ground Set: Our collection of random bitstrings
        self.ground_set = np.random.randint(0, 2, size=(n_elements, dimension))
        self.basis = []
        self.basis_indices = []

    def find_fundamental_circuit(self, new_element):
        """
        Calculates the Circuit C(B, y).
        Abstractly: Find the 'witnesses' in B that prove y is dependent.
        """
        if not self.basis:
            return []

        # We construct an augmented matrix [Basis | new_element]
        basis_matrix = np.array(self.basis).T
        target = np.array(new_element)
        r = len(self.basis)
        augmented = np.column_stack((basis_matrix, target))
        
        # GF(2) Row Reduction to solve the system
        pivot_row = 0
        pivot_cols = []
        for j in range(r):
            for i in range(pivot_row, self.dim):
                if augmented[i, j] == 1:
                    augmented[[pivot_row, i]] = augmented[[i, pivot_row]]
                    for k in range(self.dim):
                        if k != pivot_row and augmented[k, j] == 1:
                            augmented[k] = np.bitwise_xor(augmented[k], augmented[pivot_row])
                    pivot_cols.append(j)
                    pivot_row += 1
                    break
        
        # The final column reveals the circuit members
        circuit_members = []
        for i in range(len(pivot_cols)):
            if augmented[i, -1] == 1: # If the coefficient is 1, it's in the circuit
                circuit_members.append(self.basis_indices[pivot_cols[i]])
        
        return circuit_members

    def run_analysis(self):
        # 1. First, establish a Basis (The 'Greedy' anchor)
        for i in range(self.n):
            element = self.ground_set[i]
            if self.is_independent(self.basis + [element.tolist()]):
                self.basis.append(element.tolist())
                self.basis_indices.append(i)
        
        # 2. Map the Circuits for all rejected elements
        circuit_map = {}
        for i in range(self.n):
            if i not in self.basis_indices:
                circuit_map[i] = self.find_fundamental_circuit(self.ground_set[i])
        return circuit_map

    def is_independent(self, potential_basis):
        # XOR-based Gaussian Elimination (Independence Oracle)
        matrix = np.array(potential_basis)
        rows, cols = matrix.shape
        pivot_row = 0
        for j in range(cols):
            if pivot_row < rows:
                for i in range(pivot_row, rows):
                    if matrix[i, j] == 1:
                        matrix[[pivot_row, i]] = matrix[[i, pivot_row]]
                        for k in range(rows):
                            if k != pivot_row and matrix[k, j] == 1:
                                matrix[k] = np.bitwise_xor(matrix[k], matrix[pivot_row])
                        pivot_row += 1
                        break
        return pivot_row == len(potential_basis)

# --- RUN ---
matroid = CircuitMatroid(n_elements=12, dimension=6)
circuit_report = matroid.run_analysis()

print("\n" + "="*50)
print(f"CIRCUIT MAPPING REPORT (Lawler 1975 Logic)")
print("="*50)
print(f"Current Basis Indices: {matroid.basis_indices}")
print("-" * 50)
for element_idx, circuit in circuit_report.items():
    print(f"Element {element_idx:2} (Rejected) is blocked by Basis subset: {circuit}")
print("="*50)