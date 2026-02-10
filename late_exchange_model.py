"""
MODEL: Late Exchange Algorithm (Dynamic Basis) - Debugged
FILENAME: late_exchange_model.py
---------------------------------------------------------
GREEN DOCUMENTATION:
1. INITIAL STATE: B_0 = {}, W_0 = 0.
2. DYNAMICS: Basis evolves by comparing arriving weight (w_t) 
   against the circuit minimum (w_min).
3. DIMENSION: Acts as the fixed capacity 'd' of the system.---------------------------------------------------------
"""
# %%
import numpy as np

class LateExchangeMatroid:
    def __init__(self, n_elements, dimension):
        self.n = n_elements
        self.dim = dimension
        
        # Ground Set and Weights
        self.ground_set = np.random.randint(0, 2, size=(n_elements, dimension))
        self.weights = np.random.uniform(10, 100, n_elements)
        
        self.basis_indices = []
        self.basis_vectors = []

    def find_circuit_indices(self, new_vector):
        if not self.basis_vectors: return []
        
        basis_matrix = np.array(self.basis_vectors).T
        r = len(self.basis_vectors)
        augmented = np.column_stack((basis_matrix, new_vector))
        
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
        
        return [pivot_cols[i] for i in range(len(pivot_cols)) if augmented[i, -1] == 1]

    def is_independent(self, potential_basis):
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

    def run_stream(self):
        print(f"{'Step':<5} | {'Action':<18} | {'Weight':<8} | {'Total Basis Weight'}")
        print("-" * 65)
        
        for i in range(self.n):
            y_vec = self.ground_set[i]
            y_weight = self.weights[i]
            
            if self.is_independent(self.basis_vectors + [y_vec.tolist()]):
                self.basis_indices.append(i)
                self.basis_vectors.append(y_vec.tolist())
                action = f"ADD (ID:{i})"
            else:
                rel_circuit = self.find_circuit_indices(y_vec)
                
                # SAFETY CHECK: Only proceed if circuit is not empty
                if rel_circuit:
                    global_circuit = [self.basis_indices[idx] for idx in rel_circuit]
                    circuit_weights = [self.weights[idx] for idx in global_circuit]
                    
                    min_idx_in_circuit = global_circuit[np.argmin(circuit_weights)]
                    min_weight = self.weights[min_idx_in_circuit]
                    
                    if y_weight > min_weight:
                        pos = self.basis_indices.index(min_idx_in_circuit)
                        self.basis_indices.pop(pos)
                        self.basis_vectors.pop(pos)
                        self.basis_indices.append(i)
                        self.basis_vectors.append(y_vec.tolist())
                        action = f"SWAP ({min_idx_in_circuit} -> {i})"
                    else:
                        action = f"REJECT (ID:{i})"
                else:
                    # This happens if y_vec is a Zero Vector [0,0,0...]
                    action = f"ZERO-VEC REJECT"

            current_total = sum(self.weights[idx] for idx in self.basis_indices)
            print(f"{i:<5} | {action:<18} | {y_weight:<8.2f} | {current_total:.2f}")

# --- RUN ---
model = LateExchangeMatroid(n_elements=20, dimension=5)
model.run_stream()