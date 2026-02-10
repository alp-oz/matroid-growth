"""
MODEL: Matroid Secretary (Babaioff et al., 2007)
FILENAME: secretary_matroid.py
---------------------------------------------------------
MATHEMATICAL SUMMARY:
1. THE OBSERVATION (t <= m): The basis B remains empty. We evolve 
   the threshold 'tau' by recording the maximum weight seen.
2. THE SELECTION (t > m): The threshold 'tau' is fixed. We only 
   commit to vectors that are both Independent and Better than 'tau'.
3. IRREVOCABILITY: Once an element is added to B, it cannot be 
   evicted (no swaps). If rejected, it is gone forever.

This is the 'Commitment' model. It solves the problem of hiring 
without the ability to fire.
---------------------------------------------------------
"""
# %% 
import numpy as np

class SecretaryMatroid:
    def __init__(self, n_elements, dimension):
        self.n = n_elements
        self.dim = dimension
        
        # Ground Set E and Weights W
        # Random arrival order is mathematically required for this model
        self.ground_set = np.random.randint(0, 2, size=(n_elements, dimension))
        self.weights = np.random.uniform(10, 100, n_elements)
        
        # Initial State (t=0)
        self.basis_indices = []
        self.basis_vectors = []
        self.m = int(n_elements / np.e) # The 1/e stopping point (~37%)

    def is_independent(self, potential_basis):
        """Independence Oracle using GF(2) Gaussian Elimination"""
        if not potential_basis: return True
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

    def run_secretary_process(self):
        tau = -float('inf')
        
        print(f"{'t':<5} | {'Phase':<15} | {'Action':<15} | {'w_t':<8} | {'Tau (Threshold)'}")
        print("-" * 75)

        for t in range(self.n):
            y_vec = self.ground_set[t]
            y_w = self.weights[t]
            
            # PHASE 1: OBSERVATION (t <= m)
            if t < self.m:
                phase = "OBSERVE"
                action = "REJECT (AUTO)"
                if y_w > tau:
                    tau = y_w # Evolving the threshold
            
            # PHASE 2: SELECTION (t > m)
            else:
                phase = "SELECT"
                # Check for capacity, independence, and weight threshold
                if len(self.basis_indices) < self.dim:
                    if y_w > tau and self.is_independent(self.basis_vectors + [y_vec.tolist()]):
                        self.basis_indices.append(t)
                        self.basis_vectors.append(y_vec.tolist())
                        action = "HIRE / COMMIT"
                    else:
                        action = "REJECT"
                else:
                    action = "REJECT (FULL)"

            print(f"{t:<5} | {phase:<15} | {action:<15} | {y_w:<8.2f} | {tau:.2f}")

        return self.basis_indices

# --- EXECUTION ---
# N=20 elements, Dimension=5
model = SecretaryMatroid(n_elements=20, dimension=5)
final_basis = model.run_secretary_process()

print("\n" + "="*50)
print(f"BABAIOFF SECRETARY RESULT")
print("="*50)
print(f"Final Basis Indices: {final_basis}")
print(f"Final Basis Rank:    {len(final_basis)} / {model.dim}")
print(f"Total Weight:        {sum(model.weights[i] for i in final_basis):.2f}")
print("="*50)