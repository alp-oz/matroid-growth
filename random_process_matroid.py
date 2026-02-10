"""
MODEL: Random Process Matroid (Stochastic Rank Evolution)
FILENAME: random_process_matroid.py
---------------------------------------------------------
MATHEMATICAL SUMMARY:
1. THE PROCESS: A sequence of matroids M_0, M_1, ..., M_n where 
   each state is determined by a random arrival v_t in F_2^d.
2. RANK DYNAMICS: The rank r(t) is a non-decreasing Markov chain.
3. TRANSITION PROBABILITY: The probability of a rank increase 
   at time t is P(r_t > r_{t-1}) = (2^d - 2^{r_{t-1}}) / 2^d.
4. SATURATION: The 'Hitting Time' is the value of n when r(n) = d.
---------------------------------------------------------
"""
# %%
import numpy as np

class RandomProcessMatroid:
    def __init__(self, dimension):
        # We anchor the process with a fixed ambient dimension
        self.dim = dimension 
        self.basis = []
        self.n = 0
        
    def get_independence_probability(self):
        """Calculates P that the next random vector is independent of the current basis"""
        k = len(self.basis)
        return (2**self.dim - 2**k) / (2**self.dim)

    def is_independent(self, new_v):
        """XOR Oracle to check if new_v increases the span of the current basis"""
        if not self.basis: 
            return any(x == 1 for x in new_v) # Ensure it's not the zero vector
        
        # Build matrix of current basis + the new arrival
        matrix = np.array(self.basis + [new_v])
        pivot_row = 0
        for j in range(self.dim):
            if pivot_row < len(matrix):
                for i in range(pivot_row, len(matrix)):
                    if matrix[i, j] == 1:
                        matrix[[pivot_row, i]] = matrix[[i, pivot_row]]
                        for k in range(len(matrix)):
                            if k != pivot_row and matrix[k, j] == 1:
                                matrix[k] = np.bitwise_xor(matrix[k], matrix[pivot_row])
                        pivot_row += 1
                        break
        # If the number of pivots equals the number of vectors, it's independent
        return pivot_row == len(self.basis) + 1

    def step(self):
        """A single 'tick' of the stochastic process"""
        self.n += 1
        prob = self.get_independence_probability()
        
        # Stochastic Arrival: A random vector emerges from the field
        new_v = np.random.randint(0, 2, self.dim).tolist()
        
        independent = self.is_independent(new_v)
        if independent:
            self.basis.append(new_v)
            action = "RANK UP ↑"
        else:
            action = "REDUNDANT"
            
        return self.n, action, prob, len(self.basis)

# --- EXECUTION ---
if __name__ == "__main__":
    D = 4 # Small dimension to see the probability drop quickly
    process = RandomProcessMatroid(dimension=D)
    
    print(f"{'n':<5} | {'Action':<12} | {'P(Independence)':<18} | {'Current Rank'}")
    print("-" * 55)
    
    # Run the process until the space is saturated (Full Rank)
    while len(process.basis) < D and process.n < 100:
        n_val, act, p_val, r_val = process.step()
        print(f"{n_val:<5} | {act:<12} | {p_val:<18.4f} | {r_val}/{D}")

    print(f"\nSaturation reached at n = {process.n}")