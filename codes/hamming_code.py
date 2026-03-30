# codes/hamming_code.py

import numpy as np

def hamming_code():
    """
    Hamming [7,4,3] code.
    
    Parity check matrix H (3 x 7): columns are all non-zero
    binary vectors of length 3, ordered by column index 1..7.
    
    Properties:
        n = 7  (block length)
        k = 4  (dimension)
        d = 3  (minimum distance)
        r = 3  (rank = n - k)
        7 circuits, all of weight 3
    """
    H = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
    ], dtype=np.uint8)
    
    return H, 3  # matrix, rank


if __name__ == "__main__":
    H, r = hamming_code()
    n = H.shape[1]
    print(f"Hamming [7,4,3]")
    print(f"n={n}, r={r}, k={n-r}")
    print(f"H =\n{H}")