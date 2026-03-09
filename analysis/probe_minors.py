import numpy as np

def convert_to_bitsets(col_to_rows):
    """
    Converts a list of row indices into bitset integers.
    Each bit in the integer represents a row index.
    """
    bitsets = []
    for rows in col_to_rows:
        b = 0
        for r in rows:
            b |= (1 << r)
        bitsets.append(b)
    return bitsets

def calculate_binary_rank(bitsets):
    """
    Finds the rank of the matroid using Gaussian elimination in GF(2).
    """
    basis = []
    for b in bitsets:
        for f in basis:
            b = min(b, b ^ f)
        if b > 0:
            basis.append(b)
            basis.sort(reverse=True)
    return len(basis)

def check_fano_minor(bitsets):
    """
    Detects if the Fano Plane (F7) exists as a minor.
    In GF(2), this is simply finding any 'line' (3 points that XOR to 0).
    """
    if len(bitsets) < 7:
        return False
    
    # Using a set for O(1) lookups of bitsets
    lookup = set(bitsets)
    if 0 in lookup:
        lookup.remove(0)
        
    unique_bits = list(lookup)
    n = len(unique_bits)
    
    for i in range(n):
        for j in range(i + 1, n):
            third = unique_bits[i] ^ unique_bits[j]
            if third in lookup and third != unique_bits[i] and third != unique_bits[j]:
                return True 
    return False

def check_wheel_3_minor(bitsets):
    """
    Detects the Wheel-3 minor (M(K4)).
    Requires 3 independent vectors v1, v2, v3 and their pairwise sums.
    """
    if len(bitsets) < 6:
        return False
        
    lookup = set(bitsets)
    unique_bits = list(lookup)
    n = len(unique_bits)
    
    for i in range(n):
        for j in range(i + 1, n):
            sum_ij = unique_bits[i] ^ unique_bits[j]
            if sum_ij in lookup:
                # Potential triangle found, look for 3rd independent basis vector
                for k in range(j + 1, n):
                    v1, v2, v3 = unique_bits[i], unique_bits[j], unique_bits[k]
                    # Check if v3 is independent of v1 and v2
                    if calculate_binary_rank([v1, v2, v3]) == 3:
                        # Check if v1+v3 and v2+v3 also exist in the matroid
                        if (v1 ^ v3) in lookup and (v2 ^ v3) in lookup:
                            return True
    return False