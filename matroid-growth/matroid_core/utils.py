import numpy as np

def convert_to_bitsets(col_to_rows):
    col_ints = []
    for r_list in col_to_rows:
        val = 0
        for r_idx in r_list: val |= (1 << r_idx)
        col_ints.append(val)
    return col_ints

def check_fano_minor(col_ints):
    col_set = set(col_ints)
    unique_cols = list(col_set)
    l = len(unique_cols)
    if l < 7: return False
    for i in range(l):
        for j in range(i + 1, l):
            b1, b2 = unique_cols[i], unique_cols[j]
            b12 = b1 ^ b2
            if b12 in col_set and b12 != 0:
                for m in range(j + 1, l):
                    b3 = unique_cols[m]
                    if b3 in {b1, b2, b12}: continue
                    needed = {b1^b3, b2^b3, b1^b2^b3}
                    if needed.issubset(col_set) and 0 not in needed:
                        return True
    return False

def check_wheel_minor_w3(col_ints):
    col_set = set(col_ints)
    unique_cols = list(col_set)
    if len(unique_cols) < 6: return False
    for i in range(len(unique_cols)):
        for j in range(i + 1, len(unique_cols)):
            for m in range(j + 1, len(unique_cols)):
                b1, b2, b3 = unique_cols[i], unique_cols[j], unique_cols[m]
                needed = {b1^b2, b2^b3, b1^b3}
                if needed.issubset(col_set) and 0 not in needed:
                    return True
    return False

def calculate_binary_rank(col_ints, num_rows):
    """Calculates Matroid Rank r(M) over GF(2) using Gaussian Elimination."""
    pivots = 0
    matrix = list(col_ints)
    for bit in range(num_rows):
        pivot_col = -1
        for i in range(pivots, len(matrix)):
            if (matrix[i] >> bit) & 1:
                pivot_col = i
                break
        if pivot_col != -1:
            matrix[pivots], matrix[pivot_col] = matrix[pivot_col], matrix[pivots]
            for i in range(len(matrix)):
                if i != pivots and (matrix[i] >> bit) & 1:
                    matrix[i] ^= matrix[pivots]
            pivots += 1
        if pivots >= len(matrix): break
    return pivots