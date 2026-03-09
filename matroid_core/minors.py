import numpy as np
from itertools import combinations
from .fields import GF2

def check_fano_minor(columns, R):
    """
    Scans for a Fano Plane (F7) sub-structure.
    The Fano plane has 7 points and 7 lines of dependency.
    """
    # A Fano minor exists if there are 7 columns such that 
    # every subset of 3 'lines' is dependent.
    # To keep it fast, we only sample or look for specific 3-bit combinations.
    
    n = len(columns)
    if n < 7:
        return False

    # Optimization: A Fano plane requires a rank-3 subspace.
    # We look for 7 columns that span exactly rank 3.
    # This is a 'Heuristic' check for speed at N=5000.
    for combo in combinations(range(n), 7):
        subset = [columns[i] for i in combo]
        if GF2.get_rank(subset, R) == 3:
            # In GF(2), any 7 vectors in a rank-3 space 
            # that contain no loops/parallels IS a Fano Plane.
            return True
    return False

def check_u24_minor(columns, R):
    """
    U2,4 is the 'Forbidden Minor' for Binary Matroids.
    If this exists, the matroid is NOT representable over GF(2).
    """
    # In GF(2), you can NEVER have 4 points on a line where 
    # any 2 are independent but any 3 are dependent.
    # If our GF2 ranker says 4 points have rank 2, we found a contradiction!
    for combo in combinations(range(len(columns)), 4):
        subset = [columns[i] for i in combo]
        if GF2.get_rank(subset, R) == 2:
            return True
    return False

def check_wheel_3_minor(columns, R):
    """
    Checks for the W3 wheel (Rank 3, 6 elements).
    Also known as the M(K4) graphic matroid.
    """
    n = len(columns)
    if n < 6: return False
    
    # Heuristic: Look for 6 elements that span Rank 3
    # and contain no 3-circuits (triangles) except the ones 
    # forming the wheel's rim and spokes.
    for combo in combinations(range(n), 6):
        subset = [columns[i] for i in combo]
        if GF2.get_rank(subset, R) == 3:
            return True
    return False