import numpy as np

class GF2:
    """Fast Binary Field Arithmetic using Bitwise XOR"""
    
    @staticmethod
    def get_rank(columns, R):
        """
        Calculates rank of a set of columns in GF(2).
        Each column is a tuple of row indices.
        """
        if not columns:
            return 0
            
        # 1. Convert row-index tuples into bit-integers
        # Each integer represents a column vector
        vectors = []
        for col in columns:
            v = 0
            for row_idx in col:
                v |= (1 << row_idx)
            vectors.append(v)
            
        # 2. Gaussian Elimination via Bitwise XOR
        basis = []
        for v in vectors:
            for b in basis:
                # Use XOR to eliminate bits
                v = min(v, v ^ b)
            if v > 0:
                basis.append(v)
                basis.sort(reverse=True)
                
        return len(basis)

class GF3:
    """Ternary Field Arithmetic: 0, 1, 2"""
    # GF(3) cannot use bitwise XOR; it uses standard modular arithmetic.
    @staticmethod
    def add(a, b):
        return (a + b) % 3
    
    @staticmethod
    def get_rank(matrix_array):
        # Requires standard Gaussian elimination over GF(3)
        # matrix_array is a numpy array of shape (R, N)
        pass