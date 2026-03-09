# analysis/quantum.py
from matroid_core.fields import GF2

def evaluate_css_code(matroid_data):
    """
    Treats the matroid as a parity-check matrix for a Quantum CSS Code.
    Calculates the 'Logical Qubits' (k) and 'Code Rate' (R).
    """
    cols = matroid_data['columns']
    R_rows = matroid_data['R_final']
    N_cols = len(cols)
    
    # The number of logical qubits k = N - 2*Rank(H) 
    # (Simplified for a symmetric CSS code)
    rank = GF2.get_rank(cols, R_rows)
    k_logical = max(0, N_cols - 2 * rank)
    
    return {
        "code_rate": k_logical / N_cols if N_cols > 0 else 0,
        "n_physical": N_cols,
        "k_logical": k_logical,
        "is_well_defined": rank < N_cols
    }