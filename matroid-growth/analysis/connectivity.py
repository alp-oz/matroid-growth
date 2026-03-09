import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def get_bipartite_connectivity(matroid_data):
    """
    Analyzes the bipartite topology of the Matroid.
    Distinguishes between orphaned basis elements and the active manifold.
    """
    cols = matroid_data['columns']
    R = matroid_data['R_final']
    N = len(cols)
    
    # 1. Map edges
    rows_indices = []
    cols_indices = []
    active_rows = set()
    
    for c_idx, r_list in enumerate(cols):
        for r_idx in r_list:
            rows_indices.append(r_idx)
            cols_indices.append(c_idx + R)
            active_rows.add(r_idx)
            
    n_nodes = R + N
    
    # 2. Build Adjacency
    adj = csr_matrix(([1]*len(rows_indices), (rows_indices, cols_indices)), 
                     shape=(n_nodes, n_nodes))
    adj = adj + adj.T 
    
    # 3. Component Analysis
    n_components, labels = connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels)
    giant_size = np.max(comp_sizes) if len(comp_sizes) > 0 else 0
    
    # 4. Filter "Stupid" Stats vs "Research" Stats
    # Orphan rows are components of size 1 that are in the [0, R) range
    total_orphans = R - len(active_rows)
    
    # Active nodes = only the rows and columns that actually have an edge
    n_active_nodes = len(active_rows) + N
    active_giant_fraction = giant_size / n_active_nodes if n_active_nodes > 0 else 0

    return {
        "total_components": n_components,
        "orphan_rows": total_orphans,
        "active_nodes": n_active_nodes,
        "giant_component_size": giant_size,
        "active_giant_fraction": active_giant_fraction, # This is the "Real" connectivity
        "is_fully_connected": n_components == 1
    }