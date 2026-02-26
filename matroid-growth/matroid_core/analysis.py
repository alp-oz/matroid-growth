import numpy as np  # <--- THIS WAS MISSING
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

def plot_zipf(simulator, title="Degree Distribution"):
    degrees = sorted(simulator.row_degrees[:simulator.curr_r], reverse=True)
    ranks = range(1, len(degrees) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.loglog(ranks, degrees, marker='o', linestyle='none', alpha=0.6)
    plt.xlabel('Rank')
    plt.ylabel('Degree (Row Frequency)')
    plt.title(f"{title} (Beta={simulator.beta})")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()



def get_connectivity_stats(simulator):
    if not simulator.col_to_rows:
        return 0, 0.0

    rows_indices = []
    cols_indices = []
    for c_idx, r_list in enumerate(simulator.col_to_rows):
        for r_idx in r_list:
            rows_indices.append(r_idx)
            cols_indices.append(c_idx + simulator.curr_r)
            
    n_nodes = simulator.curr_r + len(simulator.col_to_rows)
    adj = csr_matrix(([1]*len(rows_indices), (rows_indices, cols_indices)), 
                     shape=(n_nodes, n_nodes))
    adj = adj + adj.T 
    
    n_components, labels = connected_components(adj, directed=False)
    
    # Use np.bincount to find the size of each component
    component_sizes = np.bincount(labels)
    giant_size = np.max(component_sizes) / n_nodes
    
    return n_components, giant_size