# %%
class Matroid:
    def __init__(self, n, subsets=None,name=None):
        
        # n: 5 #the size of the universal set {0, 1, ..., n-1}
        # subsets = [{0,1}, {1,2}]
        
        self.n = n
        self.name = name
        self.universal_set = set(range(n))
        if subsets is None:
            self.subsets = []
        else:
            # Ensure each subset is a valid subset of the universal set
            self.subsets = [set(s) & self.universal_set for s in subsets]

s_8 = Matroid(5, [{0,1}, {1,9}], name="s_8")
print(s_8.subsets)
# %%
class Graph:
    def __init__(self):
        # Initialize an empty dictionary to store our adjacency list
        self.adj_matrix = {}

    def add_node(self, node):
        if node not in self.adj_matrix:
            self.adj_matrix[node] = []

    def add_edge(self, node1, node2):
        # Ensure both nodes exist
        self.add_node(node1)
        self.add_node(node2)
        # Add the connection (Undirected Graph)
        self.adj_matrix[node1].append(node2)
        self.adj_matrix[node2].append(node1)

    def display(self):
        for node, neighbors in self.adj_matrix.items():
            print(f"Node {node} is connected to: {neighbors}")

# Let's use it!
my_graph = Graph()
my_graph.add_edge("A", "B")
my_graph.add_edge("B", "C")
my_graph.display()
print("\n my_graph.adj_matrix:", my_graph.adj_matrix)
# %%
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.adj_matrix = {}

    def add_node(self, node):
        if node not in self.adj_matrix:
            self.adj_matrix[node] = []

    def add_edge(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)
        if node2 not in self.adj_matrix[node1]:
            self.adj_matrix[node1].append(node2)
        if node1 not in self.adj_matrix[node2]:
            self.adj_matrix[node2].append(node1)

    def plot(self):
        # 1. Create a NetworkX object
        G = nx.Graph()
        
        # 2. Convert our dictionary to edges
        for node, neighbors in self.adj_matrix.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        
        # 3. Draw it
        plt.figure(figsize=(5, 5))
        nx.draw(G, with_labels=True, node_color='#FF5733', font_weight='bold', node_size=700)
        plt.show()

# Testing it out
my_graph = Graph()
my_graph.add_edge("A", "B")
my_graph.add_edge("B", "C")
my_graph.add_edge("C", "A")
my_graph.add_edge("C", "D")
my_graph.add_edge("D", "E")

my_graph.plot()
# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.adj_matrix = {}

    def add_node(self, node):
        if node not in self.adj_matrix:
            self.adj_matrix[node] = []

    def add_edge(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)
        if node2 not in self.adj_matrix[node1]:
            self.adj_matrix[node1].append(node2)
        if node1 not in self.adj_matrix[node2]:
            self.adj_matrix[node2].append(node1)

    def get_all_nodes(self):
        return list(self.adj_matrix.keys())

    def add_preferential_nodes(self, n_new_nodes):
        """Adds n nodes using the Barabási-Albert logic."""
        for i in range(n_new_nodes):
            new_node_name = f"Node_{len(self.adj_matrix)}"
            existing_nodes = self.get_all_nodes()
            
            if not existing_nodes:
                self.add_node(new_node_name)
                continue

            # 1. Calculate degrees for all existing nodes
            degrees = [len(self.adj_matrix[node]) for node in existing_nodes]
            total_degree = sum(degrees)

            # 2. Calculate probabilities (degree of node / total degrees)
            if total_degree == 0:
                # If no edges exist yet, pick a random node to connect to
                probabilities = [1/len(existing_nodes)] * len(existing_nodes)
            else:
                probabilities = [d / total_degree for d in degrees]

            # 3. Choose a node to connect to based on those probabilities
            target_node = np.random.choice(existing_nodes, p=probabilities)
            
            # 4. Add the new node and the edge
            self.add_edge(new_node_name, target_node)

    def plot(self):
        G = nx.Graph()
        for node, neighbors in self.adj_matrix.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        
        plt.figure(figsize=(8, 6))
        # Hubs will naturally gravitate toward the center in a spring layout
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, node_size=50, node_color="skyblue", alpha=0.7)
        plt.title("Preferential Attachment Graph")
        plt.show()

# --- RUNNING THE MODEL ---
pf_graph = Graph()

# Start with a small triangle to seed the process
pf_graph.add_edge("Start1", "Start2")
pf_graph.add_edge("Start2", "Start3")
pf_graph.add_edge("Start3", "Start1")

# %%
# Add 100 nodes via preferential attachment
pf_graph.add_preferential_nodes(100)

pf_graph.plot()
# %%
def grow_binary_matroid(rank, n_final):
    # Start with an Identity Matrix of size (rank x rank)
    # This represents rank independent elements
    matrix = np.eye(rank, dtype=int)
    
    while matrix.shape[1] < n_final:
        # Pick 2 or 3 random columns to XOR together
        # This is like 'Preferential Attachment' for dependencies!
        num_to_combine = np.random.randint(2, 4)
        cols_to_sum = np.random.choice(matrix.shape[1], num_to_combine, replace=False)
        
        # XOR sum in GF(2) is just (sum % 2)
        new_col = np.sum(matrix[:, cols_to_sum], axis=1) % 2
        
        # Add a little 'noise' so it's not always dependent
        if np.random.rand() > 0.8:
            new_col = np.random.randint(0, 2, rank)
            
        matrix = np.column_stack([matrix, new_col])
    
    return matrix

# Grow a matroid of rank 4 up to 10 elements
binary_matrix = grow_binary_matroid(4, 10)
print("Binary Representation Matrix:")
print(binary_matrix)
# %%
import numpy as np
import matplotlib.pyplot as plt

class BinaryMatroidGrowth:
    def __init__(self):
        # We start with None to handle the very first element easily
        self.matrix = None
        self.column_origins = [] 

    def add_independent(self):
        """Increases the rank by adding a new row and a new basis column."""
        if self.matrix is None:
            # First element ever: create a 1x1 matrix [[1]]
            self.matrix = np.array([[1]], dtype=int)
        else:
            n_rows, n_cols = self.matrix.shape
            # 1. Add a row of zeros to the existing matrix to increase its height
            new_row = np.zeros((1, n_cols), dtype=int)
            self.matrix = np.vstack([self.matrix, new_row])
            
            # 2. Create a new identity column of the new height
            new_col = np.zeros((n_rows + 1, 1), dtype=int)
            new_col[-1, 0] = 1 # The pivot for the new dimension
            
            # 3. Now hstack works because both have n_rows + 1
            self.matrix = np.hstack([self.matrix, new_col])
        
        self.column_origins.append('basis')

    def add_dependent(self, k=3):
        """Adds a column that is an XOR sum of k previous columns."""
        if self.matrix is None:
            self.add_independent()
            return

        n_rows, n_cols = self.matrix.shape
        if n_cols < k:
            self.add_independent()
            return

        # Local Lookback logic
        lookback = min(n_cols, k * 2)
        choices = np.random.choice(range(n_cols - lookback, n_cols), k, replace=False)
        
        # XOR sum the chosen columns
        new_col = np.zeros(n_rows, dtype=int)
        for idx in choices:
            new_col = (new_col + self.matrix[:, idx]) % 2
        
        # Avoid the zero vector (loops)
        if np.all(new_col == 0):
            new_col[np.random.choice(n_rows)] = 1
            
        self.matrix = np.hstack([self.matrix, new_col.reshape(-1, 1)])
        self.column_origins.append('circuit')

# --- Simulation Execution ---
model = BinaryMatroidGrowth()
total_steps = 300
history = {'n': [], 'rank': []}

for i in range(1, total_steps + 1):
    # Probability of rank increase = 1/sqrt(n)
    p_independent = 1.0 / (i**0.5)
    
    if np.random.rand() < p_independent:
        model.add_independent()
    else:
        model.add_dependent(k=3)
    
    # Store history for plotting
    history['n'].append(model.matrix.shape[1])
    history['rank'].append(model.matrix.shape[0])

# --- Visualizing the Results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Plot the Sparsity Pattern (Spy)
ax1.spy(model.matrix, markersize=1, aspect='auto', color='darkblue')
ax1.set_title(f"Matroid Structure (Rank: {model.matrix.shape[0]}, Elements: {model.matrix.shape[1]})")
ax1.set_ylabel("Basis Dimensions")
ax1.set_xlabel("Elements")

# 2. Plot the Growth Curve
ax2.plot(history['n'], history['rank'], label="Matroid Rank (r)", linewidth=2)
ax2.plot(history['n'], history['n'], '--', color='gray', alpha=0.5, label="Uniform Limit (r=n)")
ax2.set_title("Rank Growth Over Time")
ax2.set_xlabel("Ground Set Size (n)")
ax2.legend()

plt.tight_layout()
plt.show()
# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PreferentialAttachmentMatroid:
    def __init__(self, initial_rank=3):
        # A represents the dependent columns. Rows represent basis elements.
        self.A = np.zeros((initial_rank, 0), dtype=int)
        self.rank = initial_rank
        self.history = {'n': [], 'rank': [], 'girth': [], 'mean_degree': []}

    def get_row_degrees(self):
        # Basis degree = 1 (from Identity) + Number of dependent elements using it
        if self.A.shape[1] == 0:
            return np.ones(self.rank, dtype=int)
        return 1 + np.sum(self.A, axis=1)

    def calculate_girth(self):
        """Finds the size of the smallest circuit (Hamming weight + 1)."""
        if self.A.shape[1] == 0:
            return float('inf')
        # In this model, the weight of a column in A + 1 is a circuit size
        col_weights = np.sum(self.A, axis=0)
        return np.min(col_weights) + 1

    def grow(self, steps=500, k_base=3):
        for i in range(1, steps + 1):
            n = self.rank + self.A.shape[1]
            p_expand = 1.0 / np.sqrt(n)
            
            if np.random.rand() < p_expand:
                # EXPANSION: Add a new Basis Element (Row)
                new_row = np.zeros((1, self.A.shape[1]), dtype=int)
                self.A = np.vstack([self.A, new_row])
                self.rank += 1
            else:
                # PREFERENTIAL ATTACHMENT: Proportional to current row degree
                degrees = self.get_row_degrees()
                # Delta=1 ensures new rows aren't ignored forever
                probs = (degrees + 1.0) / np.sum(degrees + 1.0)
                
                # Complexity k: how many basis elements we XOR
                k = min(self.rank, k_base)
                chosen_rows = np.random.choice(range(self.rank), size=k, replace=False, p=probs)
                
                new_col = np.zeros((self.rank, 1), dtype=int)
                new_col[chosen_rows, 0] = 1
                self.A = np.hstack([self.A, new_col])

            # Logging
            if i % 20 == 0:
                self.history['n'].append(n)
                self.history['rank'].append(self.rank)
                self.history['girth'].append(self.calculate_girth())
                self.history['mean_degree'].append(np.mean(self.get_row_degrees()))

    def plot_results(self):
        degrees = self.get_row_degrees()
        plt.figure(figsize=(15, 5))

        # 1. Degree Distribution (Histogram)
        plt.subplot(1, 3, 1)
        plt.hist(degrees, bins=20, color='forestgreen', edgecolor='black')
        plt.yscale('log')
        plt.title("Basis Degree Distribution (Log)")
        plt.xlabel("Degree (Row Sum)")
        plt.ylabel("Frequency")

        # 2. Girth Evolution
        plt.subplot(1, 3, 2)
        plt.plot(self.history['n'], self.history['girth'], color='red')
        plt.title("Girth (Smallest Circuit) over Time")
        plt.xlabel("Ground Set Size (n)")
        plt.ylabel("Min Circuit Size")

        # 3. Rank vs. n
        plt.subplot(1, 3, 3)
        plt.plot(self.history['n'], self.history['rank'], label='Rank')
        plt.plot(self.history['n'], np.array(self.history['n'])/2, '--', label='n/2')
        plt.title("Rank Growth Rate")
        plt.xlabel("n")
        plt.legend()

        plt.tight_layout()
        plt.show()

# Run Project
pam = PreferentialAttachmentMatroid()
pam.grow(steps=1000)
pam.plot_results()
# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def plot_power_law_analysis(model):
    degrees = model.get_row_degrees()
    
    # 1. Count frequencies of each degree
    degree_counts = Counter(degrees)
    x = list(degree_counts.keys())
    y = list(degree_counts.values())

    plt.figure(figsize=(12, 5))

    # Plot A: Standard Frequency Diagram (The "Long Tail")
    plt.subplot(1, 2, 1)
    plt.hist(degrees, bins=30, color='skyblue', edgecolor='black')
    plt.title("Degree Frequency (Linear Scale)")
    plt.xlabel("Degree (Row Sum)")
    plt.ylabel("Number of Basis Elements")

    # Plot B: Log-Log Plot (The Power Law Test)
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color='red', alpha=0.6)
    plt.xscale('log')
    plt.yscale('log')
    
    # Add a trendline to show the 'alpha' exponent
    logx, logy = np.log10(x), np.log10(y)
    if len(x) > 1:
        m, c = np.polyfit(logx, logy, 1)
        plt.plot(x, 10**(m*np.log10(x) + c), 'k--', label=f'Slope (α) ≈ {abs(m):.2f}')
    
    plt.title("Degree Frequency (Log-Log Scale)")
    plt.xlabel("Log Degree")
    plt.ylabel("Log Frequency")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

# Run this after your simulation:
# plot_power_law_analysis(pam)
# %%  Matroid attachment
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class ProjectMatroid:
    def __init__(self, initial_rank=5):
        self.rank = initial_rank
        # A matrix stores dependencies. Each row is a Basis Element.
        self.A = np.zeros((initial_rank, 0), dtype=int)
        
    def get_degrees(self):
        # Degree = 1 (basis itself) + number of times it's used in A
        if self.A.shape[1] == 0:
            return np.ones(self.rank)
        return 1 + np.sum(self.A, axis=1)

    def grow(self, steps=1000, k=3):
        for i in range(1, steps + 1):
            n = self.rank + self.A.shape[1]
            # Expansion vs Attachment coin toss
            if np.random.rand() < (1.0 / np.sqrt(n + 1)): 
                # Expansion: Add new row (Rank +1)
                new_row = np.zeros((1, self.A.shape[1]), dtype=int)
                self.A = np.vstack([self.A, new_row])
                self.rank += 1
            else:
                # Preferential Attachment
                degs = self.get_degrees()
                probs = degs / np.sum(degs)
                
                # Pick k basis elements to XOR based on their degree
                idx = np.random.choice(range(self.rank), size=min(self.rank, k), replace=False, p=probs)
                new_col = np.zeros((self.rank, 1), dtype=int)
                new_col[idx, 0] = 1
                self.A = np.hstack([self.A, new_col])

# --- Run Simulation ---
model = ProjectMatroid()
model.grow(steps=1000, k=6)
degrees = model.get_degrees()

# --- Statistical Visualization ---
plt.figure(figsize=(14, 6))

# 1. Frequency Diagram (Linear)
plt.subplot(1, 2, 1)
plt.hist(degrees, bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
plt.title("Degree Frequency (Linear Scale)\n'The Long Tail'", fontsize=14)
plt.xlabel("Degree (Number of Connections)")
plt.ylabel("Count of Basis Elements")
plt.grid(axis='y', alpha=0.3)

# 2. Power Law Plot (Log-Log)
plt.subplot(1, 2, 2)
counts = Counter(degrees)
d_vals, freq_vals = zip(*sorted(counts.items()))

plt.scatter(d_vals, freq_vals, color='#e74c3c', s=40, label='Data Points')
plt.xscale('log')
plt.yscale('log')

# Fit a line to the log-log data to find the Alpha exponent
log_d = np.log10(d_vals)
log_f = np.log10(freq_vals)
m, c = np.polyfit(log_d, log_f, 1)
plt.plot(d_vals, 10**(m*np.log10(d_vals) + c), color='black', linestyle='--', label=f'Slope (α) = {m:.2f}')

plt.title("Degree Frequency (Log-Log Scale)\n'The Power Law Test'", fontsize=14)
plt.xlabel("Log Degree")
plt.ylabel("Log Frequency")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.1)

plt.tight_layout()
plt.show()
# %% PFG
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx

def simulate_graph_ba(n_total=1000, m=3):
    # Use NetworkX to generate a Barabasi-Albert graph
    # n = total nodes, m = edges to attach from a new node to existing nodes
    G = nx.barabasi_albert_graph(n_total, m)
    degrees = [val for (node, val) in G.degree()]
    return degrees

# --- Run Graph Simulation ---
graph_degrees = simulate_graph_ba(n_total=3000, m=6)

# --- Statistical Visualization ---
plt.figure(figsize=(14, 6))

# 1. Frequency Diagram (Linear)
plt.subplot(1, 2, 1)
plt.hist(graph_degrees, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
plt.title("Graph Degree Frequency (Linear Scale)\n'The Power Grid Nodes'", fontsize=14)
plt.xlabel("Degree (Edges per Node)")
plt.ylabel("Count of Nodes")
plt.grid(axis='y', alpha=0.3)

# 2. Power Law Plot (Log-Log)
plt.subplot(1, 2, 2)
counts = Counter(graph_degrees)
d_vals, freq_vals = zip(*sorted(counts.items()))

plt.scatter(d_vals, freq_vals, color='#9b59b6', s=40, label='Graph Data')
plt.xscale('log')
plt.yscale('log')

# Fit line
log_d = np.log10(d_vals)
log_f = np.log10(freq_vals)
m_slope, c = np.polyfit(log_d, log_f, 1)
plt.plot(d_vals, 10**(m_slope*np.log10(d_vals) + c), color='black', linestyle='--', label=f'Slope (α) = {m_slope:.2f}')

plt.title("Graph Degree (Log-Log Scale)\n'Scale-Free Topology'", fontsize=14)
plt.xlabel("Log Degree")
plt.ylabel("Log Frequency")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.1)

plt.tight_layout()
plt.show()

# %% Matroid attachment wth no ind. elements
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class FixedRankMatroid:
    def __init__(self, rank=100):
        self.rank = rank
        # We start with a fixed number of rows (Basis Elements)
        # A matrix starts empty (no redundant connections yet)
        self.A = np.zeros((rank, 0), dtype=int)
        
    def get_degrees(self):
        # Degree = 1 (Identity) + Number of times used in A
        if self.A.shape[1] == 0:
            return np.ones(self.rank)
        return 1 + np.sum(self.A, axis=1)

    def grow(self, n_connections, k):
        """
        n_connections: How many redundant elements to add
        k: How many basis elements each connection XORs
        """
        for i in range(n_connections):
            degs = self.get_degrees()
            # Preferential attachment probability
            probs = degs / np.sum(degs)
            
            # Pick k basis elements to form a new circuit
            idx = np.random.choice(range(self.rank), size=min(self.rank, k), replace=False, p=probs)
            
            new_col = np.zeros((self.rank, 1), dtype=int)
            new_col[idx, 0] = 1
            
            # Always connect (Always add a column)
            self.A = np.hstack([self.A, new_col])

# --- Parameters ---
RANK = 200        # Fixed number of basis elements
N_ADDED = 6800    # Total elements added (Total n will be 4000)
K_VALUE = 4       # Connectivity

model = FixedRankMatroid(rank=RANK)
model.grow(n_connections=N_ADDED, k=K_VALUE)
degrees = model.get_degrees()

# --- Visualization ---
plt.figure(figsize=(14, 6))

# 1. Linear Frequency
plt.subplot(1, 2, 1)
plt.hist(degrees, bins=30, color='#e67e22', edgecolor='black', alpha=0.7)
plt.title(f"Fixed-Rank Matroid (Rank={RANK}, k={K_VALUE})\nAlways Connecting", fontsize=13)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.3)

# 2. Log-Log Power Law
plt.subplot(1, 2, 2)
counts = Counter(degrees)
d_vals, freq_vals = zip(*sorted(counts.items()))
plt.scatter(d_vals, freq_vals, color='#d35400', s=40)
plt.xscale('log')
plt.yscale('log')

# Fit line
log_d, log_f = np.log10(d_vals), np.log10(freq_vals)
m, c = np.polyfit(log_d, log_f, 1)
plt.plot(d_vals, 10**(m*np.log10(d_vals) + c), color='black', linestyle='--', label=f'Slope (α) = {m:.2f}')

plt.title("Log-Log Degree Distribution", fontsize=13)
plt.xlabel("Log Degree")
plt.ylabel("Log Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class MatroidPivotTest:
    def __init__(self, rank, n_added, k):
        self.rank = rank
        # Start with fixed rank and grow via preferential attachment
        self.A = np.zeros((rank, 0), dtype=int)
        self.grow(n_added, k)

    def grow(self, n, k):
        for _ in range(n):
            degs = 1 + np.sum(self.A, axis=1)
            probs = degs / np.sum(degs)
            idx = np.random.choice(range(self.rank), size=min(self.rank, k), replace=False, p=probs)
            new_col = np.zeros((self.rank, 1), dtype=int)
            new_col[idx, 0] = 1
            self.A = np.hstack([self.A, new_col])

    def get_degrees(self):
        return 1 + np.sum(self.A, axis=1)

    def pivot(self, row_idx, col_idx):
        """ Performs a pivot in GF(2). Swaps Basis Element 'row_idx' with Dependent Element 'col_idx' """
        if self.A[row_idx, col_idx] == 0:
            return False # Cannot pivot on a zero entry
        
        new_A = self.A.copy()
        pivot_row = self.A[row_idx, :].copy()
        
        for r in range(self.rank):
            if r != row_idx and self.A[r, col_idx] == 1:
                new_A[r, :] = np.bitwise_xor(self.A[r, :], pivot_row)
        
        # In a real pivot, the column becomes a unit vector. 
        # To keep our [I|A] format, we logically 'swap' them.
        self.A = new_A
        return True

    def random_pivots(self, num_pivots=100):
        successes = 0
        while successes < num_pivots:
            r = np.random.randint(0, self.rank)
            c = np.random.randint(0, self.A.shape[1])
            if self.pivot(r, c):
                successes += 1

# --- Run the Experiment ---
RANK = 300
N_ELEMENTS = 2000
K = 4

test = MatroidPivotTest(RANK, N_ELEMENTS, K)

# 1. Degrees before pivoting
degrees_before = test.get_degrees()

# 2. Perform 500 random pivots to completely scramble the basis
test.random_pivots(500)
degrees_after = test.get_degrees()

# --- Plot Comparison ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(degrees_before, bins=30, color='skyblue', alpha=0.7, label='Original Basis', log=True)
plt.title("Degree Distribution: Original Basis")
plt.xlabel("Degree")
plt.ylabel("Frequency (Log)")

plt.subplot(1, 2, 2)
plt.hist(degrees_after, bins=30, color='salmon', alpha=0.7, label='After 500 Pivots', log=True)
plt.title("Degree Distribution: After 500 Pivots")
plt.xlabel("Degree")

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

class SmoothGrowthMatroid:
    def __init__(self, initial_rank=10, k=3):
        self.rank = initial_rank
        # Matrix A (Redundant lines) starts with a few random connections
        self.A = np.random.randint(0, 2, (initial_rank, 5))
        self.k = k

    def get_degrees(self):
        # Degree = Identity (1) + sum of XOR connections in A
        return 1 + np.sum(self.A, axis=1)

    def grow(self, steps=2000, expansion_prob=0.1):
        for _ in range(steps):
            if np.random.rand() < expansion_prob:
                # --- EXPANSION (Add a new "Substation") ---
                new_row = np.zeros((1, self.A.shape[1]), dtype=int)
                self.A = np.vstack([self.A, new_row])
                self.rank += 1
                
                # IMMEDIATE ATTACHMENT: Don't let it stay at Degree 1
                # We connect the new element to 'k' existing ones immediately
                degs = self.get_degrees()[:-1] 
                probs = degs / np.sum(degs)
                targets = np.random.choice(range(self.rank-1), size=self.k-1, replace=False, p=probs)
                
                new_col = np.zeros((self.rank, 1), dtype=int)
                new_col[targets, 0] = 1
                new_col[-1, 0] = 1 # Connect to itself
                self.A = np.hstack([self.A, new_col])
            else:
                # --- ATTACHMENT (Add a "Redundant Line") ---
                degs = self.get_degrees()
                probs = degs / np.sum(degs)
                idx = np.random.choice(range(self.rank), size=self.k, replace=False, p=probs)
                
                new_col = np.zeros((self.rank, 1), dtype=int)
                new_col[idx, 0] = 1
                self.A = np.hstack([self.A, new_col])

# --- Run and Plot ---
model = SmoothGrowthMatroid(initial_rank=10, k=4)
model.grow(steps=3000, expansion_prob=0.1)
degrees = model.get_degrees()

# Sort for Zipf Plot
sorted_degs = sorted(degrees, reverse=True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.loglog(range(1, len(sorted_degs)+1), sorted_degs, 'o', markersize=3)
plt.title("Zipf Plot (Log Rank vs Log Degree)")
plt.xlabel("Rank (1st largest, 2nd, ...)")
plt.ylabel("Degree")

plt.subplot(1, 2, 2)
plt.hist(degrees, bins=40, color='coral', log=True)
plt.title("Degree Histogram (Log-Y)")
plt.xlabel("Degree")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def test_smooth_power_law(steps=3000, k=4, p_expand=0.1):
    # Start with a small identity matrix (10 rows, 0 columns in A)
    rank = 10
    A = np.zeros((rank, 0), dtype=int)
    
    for i in range(steps):
        # 1. Calculate degrees of CURRENT rows
        # Degree = 1 (Identity) + sum of connections in A
        current_degs = 1 + np.sum(A, axis=1)
        current_probs = current_degs / np.sum(current_degs)
        
        if np.random.rand() < p_expand:
            # --- EXPANSION ---
            # Step A: Add a new row (New Substation)
            new_row = np.zeros((1, A.shape[1]), dtype=int)
            A = np.vstack([A, new_row])
            rank += 1
            
            # Step B: Connect it to (k-1) EXISTING rows
            # We use current_probs because the new row doesn't have a probability yet
            targets = np.random.choice(range(rank-1), size=min(rank-1, k-1), replace=False, p=current_probs)
            
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[targets, 0] = 1
            new_col[-1, 0] = 1 # The '1' connecting to the brand new row
            A = np.hstack([A, new_col])
        else:
            # --- ATTACHMENT ---
            # Pick k existing rows to connect via a redundant line
            idx = np.random.choice(range(rank), size=min(rank, k), replace=False, p=current_probs)
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[idx, 0] = 1
            A = np.hstack([A, new_col])
            
    # Final degree calculation
    final_degs = 1 + np.sum(A, axis=1)
    return final_degs

# Run the fixed simulation
final_degs = test_smooth_power_law()

# Visualization
plt.figure(figsize=(10, 5))
plt.loglog(sorted(final_degs, reverse=True), 'o-', markersize=4, color='#2c3e50')
plt.title("Zipf's Law: Linear Matroid Growth\n(No Spikes at Ends)")
plt.xlabel("Rank of Substation (Log)")
plt.ylabel("Degree (Log)")
plt.grid(True, which="both", alpha=0.2)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def simulate_smooth_matroid(steps=4000, k=3, p_expand=0.1):
    rank = 10
    A = np.zeros((rank, 0), dtype=int)
    
    for i in range(steps):
        current_degs = 1 + np.sum(A, axis=1)
        current_probs = current_degs / np.sum(current_degs)
        
        if np.random.rand() < p_expand:
            new_row = np.zeros((1, A.shape[1]), dtype=int)
            A = np.vstack([A, new_row])
            rank += 1
            targets = np.random.choice(range(rank-1), size=min(rank-1, k-1), replace=False, p=current_probs)
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[targets, 0] = 1
            new_col[-1, 0] = 1 
            A = np.hstack([A, new_col])
        else:
            idx = np.random.choice(range(rank), size=min(rank, k), replace=False, p=current_probs)
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[idx, 0] = 1
            A = np.hstack([A, new_col])
            
    final_degs = 1 + np.sum(A, axis=1)
    return final_degs, rank, A.shape[1]

# --- Run Simulation ---
degrees, final_rank, total_cols = simulate_smooth_matroid()
total_elements = final_rank + total_cols

# --- Visualizations ---
plt.figure(figsize=(14, 6))

# 1. Linear Frequency Diagram
plt.subplot(1, 2, 1)
plt.hist(degrees, bins=30, color='#27ae60', edgecolor='black', alpha=0.7)
plt.title(f"Degree Frequency (Linear)\nRank={final_rank}, n={total_elements}", fontsize=12)
plt.xlabel("Degree (Connections per Substation)")
plt.ylabel("Count of Substations")
plt.grid(axis='y', alpha=0.3)

# 2. Zipf / Power Law Plot
plt.subplot(1, 2, 2)
sorted_degs = sorted(degrees, reverse=True)
ranks = np.arange(1, len(sorted_degs) + 1)

plt.loglog(ranks, sorted_degs, 'o', color='#8e44ad', markersize=4, label='Matroid Data')

# Fit a line to the log-log data to find the Alpha exponent
log_ranks = np.log10(ranks)
log_degs = np.log10(sorted_degs)
m, c = np.polyfit(log_ranks, log_degs, 1)
plt.plot(ranks, 10**(m*log_ranks + c), '--', color='black', label=f'Zipf Slope (α) = {abs(m):.2f}')

plt.title("Zipf's Law (Log-Log Scale)\nScale-Free Infrastructure", fontsize=12)
plt.xlabel("Rank (1=Largest Hub)")
plt.ylabel("Degree")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.1)

plt.tight_layout()
plt.show()

# --- Print Statistics ---
print(f"{'--- Matroid Statistics ---':^30}")
print(f"Final Rank (Rows):      {final_rank}")
print(f"Redundant Elements (Cols): {total_cols}")
print(f"Total Ground Set (N):   {total_elements}")
print(f"Density Ratio (N/Rank): {total_elements/final_rank:.2f}")
print(f"Max Degree (Hub):       {max(degrees)}")
print(f"Min Degree (Floor):     {min(degrees)}")

# %%
import numpy as np
import matplotlib.pyplot as plt

def generate_true_power_law_matroid(steps=4000, k=4, p_expand=0.1):
    # Start with k+1 rows so we have enough targets for the first connections
    rank = k + 1
    # A starts with one connection per initial row to avoid 'Degree 1'
    A = np.eye(rank, dtype=int) 
    
    for i in range(steps):
        # Current degrees
        current_degs = 1 + np.sum(A, axis=1)
        
        if np.random.rand() < p_expand:
            # --- BIRTH RULE ---
            # 1. Add the row
            new_row = np.zeros((1, A.shape[1]), dtype=int)
            A = np.vstack([A, new_row])
            rank += 1
            
            # 2. FORCE MINIMUM DEGREE: Add k connections IMMEDIATELY
            # This mimics a node being born with m edges
            probs = current_degs / np.sum(current_degs)
            targets = np.random.choice(range(rank-1), size=k-1, replace=False, p=probs)
            
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[targets, 0] = 1
            new_col[-1, 0] = 1 # The new row's own connection
            A = np.hstack([A, new_col])
        else:
            # --- STANDARD ATTACHMENT ---
            probs = current_degs / np.sum(current_degs)
            idx = np.random.choice(range(rank), size=k, replace=False, p=probs)
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[idx, 0] = 1
            A = np.hstack([A, new_col])
            
    return 1 + np.sum(A, axis=1), rank

# Run and Plot
degs, r = generate_true_power_law_matroid()
sorted_degs = sorted(degs, reverse=True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(degs, bins=range(k, max(degs)), color='royalblue', alpha=0.7)
plt.title("No Spike at 1: Starts at Degree K")

plt.subplot(1, 2, 2)
plt.loglog(range(1, len(sorted_degs)+1), sorted_degs, 'o')
plt.title("Clean Power Law (Zipf Plot)")
plt.show()

print(f"Min degree: {min(degs)} (Should be {k})")
# %%
import numpy as np
import matplotlib.pyplot as plt

def generate_true_smooth_matroid(steps=5000, k=4):
    # Start with a small seed rank
    rank = k + 1
    # Initialize A so everyone starts with a decent degree
    A = np.random.randint(0, 2, (rank, k))
    
    for n in range(1, steps):
        # Current degrees (Basis + Dependents)
        degs = 1 + np.sum(A, axis=1)
        probs = degs / np.sum(degs)
        
        # 1. EXPANSION (1/sqrt(n))
        if np.random.rand() < (1.0 / np.sqrt(n + 1)):
            # Add the row
            new_row = np.zeros((1, A.shape[1]), dtype=int)
            A = np.vstack([A, new_row])
            rank += 1
            
            # THE FIX: Immediately connect this row to 'k' existing hubs
            # This 'integrates' the new row into the power law immediately
            targets = np.random.choice(range(rank-1), size=k, replace=False, p=probs)
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[targets, 0] = 1
            new_col[-1, 0] = 1 # Connect to itself
            A = np.hstack([A, new_col])
        else:
            # 2. ATTACHMENT
            # Standard preferential attachment for a new column
            idx = np.random.choice(range(rank), size=k, replace=False, p=probs)
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[idx, 0] = 1
            A = np.hstack([A, new_col])
            
    return 1 + np.sum(A, axis=1)

# --- RUN AND PLOT ---
degrees = generate_true_smooth_matroid()

plt.figure(figsize=(12, 5))
# Linear Histogram - Should look like a smooth 'Slide'
plt.subplot(1, 2, 1)
plt.hist(degrees, bins=range(k, max(degrees)), color='navy', alpha=0.7)
plt.title("Degree Frequency (Linear)\nSmooth Decay starting from K")

# Log-Log Zipf Plot - Should be a straight line
plt.subplot(1, 2, 2)
sorted_degs = sorted(degrees, reverse=True)
plt.loglog(range(1, len(sorted_degs)+1), sorted_degs, 'o', markersize=3)
plt.title("Zipf Plot (Log-Log)\nClean Power Law")
plt.show()

# %% Binary Matroid Growth Simulation
import numpy as np
import matplotlib.pyplot as plt

def generate_circuit_weaving_matroid(steps=4000, max_rank=500):
    # Start with a small Identity Basis
    rank = 10
    A = np.eye(rank, dtype=int) 
    
    for n in range(1, steps):
        # 1. Choose size of the Independent Set S
        # We want to pick small sets more often (1/size)
        sizes = np.arange(1, rank + 1)
        size_probs = (1.0 / sizes) / np.sum(1.0 / sizes)
        chosen_size = np.random.choice(sizes, p=size_probs)
        
        # 2. Pick the actual elements of the set S from the Basis (Rows)
        # To keep it scale-free, we can use preferential attachment here too
        row_degs = np.sum(A, axis=1) + 1
        row_probs = row_degs / np.sum(row_degs)
        
        S_indices = np.random.choice(range(rank), size=chosen_size, replace=False, p=row_probs)
        
        # 3. Create the new element (The Circuit-maker)
        # The new column is the XOR sum of the rows in S
        new_col = np.zeros((rank, 1), dtype=int)
        new_col[S_indices, 0] = 1
        
        # 4. Expansion (1/sqrt(n))
        if np.random.rand() < (1.0 / np.sqrt(n + 1)) and rank < max_rank:
            # Add a new Row (A new Basis element)
            new_row = np.zeros((1, A.shape[1]), dtype=int)
            A = np.vstack([A, new_row])
            rank += 1
            # The new column connects this new row to the chosen set S
            new_col = np.vstack([new_col, [[1]]])
        
        A = np.hstack([A, new_col])
        
    return np.sum(A, axis=1)

# --- Visualize ---
degrees = generate_circuit_weaving_matroid()
sorted_degs = sorted(degrees, reverse=True)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(degrees, bins=50, color='#d35400', edgecolor='white', log=True)
plt.title("Matroid Circuit-Weaving\n(Log-Frequency Histogram)")
plt.xlabel("Degree (Participation in Circuits)")

plt.subplot(1, 2, 2)
plt.loglog(range(1, len(sorted_degs)+1), sorted_degs, 'o', color='#e67e22', markersize=3)
plt.title("Zipf Scaling: Smooth Transition")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def generate_uniform_circuit_matroid(steps=5000):
    rank = 10
    # A starts as a small identity seed
    A = np.eye(rank, dtype=int)
    
    for n in range(1, steps):
        # 1. UNIFORM SELECTION
        # Every row has a 50% chance of being in the set S
        # This is equivalent to picking any independent set with equal prob
        new_col = np.random.randint(0, 2, (rank, 1))
        
        # 2. EXPANSION (1/sqrt(n))
        if np.random.rand() < (1.0 / np.sqrt(n + 1)):
            rank += 1
            # Add a row to A
            A = np.vstack([A, np.zeros((1, A.shape[1]), dtype=int)])
            # Add the new row's participation to the new column
            new_col = np.vstack([new_col, [[1]]])
            
        A = np.hstack([A, new_col])
        
    # Degree is 1 (Identity) + sum of participation in A
    return 1 + np.sum(A, axis=1)

# --- Visualize ---
degrees = generate_uniform_circuit_matroid()
sorted_degs = sorted(degrees, reverse=True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(degrees, bins=30, color='teal', edgecolor='white')
plt.title("Uniform Selection Degree Distribution")

plt.subplot(1, 2, 2)
plt.loglog(range(1, len(sorted_degs)+1), sorted_degs, 'o', markersize=3)
plt.title("Zipf Plot (Uniform S)")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def generate_matroid_with_metrics(steps=5000):
    rank = 10
    # Start with an Identity Basis seed
    A = np.eye(rank, dtype=int)
    
    birth_degrees = [1] * rank  # Everyone in the seed starts at degree 1
    circuit_lengths = []        # To track the size of each column's circuit

    for n in range(1, steps):
        # 1. Calculate current degrees for preferential attachment
        current_degs = 1 + np.sum(A, axis=1)
        row_probs = current_degs / np.sum(current_degs)
        
        # 2. Expansion Logic (1/sqrt(n))
        if np.random.rand() < (1.0 / np.sqrt(n + 1)):
            rank += 1
            # Add new row to A
            A = np.vstack([A, np.zeros((1, A.shape[1]), dtype=int)])
            # Track birth degree: 1 (Identity)
            birth_degrees.append(1)
            # Re-update row_probs for the new row
            current_degs = 1 + np.sum(A, axis=1)
            row_probs = current_degs / np.sum(current_degs)

        # 3. Choose Size of Independent Set S: P(size) ~ 1/size
        # We cap this at the current rank
        sizes = np.arange(1, rank + 1)
        size_weights = 1.0 / sizes
        size_probs = size_weights / np.sum(size_weights)
        
        # This was the error: ensuring 'sizes' and 'size_probs' match current rank
        chosen_size = np.random.choice(sizes, p=size_probs)
        
        # 4. Pick the members of S using preferential attachment
        S_indices = np.random.choice(range(rank), size=chosen_size, replace=False, p=row_probs)
        
        # 5. Create the new element (XOR/sum of basis elements)
        new_col = np.zeros((rank, 1), dtype=int)
        new_col[S_indices, 0] = 1
        A = np.hstack([A, new_col])
        
        # 6. Track Circuit Length: |S| + 1 (the new element itself)
        circuit_lengths.append(chosen_size + 1)
            
    return birth_degrees, circuit_lengths, 1 + np.sum(A, axis=1)

birth_degs, circ_lengths, final_degs = generate_matroid_with_metrics()

# --- Plotting ---
plt.figure(figsize=(15, 5))

# Plot 1: Birth Degree vs Time
plt.subplot(1, 3, 1)
plt.scatter(range(len(birth_degs)), birth_degs, alpha=0.3, s=10, color='crimson')
plt.title("Degree of Element at Birth")
plt.xlabel("Row Index (Chronological)")
plt.ylabel("Initial Degree")

# Plot 2: Histogram of Circuit Lengths
plt.subplot(1, 3, 2)
plt.hist(circ_lengths, bins=range(2, 25), color='teal', edgecolor='white', density=True)
plt.title("Fundamental Circuit Lengths")
plt.xlabel("Length (Elements)")
plt.ylabel("Probability")

# Plot 3: Final Degree (The Slide)
plt.subplot(1, 3, 3)
sorted_final = sorted(final_degs, reverse=True)
plt.loglog(range(1, len(sorted_final)+1), sorted_final, 'o', markersize=2, color='navy')
plt.title("Final Degree (Zipf Plot)")
plt.xlabel("Rank")
plt.ylabel("Degree")

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def competing_growth_matroid(steps=8000):
    # Initial seed
    rank = 5
    A = np.zeros((rank, 0), dtype=int)
    circuit_lengths = []
    
    for n in range(1, steps + 1):
        # Current Row Participation (Degree)
        # Identity (1) + existing columns
        current_degs = 1 + np.sum(A, axis=1)
        row_probs = current_degs / np.sum(current_degs)
        
        # DECISION: Add a Row (Independent) or a Column (Dependent)?
        if np.random.rand() < (1.0 / np.sqrt(n)):
            # EXPANSION: Add a new basis element (Row)
            new_row = np.zeros((1, A.shape[1]), dtype=int)
            A = np.vstack([A, new_row])
            rank += 1
        else:
            # REINFORCEMENT: Add a circuit (Column)
            # Pick size |S| proportional to 1/|S|
            sizes = np.arange(1, rank + 1)
            size_probs = (1.0 / sizes) / np.sum(1.0 / sizes)
            chosen_size = np.random.choice(sizes, p=size_probs)
            
            # Pick which rows participate (Preferential Attachment)
            S_indices = np.random.choice(range(rank), size=chosen_size, replace=False, p=row_probs)
            
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[S_indices, 0] = 1
            A = np.hstack([A, new_col])
            
            # Record fundamental circuit length: |S| + 1
            circuit_lengths.append(chosen_size + 1)
            
    return circuit_lengths, 1 + np.sum(A, axis=1)

circ_lengths, final_degs = competing_growth_matroid()

# --- Plotting ---
plt.figure(figsize=(12, 5))

# Plot 1: Circuit Lengths
plt.subplot(1, 2, 1)
plt.hist(circ_lengths, bins=range(2, 30), color='#2ecc71', edgecolor='white', density=True)
plt.title("Reinforcement Phase: Circuit Lengths")
plt.xlabel("Elements in Circuit")
plt.ylabel("Probability")

# Plot 2: Zipf's Law (Final Participation)
plt.subplot(1, 2, 2)
sorted_final = sorted(final_degs, reverse=True)
plt.loglog(range(1, len(sorted_final)+1), sorted_final, 'o', markersize=2, color='#27ae60')
plt.title("Zipf Plot: Competing Growth Matroid")
plt.xlabel("Rank of Row")
plt.ylabel("Participation Count")

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def final_matroid_model(steps=10000):
    rank = 5
    A = np.zeros((rank, 0), dtype=int)
    circuit_lengths = []
    
    for n in range(1, steps + 1):
        # Current degrees for preferential attachment
        current_degs = 1 + np.sum(A, axis=1)
        row_probs = current_degs / np.sum(current_degs)
        
        # Expansion vs Reinforcement
        if np.random.rand() < (1.0 / np.sqrt(n)):
            # Expand Basis (New Row)
            A = np.vstack([A, np.zeros((1, A.shape[1]), dtype=int)])
            rank += 1
        else:
            # Reinforce (New Column/Circuit)
            sizes = np.arange(1, rank + 1)
            size_probs = (1.0 / sizes) / np.sum(1.0 / sizes)
            chosen_size = np.random.choice(sizes, p=size_probs)
            
            S_indices = np.random.choice(range(rank), size=chosen_size, replace=False, p=row_probs)
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[S_indices, 0] = 1
            A = np.hstack([A, new_col])
            circuit_lengths.append(chosen_size + 1)
            
    return circuit_lengths, 1 + np.sum(A, axis=1)

circ_lengths, degrees = final_matroid_model()

# --- Visualization ---
plt.figure(figsize=(18, 5))

# Plot 1: Degree Distribution (Histogram)
plt.subplot(1, 3, 1)
plt.hist(degrees, bins=50, color='royalblue', edgecolor='white', log=True)
plt.title("Degree Distribution (Log-Frequency)")
plt.xlabel("Degree (Participation Count)")
plt.ylabel("Frequency (Count of Rows)")

# Plot 2: Zipf's Law (Rank-Frequency)
plt.subplot(1, 3, 2)
sorted_degs = sorted(degrees, reverse=True)
plt.loglog(range(1, len(sorted_degs)+1), sorted_degs, 'o', markersize=2, color='darkblue')
plt.title("Zipf's Law (Log-Log)")
plt.xlabel("Rank")
plt.ylabel("Degree")

# Plot 3: Circuit Lengths
plt.subplot(1, 3, 3)
plt.hist(circ_lengths, bins=range(2, 30), color='skyblue', edgecolor='white', density=True)
plt.title("Circuit Lengths")
plt.xlabel("Elements")
plt.ylabel("Probability")

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def zipf_tuned_matroid(steps=10000, gamma=1.2):
    rank = 10
    A = np.zeros((rank, 0), dtype=int)
    
    for n in range(1, steps + 1):
        # 1. Row Selection with Super-Linear Attachment (gamma)
        current_degs = 1 + np.sum(A, axis=1)
        # Use power gamma to sharpen the Zipf slope
        row_weights = np.power(current_degs, gamma)
        row_probs = row_weights / np.sum(row_weights)
        
        # 2. SLOWER EXPANSION: 1/n 
        # This gives the 'old' nodes more time to become massive hubs
        if np.random.rand() < (2.0 / (rank + 10)): 
            A = np.vstack([A, np.zeros((1, A.shape[1]), dtype=int)])
            rank += 1
        else:
            # 3. SHARPER SIZE DECAY: 1/k^2
            sizes = np.arange(1, rank + 1)
            size_weights = 1.0 / (sizes**2)
            size_probs = size_weights / np.sum(size_weights)
            chosen_size = np.random.choice(sizes, p=size_probs)
            
            S_indices = np.random.choice(range(rank), size=chosen_size, replace=False, p=row_probs)
            
            new_col = np.zeros((rank, 1), dtype=int)
            new_col[S_indices, 0] = 1
            A = np.hstack([A, new_col])
            
    return 1 + np.sum(A, axis=1)

degrees = zipf_tuned_matroid()
sorted_degs = sorted(degrees, reverse=True)

# --- Visualize ---
plt.figure(figsize=(12, 5))

# Plot 1: Log-Log Zipf Plot
plt.subplot(1, 2, 1)
plt.loglog(range(1, len(sorted_degs)+1), sorted_degs, 'o', markersize=2, color='firebrick')
# Reference line for ideal Zipf (Slope -1)
plt.loglog([1, len(sorted_degs)], [sorted_degs[0], sorted_degs[0]/len(sorted_degs)], '--', color='gray', label='Ideal Zipf (Slope -1)')
plt.title("Zipf's Law: Linear Log-Log Scaling")
plt.xlabel("Rank")
plt.ylabel("Degree")
plt.legend()

# Plot 2: Linear Degree Distribution
plt.subplot(1, 2, 2)
plt.hist(degrees, bins=50, color='salmon', edgecolor='white', log=True)
plt.title("Smooth Decay (No Pikes)")
plt.xlabel("Degree")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def simple_matroid_model(steps=10000, rank=100):
    # A is our dependency matrix (the circuits we add)
    # We start with a fixed number of basis elements (rows)
    A = np.zeros((rank, 0), dtype=int)
    
    for n in range(1, steps + 1):
        # 1. Row Degrees: How many columns is each row in?
        # We add 1 to avoid zeros in the probability calculation
        row_degrees = np.sum(A, axis=1) + 1
        
        # 2. Preferential Attachment: Prob of picking row i
        row_probs = row_degrees / np.sum(row_degrees)
        
        # 3. Choose size of S (1/size^2 for sharper Zipf)
        sizes = np.arange(1, rank + 1)
        size_weights = 1.0 / (sizes**2)
        size_probs = size_weights / np.sum(size_weights)
        chosen_size = np.random.choice(sizes, p=size_probs)
        
        # 4. Pick rows to form the new circuit
        S_indices = np.random.choice(range(rank), size=chosen_size, replace=False, p=row_probs)
        
        # 5. Add the new column
        new_col = np.zeros((rank, 1), dtype=int)
        new_col[S_indices, 0] = 1
        A = np.hstack([A, new_col])
        
    # Final Degrees
    basis_degrees = np.sum(A, axis=1) + 1  # Rows (The Basis)
    circuit_degrees = np.sum(A, axis=0) + 1 # Columns (The Circuits)
    
    return basis_degrees, circuit_degrees

basis_degs, circ_degs = simple_matroid_model()

# --- Plotting ---
plt.figure(figsize=(15, 6))

# Plot 1: Stacked Histogram of Degrees
plt.subplot(1, 2, 1)
plt.hist([basis_degs, circ_degs], bins=30, stacked=True, 
         color=['#FF5733', '#33B2FF'], label=['Basis Elements', 'Circuit Elements'], log=True)
plt.title("Degree Distribution (Log Scale)")
plt.xlabel("Degree (Participation Count)")
plt.ylabel("Frequency")
plt.legend()

# Plot 2: Zipf's Law (Combined)
plt.subplot(1, 2, 2)
all_degs = np.concatenate([basis_degs, circ_degs])
sorted_all = sorted(all_degs, reverse=True)
plt.loglog(range(1, len(sorted_all)+1), sorted_all, 'o', markersize=2, color='purple')
plt.title("Zipf Plot (Log-Log)")
plt.xlabel("Rank")
plt.ylabel("Degree")

plt.tight_layout()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

def pivot_matroid_model(steps=5000, rank=100):
    # A is our dependency matrix
    A = np.zeros((rank, 0), dtype=int)
    
    # We need to track the "Global Degree" of every element added
    # element_degrees[0:rank] are the initial basis elements
    element_degrees = np.ones(rank) 
    
    # To track which elements are currently 'Rows' (Basis)
    current_basis_indices = list(range(rank))
    
    for n in range(1, steps + 1):
        # 1. Row Selection via Preferential Attachment
        # Only look at the degrees of elements currently in the Basis
        basis_probs = element_degrees[current_basis_indices] / np.sum(element_degrees[current_basis_indices])
        
        # 2. Choose size of S (1/size^2)
        sizes = np.arange(1, rank)
        size_weights = 1.0 / (sizes**2)
        size_probs = size_weights / np.sum(size_weights)
        chosen_size = np.random.choice(sizes, p=size_probs)
        
        # 3. Pick rows to form the circuit
        # These are indices relative to our 'current_basis_indices' list
        local_S_indices = np.random.choice(range(rank), size=chosen_size, replace=False, p=basis_probs)
        global_S_indices = [current_basis_indices[i] for i in local_S_indices]
        
        # 4. Update Degrees
        # The new element (index: rank + n - 1) is born
        new_element_index = rank + n - 1
        new_element_degree = chosen_size + 1
        element_degrees = np.append(element_degrees, new_element_degree)
        
        # The rows in S gain +1 degree because they participate in a new circuit
        element_degrees[global_S_indices] += 1
        
        # 5. BASIS EXCHANGE (The Pivot)
        # Pick one row from S to leave the Basis, and let the new element enter
        out_index_in_S = np.random.choice(local_S_indices)
        current_basis_indices[out_index_in_S] = new_element_index

    # Identify who ended up as a Basis element vs a Circuit element
    final_basis_degs = element_degrees[current_basis_indices]
    
    # All other elements are Circuit (Dependent) elements
    all_indices = set(range(len(element_degrees)))
    circuit_indices = list(all_indices - set(current_basis_indices))
    final_circ_degs = element_degrees[circuit_indices]
    
    return final_basis_degs, final_circ_degs

basis_degs, circ_degs = pivot_matroid_model()

# --- Visualization ---
plt.figure(figsize=(15, 6))

# Plot 1: Histogram
plt.subplot(1, 2, 1)
plt.hist([basis_degs, circ_degs], bins=50, stacked=True, 
         color=['#FF5733', '#33B2FF'], label=['Current Basis', 'Dependent Elements'], log=True)
plt.title("Degree Distribution with Basis Exchange")
plt.xlabel("Total Participation Count")
plt.ylabel("Frequency")
plt.legend()

# Plot 2: Zipf's Law
plt.subplot(1, 2, 2)
all_degs = np.concatenate([basis_degs, circ_degs])
sorted_all = sorted(all_degs, reverse=True)
plt.loglog(range(1, len(sorted_all)+1), sorted_all, 'o', markersize=2, color='purple')
plt.title("Zipf Plot (Log-Log)")
plt.xlabel("Rank")
plt.ylabel("Degree")

plt.tight_layout()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

def pivot_matroid_with_tracking(steps=5000, rank=100):
    A = np.zeros((rank, 0), dtype=int)
    element_degrees = np.ones(rank) 
    current_basis_indices = list(range(rank))
    
    # Trackers
    max_circuit_history = []
    current_max = 0
    
    for n in range(1, steps + 1):
        # 1. Preferential Attachment
        basis_probs = element_degrees[current_basis_indices] / np.sum(element_degrees[current_basis_indices])
        
        # 2. Choose size of S (1/size^2)
        sizes = np.arange(1, rank)
        size_weights = 1.0 / (sizes**2)
        size_probs = size_weights / np.sum(size_weights)
        chosen_size = np.random.choice(sizes, p=size_probs)
        
        # Track largest circuit (size of S + the new element)
        circuit_size = chosen_size + 1
        if circuit_size > current_max:
            current_max = circuit_size
        max_circuit_history.append(current_max)
        
        # 3. Pick rows and update degrees
        local_S_indices = np.random.choice(range(rank), size=chosen_size, replace=False, p=basis_probs)
        global_S_indices = [current_basis_indices[i] for i in local_S_indices]
        
        new_element_index = rank + n - 1
        element_degrees = np.append(element_degrees, circuit_size)
        element_degrees[global_S_indices] += 1
        
        # 4. Basis Exchange (Pivot)
        out_index_in_S = np.random.choice(local_S_indices)
        current_basis_indices[out_index_in_S] = new_element_index

    return element_degrees[current_basis_indices], element_degrees[list(set(range(len(element_degrees))) - set(current_basis_indices))], max_circuit_history

basis_degs, circ_degs, max_circ_hist = pivot_matroid_with_tracking()

# --- Visualization ---
plt.figure(figsize=(18, 5))

# Plot 1: Histogram (Basis vs Dependent)
plt.subplot(1, 3, 1)
plt.hist([basis_degs, circ_degs], bins=50, stacked=True, color=['#FF5733', '#33B2FF'], label=['Basis', 'Dependent'], log=True)
plt.title("Degree Distribution")
plt.legend()

# Plot 2: Zipf Plot
plt.subplot(1, 3, 2)
all_degs = np.sort(np.concatenate([basis_degs, circ_degs]))[::-1]
plt.loglog(range(1, len(all_degs)+1), all_degs, 'o', markersize=2, color='purple')
plt.title("Zipf's Law")

# Plot 3: Growth of Largest Circuit
plt.subplot(1, 3, 3)
plt.plot(max_circ_hist, color='darkgreen', linewidth=2)
plt.title("Evolution of Largest Circuit Size")
plt.xlabel("Step (n)")
plt.ylabel("Max Circuit Size")

plt.tight_layout()
plt.show()

print(f"The largest circuit found contained {max_circ_hist[-1]} elements.")

# %%
import numpy as np
import matplotlib.pyplot as plt

def dynamic_matroid_model(steps=8000):
    # Start with a small initial basis
    rank = 10
    element_degrees = np.ones(rank) 
    current_basis_indices = list(range(rank))
    
    max_circuit_history = []
    rank_history = []
    current_max = 0
    
    for n in range(1, steps + 1):
        # 1. DECISION: Expand Rank or Reinforce?
        # Probability of adding a new row (Independent Element)
        if np.random.rand() < (2.0 / (rank + 1)):
            # Add a new row: Rank increases
            new_basis_element_index = len(element_degrees)
            element_degrees = np.append(element_degrees, 1)
            current_basis_indices.append(new_basis_element_index)
            rank += 1
        else:
            # Reinforce: Add a Circuit (Dependent Element)
            basis_probs = element_degrees[current_basis_indices] / np.sum(element_degrees[current_basis_indices])
            
            # Size of S limited by the CURRENT rank
            sizes = np.arange(1, rank)
            size_weights = 1.0 / (sizes**2)
            size_probs = size_weights / np.sum(size_weights)
            chosen_size = np.random.choice(sizes, p=size_probs)
            
            circuit_size = chosen_size + 1
            if circuit_size > current_max:
                current_max = circuit_size
            
            # Pick rows and update degrees
            local_S_indices = np.random.choice(range(rank), size=chosen_size, replace=False, p=basis_probs)
            global_S_indices = [current_basis_indices[i] for i in local_S_indices]
            
            new_element_index = len(element_degrees)
            element_degrees = np.append(element_degrees, circuit_size)
            element_degrees[global_S_indices] += 1
            
            # Basis Exchange (Pivot)
            out_index_in_S = np.random.choice(local_S_indices)
            current_basis_indices[out_index_in_S] = new_element_index

        max_circuit_history.append(current_max)
        rank_history.append(rank)

    # Separate final degrees
    final_basis_degs = element_degrees[current_basis_indices]
    all_indices = set(range(len(element_degrees)))
    circuit_indices = list(all_indices - set(current_basis_indices))
    final_circ_degs = element_degrees[circuit_indices]
    
    return final_basis_degs, final_circ_degs, max_circuit_history, rank_history

basis_degs, circ_degs, max_circ_hist, rank_hist = dynamic_matroid_model()

# --- Visualization ---
plt.figure(figsize=(18, 5))

# Plot 1: Stacked Histogram
plt.subplot(1, 3, 1)
plt.hist([basis_degs, circ_degs], bins=50, stacked=True, color=['#FF5733', '#33B2FF'], label=['Basis', 'Dependent'], log=True)
plt.title("Degree Distribution (Expanding Rank)")
plt.legend()

# Plot 2: Zipf's Law
plt.subplot(1, 3, 2)
all_degs = np.sort(np.concatenate([basis_degs, circ_degs]))[::-1]
plt.loglog(range(1, len(all_degs)+1), all_degs, 'o', markersize=2, color='purple')
plt.title("Zipf's Law (Log-Log)")

# Plot 3: Max Circuit vs Rank Growth
plt.subplot(1, 3, 3)
plt.plot(max_circ_hist, color='darkgreen', label='Max Circuit Size', linewidth=2)
plt.plot(rank_hist, color='orange', label='Current Rank', linestyle='--')
plt.title("Circuit Size Ceiling vs. Rank")
plt.xlabel("Step")
plt.legend()

plt.tight_layout()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

def run_fixed_basis_simulation(steps=5000, initial_rank=20, k_param=2.0):
    """
    Simulates matroid growth WITHOUT basis exchange.
    k_param: the exponent for the circuit size distribution (Zipfian size).
    """
    # 1. Initialize Basis (Rows)
    # Since there is no exchange, these rows are the only hubs possible.
    basis_indices = list(range(initial_rank))
    basis_degrees = {i: 1 for i in basis_indices}
    
    circuit_sizes = []
    total_elements = initial_rank

    for n in range(steps):
        # 2. Determine size of the new Fundamental Circuit
        # We use the power-law rule: P(size) ~ 1 / size^k_param
        possible_sizes = np.arange(1, initial_rank + 1)
        weights = 1.0 / (possible_sizes ** k_param)
        weights /= weights.sum()
        
        size = np.random.choice(possible_sizes, p=weights)
        circuit_sizes.append(size)
        
        # 3. Preferential Attachment to the FIXED Basis
        # New elements can only 'attach' to the original rows.
        rows = list(basis_degrees.keys())
        degs = np.array([basis_degrees[r] for r in rows])
        prob = degs / degs.sum()
        
        # Pick 'size' unique basis elements to form the circuit
        selected_rows = np.random.choice(rows, size=min(size, len(rows)), replace=False, p=prob)
        
        # 4. Update Degrees
        for r in selected_rows:
            basis_degrees[r] += 1
            
        total_elements += 1

    # --- Visualizations ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Zipf Plot of Basis Degrees
    sorted_degrees = sorted(basis_degrees.values(), reverse=True)
    ax1.loglog(sorted_degrees, 'ro-', markersize=4, label='Basis Elements')
    ax1.set_title(f"Zipf Plot (Steps={steps}, k={k_param})")
    ax1.set_xlabel("Rank (Log)")
    ax1.set_ylabel("Degree (Log)")
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # Plot 2: Histogram of Fundamental Circuit Sizes
    ax2.hist(circuit_sizes, bins=range(1, initial_rank + 2), color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_title("Distribution of Fundamental Circuit Sizes")
    ax2.set_xlabel("Size (Number of Basis Elements)")
    ax2.set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

# Manual Entry: Set your steps and k-exponent here
# k=1.0 creates many large circuits; k=3.0 creates mostly tiny ones.
run_fixed_basis_simulation(steps=10000, initial_rank=50, k_param=5.0)

# %%
import numpy as np
import matplotlib.pyplot as plt
import random

def run_matroid_k_evolution(steps=5000, initial_rank=20, alpha=2.0):
    rank = initial_rank
    rank_history = []
    k_values = []
    steps_axis = list(range(steps))
    
    # Track basis degrees for preferential attachment
    basis_degrees = {i: 1 for i in range(initial_rank)}
    total_elements = initial_rank

    for n in range(steps):
        # 1. The Ceiling: Current Rank
        r_n = rank
        
        # 2. The Roll: Distribution of k ~ 1/k^alpha
        possible_ks = np.arange(1, r_n + 1)
        weights = 1.0 / (possible_ks ** alpha)
        weights /= weights.sum() # Normalize
        
        k = np.random.choice(possible_ks, p=weights)
        k_values.append(k)
        
        # 3. Preferential Attachment (The "Which Ones")
        # We pick k elements based on their current degree
        rows = list(basis_degrees.keys())
        degs = np.array([basis_degrees[r] for r in rows])
        prob = degs / degs.sum()
        
        selected = np.random.choice(rows, size=min(k, len(rows)), replace=False, p=prob)
        for r in selected:
            basis_degrees[r] += 1

        # 4. Growth: Occasionally increase Rank (to lift the k-ceiling)
        # Using a simple square-root growth for demonstration
        if random.random() < (1.0 / np.sqrt(n + 1)):
            rank += 1
            basis_degrees[total_elements] = 1
            total_elements += 1
            
        rank_history.append(rank)

    # --- Visualizations ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: The Evolution of k
    ax1.scatter(steps_axis, k_values, alpha=0.3, s=2, color='purple', label='Selected k (Circuit Size)')
    ax1.plot(steps_axis, rank_history, color='red', linewidth=2, label='Rank Ceiling (Max possible k)')
    ax1.set_title("Evolution of $k$ over Time")
    ax1.set_xlabel("Steps (n)")
    ax1.set_ylabel("Value of $k$")
    ax1.legend()

    # Plot 2: Zipf Plot (The Resulting Degrees)
    sorted_degs = sorted(basis_degrees.values(), reverse=True)
    ax2.loglog(sorted_degs, 'b-')
    ax2.set_title("Resulting Zipf Distribution of Degrees")
    ax2.set_xlabel("Rank (Log)")
    ax2.set_ylabel("Degree (Log)")
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

run_matroid_k_evolution(steps=10000, alpha=2.0)

# %%
import numpy as np
import matplotlib.pyplot as plt
import random

def run_late_exchange_model(steps=2000, initial_rank=50, fixed_k=3):
    # 1. Initialize a Binary Matrix (Representation of the Matroid)
    # Part 1: Identity Matrix (The Basis)
    # Part 2: The A Matrix (The Fundamental Circuits)
    # We use a dictionary of sets to represent the A matrix rows for speed
    # rows[i] contains the indices of the columns (dependent elements) that depend on basis element i
    
    rank = initial_rank
    matrix_A = np.zeros((rank, steps), dtype=int)
    basis_degrees = np.ones(rank)
    
    # 2. Growth Phase: NO BASIS EXCHANGE
    for n in range(steps):
        # Preferential Attachment to pick which rows this column connects to
        prob = basis_degrees / basis_degrees.sum()
        selected_rows = np.random.choice(range(rank), size=fixed_k, replace=False, p=prob)
        
        for r in selected_rows:
            matrix_A[r, n] = 1
            basis_degrees[r] += 1
            
    # Calculate sizes before exchange
    sizes_before = np.sum(matrix_A, axis=0) + 1 # +1 for the element itself
    max_before = np.max(sizes_before)

    # 3. THE REVELATION: Perform ONE Basis Exchange (Pivot)
    # We pick the most "popular" dependent element (column with many connections)
    # and swap it with one of its basis elements.
    pivot_col = np.argmax(np.sum(matrix_A, axis=0))
    pivot_row = np.where(matrix_A[:, pivot_col] == 1)[0][0]
    
    # Standard Binary Row Operations (Pivoting)
    # To swap Col[pivot_col] into the Basis at Row[pivot_row]:
    # XOR the pivot row into all other rows that have a 1 in that column.
    new_matrix_A = matrix_A.copy()
    pivot_row_content = new_matrix_A[pivot_row, :].copy()
    
    for r in range(rank):
        if r != pivot_row and new_matrix_A[r, pivot_col] == 1:
            new_matrix_A[r, :] = np.bitwise_xor(new_matrix_A[r, :], pivot_row_content)
    
    # Calculate sizes after exchange
    sizes_after = np.sum(new_matrix_A, axis=0) + 1
    max_after = np.max(sizes_after)

    # --- Visualizing the "Complexity Explosion" ---
    plt.figure(figsize=(12, 6))
    plt.hist(sizes_before, bins=20, alpha=0.5, label=f'Before Pivot (Max: {max_before})', color='blue')
    plt.hist(sizes_after, bins=20, alpha=0.5, label=f'After Pivot (Max: {max_after})', color='red')
    plt.title(f"Fundamental Circuit Size Distribution (Fixed Input k={fixed_k})")
    plt.xlabel("Circuit Size (Number of Elements)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    print(f"Largest Fundamental Circuit before pivot: {max_before}")
    print(f"Largest Fundamental Circuit after pivot: {max_after}")

run_late_exchange_model(steps=5000, fixed_k=30)

# %%
import numpy as np
import matplotlib.pyplot as plt
import random

def run_late_exchange_analysis(steps=3000, initial_rank=50, fixed_k=3):
    rank = initial_rank
    # Matrix A: Rows are Basis, Columns are Dependent Elements
    matrix_A = np.zeros((rank, steps), dtype=int)
    basis_indices = list(range(rank))
    basis_degrees = np.ones(rank)
    
    # 1. GROWTH PHASE (Fixed Basis)
    for n in range(steps):
        prob = basis_degrees / basis_degrees.sum()
        selected_rows = np.random.choice(range(rank), size=fixed_k, replace=False, p=prob)
        for r in selected_rows:
            matrix_A[r, n] = 1
            basis_degrees[r] += 1
            
    # 2. THE PIVOT (At the very end)
    # We pick the most connected dependent element to promote
    pivot_col = np.argmax(np.sum(matrix_A, axis=0))
    # We pick its strongest basis connection to kick out
    rows_in_circuit = np.where(matrix_A[:, pivot_col] == 1)[0]
    pivot_row = rows_in_circuit[np.argmax(basis_degrees[rows_in_circuit])]
    
    # Perform Binary Pivot (XOR rows)
    new_matrix_A = matrix_A.copy()
    pivot_row_content = new_matrix_A[pivot_row, :].copy()
    for r in range(rank):
        if r != pivot_row and new_matrix_A[r, pivot_col] == 1:
            new_matrix_A[r, :] = np.bitwise_xor(new_matrix_A[r, :], pivot_row_content)

    # 3. CALCULATE NEW DEGREES
    # Degree of a basis element = number of 1s in its row
    # Degree of a dependent element = number of 1s in its column
    new_basis_degrees = np.sum(new_matrix_A, axis=1) + 1
    new_dep_degrees = np.sum(new_matrix_A, axis=0) + 1

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Circuit Size Distribution (Column Weights)
    sizes_before = np.sum(matrix_A, axis=0) + 1
    sizes_after = new_dep_degrees
    ax1.hist(sizes_before, bins=30, alpha=0.5, label='Pre-Pivot (All size k+1)', color='gray')
    ax1.hist(sizes_after, bins=30, alpha=0.7, label='Post-Pivot (Emergent)', color='orange')
    ax1.set_title("Circuit Size Explosion")
    ax1.set_xlabel("Size")
    ax1.legend()

    # Plot 2: Element Degree Distribution (Log-Log)
    # We mark the old basis elements vs the newcomers
    all_degrees = np.concatenate([new_basis_degrees, new_dep_degrees])
    is_basis = np.array([True]*rank + [False]*steps)
    
    # Sort for Zipf plot
    sort_idx = np.argsort(all_degrees)[::-1]
    sorted_degrees = all_degrees[sort_idx]
    sorted_is_basis = is_basis[sort_idx]
    
    ranks = np.arange(1, len(all_degrees) + 1)
    
    ax2.loglog(ranks[sorted_is_basis], sorted_degrees[sorted_is_basis], 'bo', markersize=4, label='Original Basis Elements', alpha=0.6)
    ax2.loglog(ranks[~sorted_is_basis], sorted_degrees[~sorted_is_basis], 'r.', markersize=2, label='Dependent Elements', alpha=0.4)
    
    ax2.set_title("Degree Distribution after Single Pivot")
    ax2.set_xlabel("Rank (Log)")
    ax2.set_ylabel("Degree (Log)")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

run_late_exchange_analysis()