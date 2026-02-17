# %%
import numpy as np
# r = rank, n = total elements
def direct_random_binary_matroid(r, n):
    A = np.random.randint(0, 2, size=(r, n-r))
    return np.hstack([np.eye(r, dtype=int), A])

direct_random_binary_matroid(5, 10)