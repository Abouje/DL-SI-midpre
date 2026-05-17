# lap shim using scipy
__version__ = "0.5.12"
import numpy as np
from scipy.optimize import linear_sum_assignment

def lapjv(cost, extend_cost=False, cost_limit=float("inf"), return_cost=True):
    cost = np.array(cost, dtype=float)
    rows, cols = cost.shape
    C = cost.copy()
    if cost_limit < float("inf"):
        large = C.max() * max(rows, cols) * 2 + 1 if C.size else 1
        C[C > cost_limit] = large
    if rows != cols:
        n = max(rows, cols)
        large = C.max() * n * 2 + 1 if C.size else 1
        pad = np.full((n, n), large)
        pad[:rows, :cols] = C
        C = pad
    row_ind, col_ind = linear_sum_assignment(C)
    x = np.full(rows, -1, dtype=int)
    y = np.full(cols, -1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        if r < rows and c < cols:
            x[r] = c
            y[c] = r
    total = float(sum(cost[r, c] for r, c in zip(row_ind, col_ind) if r < rows and c < cols))
    if return_cost:
        return total, x, y
    return x, y
