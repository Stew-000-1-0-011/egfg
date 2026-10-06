"""egfg program for ternary6_k3 (cost model: 450 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[3]
    v1 = v0.sum(axis=2)
    v2 = tables[2]
    v4 = np.einsum(v2, [0, 1, 2], v1, [1, 2], [0, 1])
    v5 = tables[1]
    v7 = np.einsum(v5, [0, 1, 2], v4, [1, 2], [0, 1])
    v8 = tables[0]
    v10 = np.einsum(v8, [0, 1, 2], v7, [1, 2], [0, 1])
    v11 = v10.sum(axis=1)
    v12 = v10.sum(axis=0)
    v13 = v8.sum(axis=0)
    v15 = np.einsum(v5, [0, 1, 2], v13, [0, 1], [1, 2])
    v16 = np.einsum(v15, [0, 1], v4, [0, 1], [0, 1])
    v17 = v16.sum(axis=1)
    v18 = v16.sum(axis=0)
    v20 = np.einsum(v2, [0, 1, 2], v15, [0, 1], [1, 2])
    v22 = np.einsum(v20, [0, 1], v1, [0, 1], [1])
    v25 = np.einsum(v0, [0, 1, 2], v20, [0, 1], [2])
    out = {}
    out['x0'] = v11 / v11.sum()
    out['x1'] = v12 / v12.sum()
    out['x2'] = v17 / v17.sum()
    out['x3'] = v18 / v18.sum()
    out['x4'] = v22 / v22.sum()
    out['x5'] = v25 / v25.sum()
    return out
