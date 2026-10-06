"""egfg program for grid2x3_k2 (cost model: 168 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[4]
    v1 = tables[2]
    v2 = np.einsum(v1, [0, 1], v0, [1, 2], [0, 1, 2])
    v3 = v2.sum(axis=1)
    v4 = tables[6]
    v5 = tables[3]
    v6 = np.einsum(v5, [0, 1], v4, [1, 2], [0, 1, 2])
    v8 = np.einsum(v6, [0, 1, 2], v3, [0, 2], [0, 1])
    v9 = tables[0]
    v11 = np.einsum(v9, [0, 1], v8, [1, 2], [0, 2])
    v12 = tables[5]
    v13 = tables[1]
    v14 = np.einsum(v13, [0, 1], v12, [1, 2], [0, 1, 2])
    v16 = np.einsum(v14, [0, 1, 2], v11, [0, 2], [0, 1])
    v17 = v16.sum(axis=1)
    v18 = v14.sum(axis=1)
    v20 = np.einsum(v9, [0, 1], v18, [0, 2], [1, 2])
    v22 = np.einsum(v6, [0, 1, 2], v20, [0, 1], [0, 2])
    v24 = np.einsum(v2, [0, 1, 2], v22, [0, 2], [0, 1])
    v25 = v24.sum(axis=1)
    v26 = v24.sum(axis=0)
    v27 = v16.sum(axis=0)
    v29 = np.einsum(v11, [0, 1], v18, [0, 1], [1])
    v31 = np.einsum(v22, [0, 1], v3, [0, 1], [1])
    out = {}
    out['x0'] = v17 / v17.sum()
    out['x1'] = v25 / v25.sum()
    out['x2'] = v26 / v26.sum()
    out['x3'] = v27 / v27.sum()
    out['x4'] = v29 / v29.sum()
    out['x5'] = v31 / v31.sum()
    return out
