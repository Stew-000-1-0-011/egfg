"""egfg program for star6_k4 (cost model: 284 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[4]
    v1 = v0.sum(axis=1)
    v2 = tables[3]
    v3 = v2.sum(axis=1)
    v4 = np.einsum(v3, [0], v1, [0], [0])
    v5 = tables[1]
    v6 = v5.sum(axis=1)
    v7 = np.einsum(v6, [0], v4, [0], [0])
    v8 = tables[0]
    v9 = v8.sum(axis=1)
    v10 = tables[2]
    v11 = v10.sum(axis=1)
    v12 = np.einsum(v11, [0], v9, [0], [0])
    v13 = np.einsum(v12, [0], v7, [0], [0])
    v14 = np.einsum(v11, [0], v7, [0], [0])
    v16 = np.einsum(v8, [0, 1], v14, [0], [1])
    v17 = np.einsum(v4, [0], v12, [0], [0])
    v19 = np.einsum(v5, [0, 1], v17, [0], [1])
    v20 = np.einsum(v6, [0], v9, [0], [0])
    v21 = np.einsum(v4, [0], v20, [0], [0])
    v23 = np.einsum(v10, [0, 1], v21, [0], [1])
    v24 = np.einsum(v11, [0], v20, [0], [0])
    v25 = np.einsum(v1, [0], v24, [0], [0])
    v27 = np.einsum(v2, [0, 1], v25, [0], [1])
    v28 = np.einsum(v3, [0], v24, [0], [0])
    v30 = np.einsum(v0, [0, 1], v28, [0], [1])
    out = {}
    out['x0'] = v13 / v13.sum()
    out['x1'] = v16 / v16.sum()
    out['x2'] = v19 / v19.sum()
    out['x3'] = v23 / v23.sum()
    out['x4'] = v27 / v27.sum()
    out['x5'] = v30 / v30.sum()
    return out
