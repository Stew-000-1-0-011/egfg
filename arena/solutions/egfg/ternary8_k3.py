"""egfg program for ternary8_k3 (cost model: 693 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[5]
    v1 = v0.sum(axis=2)
    v2 = tables[4]
    v4 = np.einsum(v2, [0, 1, 2], v1, [1, 2], [0, 1])
    v5 = tables[3]
    v7 = np.einsum(v5, [0, 1, 2], v4, [1, 2], [0, 1])
    v8 = tables[2]
    v10 = np.einsum(v8, [0, 1, 2], v7, [1, 2], [0, 1])
    v11 = tables[1]
    v13 = np.einsum(v11, [0, 1, 2], v10, [1, 2], [0, 1])
    v14 = tables[0]
    v17 = np.einsum(v14, [0, 1, 2], v13, [1, 2], [0])
    v18 = v14.sum(axis=0)
    v19 = np.einsum(v18, [0, 1], v13, [0, 1], [0, 1])
    v20 = v19.sum(axis=1)
    v21 = v19.sum(axis=0)
    v23 = np.einsum(v11, [0, 1, 2], v18, [0, 1], [1, 2])
    v25 = np.einsum(v8, [0, 1, 2], v23, [0, 1], [1, 2])
    v26 = np.einsum(v25, [0, 1], v7, [0, 1], [0, 1])
    v27 = v26.sum(axis=1)
    v28 = v26.sum(axis=0)
    v30 = np.einsum(v5, [0, 1, 2], v25, [0, 1], [1, 2])
    v32 = np.einsum(v2, [0, 1, 2], v30, [0, 1], [1, 2])
    v33 = np.einsum(v32, [0, 1], v1, [0, 1], [0, 1])
    v34 = v33.sum(axis=1)
    v35 = v33.sum(axis=0)
    v38 = np.einsum(v0, [0, 1, 2], v32, [0, 1], [2])
    out = {}
    out['x0'] = v17 / v17.sum()
    out['x1'] = v20 / v20.sum()
    out['x2'] = v21 / v21.sum()
    out['x3'] = v27 / v27.sum()
    out['x4'] = v28 / v28.sum()
    out['x5'] = v34 / v34.sum()
    out['x6'] = v35 / v35.sum()
    out['x7'] = v38 / v38.sum()
    return out
