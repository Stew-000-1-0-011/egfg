"""egfg program for star7_k4 (cost model: 348 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[4]
    v1 = v0.sum(axis=1)
    v2 = tables[3]
    v3 = v2.sum(axis=1)
    v4 = np.einsum(v3, [0], v1, [0], [0])
    v5 = tables[5]
    v6 = v5.sum(axis=1)
    v7 = tables[2]
    v8 = v7.sum(axis=1)
    v9 = np.einsum(v8, [0], v6, [0], [0])
    v10 = np.einsum(v9, [0], v4, [0], [0])
    v11 = tables[1]
    v12 = v11.sum(axis=1)
    v13 = tables[0]
    v14 = v13.sum(axis=1)
    v15 = np.einsum(v14, [0], v12, [0], [0])
    v16 = np.einsum(v15, [0], v10, [0], [0])
    v17 = np.einsum(v12, [0], v10, [0], [0])
    v19 = np.einsum(v13, [0, 1], v17, [0], [1])
    v20 = np.einsum(v14, [0], v10, [0], [0])
    v22 = np.einsum(v11, [0, 1], v20, [0], [1])
    v23 = np.einsum(v4, [0], v15, [0], [0])
    v24 = np.einsum(v6, [0], v23, [0], [0])
    v26 = np.einsum(v7, [0, 1], v24, [0], [1])
    v27 = np.einsum(v9, [0], v15, [0], [0])
    v28 = np.einsum(v1, [0], v27, [0], [0])
    v30 = np.einsum(v2, [0, 1], v28, [0], [1])
    v31 = np.einsum(v8, [0], v3, [0], [0])
    v32 = np.einsum(v31, [0], v15, [0], [0])
    v33 = np.einsum(v6, [0], v32, [0], [0])
    v35 = np.einsum(v0, [0, 1], v33, [0], [1])
    v36 = np.einsum(v1, [0], v32, [0], [0])
    v38 = np.einsum(v5, [0, 1], v36, [0], [1])
    out = {}
    out['x0'] = v16 / v16.sum()
    out['x1'] = v19 / v19.sum()
    out['x2'] = v22 / v22.sum()
    out['x3'] = v26 / v26.sum()
    out['x4'] = v30 / v30.sum()
    out['x5'] = v35 / v35.sum()
    out['x6'] = v38 / v38.sum()
    return out
