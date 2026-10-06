"""egfg program for tree9_k4 (cost model: 492 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[3]
    v1 = v0.sum(axis=1)
    v2 = tables[5]
    v3 = v2.sum(axis=1)
    v4 = tables[6]
    v5 = v4.sum(axis=1)
    v6 = tables[4]
    v8 = np.einsum(v6, [0, 1], v5, [1], [0])
    v9 = np.einsum(v8, [0], v3, [0], [0])
    v10 = tables[2]
    v12 = np.einsum(v10, [0, 1], v9, [1], [0])
    v13 = tables[1]
    v15 = np.einsum(v13, [0, 1], v12, [1], [0])
    v16 = np.einsum(v15, [0], v1, [0], [0])
    v17 = tables[0]
    v19 = np.einsum(v17, [0, 1], v16, [1], [0])
    v20 = tables[7]
    v21 = v20.sum(axis=1)
    v22 = np.einsum(v21, [0], v19, [0], [0])
    v24 = np.einsum(v17, [0, 1], v21, [0], [1])
    v25 = np.einsum(v24, [0], v1, [0], [0])
    v26 = np.einsum(v15, [0], v25, [0], [0])
    v28 = np.einsum(v13, [0, 1], v25, [0], [1])
    v29 = np.einsum(v12, [0], v28, [0], [0])
    v31 = np.einsum(v10, [0, 1], v28, [0], [1])
    v32 = np.einsum(v31, [0], v3, [0], [0])
    v33 = np.einsum(v8, [0], v32, [0], [0])
    v34 = np.einsum(v15, [0], v24, [0], [0])
    v36 = np.einsum(v0, [0, 1], v34, [0], [1])
    v38 = np.einsum(v6, [0, 1], v32, [0], [1])
    v39 = np.einsum(v5, [0], v38, [0], [0])
    v40 = np.einsum(v8, [0], v31, [0], [0])
    v42 = np.einsum(v2, [0, 1], v40, [0], [1])
    v44 = np.einsum(v4, [0, 1], v38, [0], [1])
    v46 = np.einsum(v20, [0, 1], v19, [0], [1])
    out = {}
    out['x0'] = v22 / v22.sum()
    out['x1'] = v26 / v26.sum()
    out['x2'] = v29 / v29.sum()
    out['x3'] = v33 / v33.sum()
    out['x4'] = v36 / v36.sum()
    out['x5'] = v39 / v39.sum()
    out['x6'] = v42 / v42.sum()
    out['x7'] = v44 / v44.sum()
    out['x8'] = v46 / v46.sum()
    return out
