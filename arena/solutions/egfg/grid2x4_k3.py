"""egfg program for grid2x4_k3 (cost model: 801 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[9]
    v1 = tables[6]
    v2 = np.einsum(v1, [0, 2], v0, [1, 2], [0, 1, 2])
    v3 = v2.sum(axis=2)
    v4 = tables[5]
    v5 = tables[4]
    v6 = np.einsum(v5, [0, 1], v4, [0, 2], [0, 1, 2])
    v8 = np.einsum(v6, [0, 1, 2], v3, [1, 2], [0, 2])
    v9 = tables[8]
    v11 = np.einsum(v9, [1, 2], v8, [0, 2], [0, 1])
    v12 = tables[2]
    v14 = np.einsum(v12, [0, 1], v11, [1, 2], [0, 2])
    v15 = tables[3]
    v16 = tables[0]
    v17 = np.einsum(v16, [0, 1], v15, [1, 2], [0, 1, 2])
    v19 = np.einsum(v17, [0, 1, 2], v14, [1, 2], [0, 2])
    v20 = tables[7]
    v21 = tables[1]
    v22 = np.einsum(v21, [0, 1], v20, [1, 2], [0, 1, 2])
    v24 = np.einsum(v22, [0, 1, 2], v19, [0, 2], [0, 1])
    v25 = v24.sum(axis=1)
    v26 = v22.sum(axis=1)
    v28 = np.einsum(v17, [0, 1, 2], v26, [0, 2], [1, 2])
    v29 = np.einsum(v14, [0, 1], v28, [0, 1], [0, 1])
    v30 = v29.sum(axis=1)
    v32 = np.einsum(v12, [0, 1], v28, [0, 2], [1, 2])
    v34 = np.einsum(v11, [0, 1], v32, [0, 1], [0])
    v36 = np.einsum(v9, [1, 2], v32, [0, 1], [0, 2])
    v38 = np.einsum(v6, [0, 1, 2], v36, [0, 2], [1, 2])
    v39 = np.einsum(v38, [0, 1], v3, [0, 1], [0, 1])
    v40 = v39.sum(axis=1)
    v41 = v24.sum(axis=0)
    v42 = v29.sum(axis=0)
    v43 = v39.sum(axis=0)
    v46 = np.einsum(v2, [0, 1, 2], v38, [0, 1], [2])
    out = {}
    out['x0'] = v25 / v25.sum()
    out['x1'] = v30 / v30.sum()
    out['x2'] = v34 / v34.sum()
    out['x3'] = v40 / v40.sum()
    out['x4'] = v41 / v41.sum()
    out['x5'] = v42 / v42.sum()
    out['x6'] = v43 / v43.sum()
    out['x7'] = v46 / v46.sum()
    return out
