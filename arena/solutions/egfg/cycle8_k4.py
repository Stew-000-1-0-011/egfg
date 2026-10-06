"""egfg program for cycle8_k4 (cost model: 1936 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[2]
    v1 = tables[1]
    v2 = np.einsum(v1, [0, 1], v0, [1, 2], [0, 1, 2])
    v3 = v2.sum(axis=1)
    v4 = tables[4]
    v5 = tables[3]
    v6 = np.einsum(v5, [0, 1], v4, [1, 2], [0, 1, 2])
    v7 = v6.sum(axis=1)
    v9 = np.einsum(v7, [1, 2], v3, [0, 1], [0, 2])
    v10 = tables[6]
    v11 = tables[5]
    v12 = np.einsum(v11, [0, 1], v10, [1, 2], [0, 1, 2])
    v13 = v12.sum(axis=1)
    v15 = np.einsum(v13, [1, 2], v9, [0, 1], [0, 2])
    v16 = np.transpose(tables[7], (1, 0))
    v17 = tables[0]
    v18 = np.einsum(v17, [0, 1], v16, [0, 2], [0, 1, 2])
    v20 = np.einsum(v18, [0, 1, 2], v15, [1, 2], [0, 1])
    v21 = v20.sum(axis=1)
    v22 = v20.sum(axis=0)
    v23 = v18.sum(axis=0)
    v25 = np.einsum(v13, [1, 2], v23, [0, 2], [0, 1])
    v27 = np.einsum(v7, [1, 2], v25, [0, 2], [0, 1])
    v30 = np.einsum(v2, [0, 1, 2], v27, [0, 2], [1])
    v32 = np.einsum(v3, [0, 1], v25, [0, 2], [1, 2])
    v34 = np.einsum(v6, [0, 1, 2], v32, [0, 2], [0, 1])
    v35 = v34.sum(axis=1)
    v36 = v34.sum(axis=0)
    v38 = np.einsum(v23, [0, 2], v9, [0, 1], [1, 2])
    v40 = np.einsum(v12, [0, 1, 2], v38, [0, 2], [0, 1])
    v41 = v40.sum(axis=1)
    v42 = v40.sum(axis=0)
    v44 = np.einsum(v15, [0, 1], v23, [0, 1], [1])
    out = {}
    out['x0'] = v21 / v21.sum()
    out['x1'] = v22 / v22.sum()
    out['x2'] = v30 / v30.sum()
    out['x3'] = v35 / v35.sum()
    out['x4'] = v36 / v36.sum()
    out['x5'] = v41 / v41.sum()
    out['x6'] = v42 / v42.sum()
    out['x7'] = v44 / v44.sum()
    return out
