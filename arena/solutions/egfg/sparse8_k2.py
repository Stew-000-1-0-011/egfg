"""egfg program for sparse8_k2 (cost model: 254 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[2]
    v1 = tables[0]
    v2 = np.einsum(v1, [0, 1], v0, [1, 2], [0, 1, 2])
    v3 = v2.sum(axis=1)
    v4 = tables[7]
    v5 = tables[5]
    v6 = np.einsum(v5, [0, 2], v4, [1, 2], [0, 1, 2])
    v7 = v6.sum(axis=2)
    v8 = tables[3]
    v9 = v8.sum(axis=1)
    v10 = np.einsum(v9, [1], v7, [0, 1], [0, 1])
    v11 = tables[1]
    v13 = np.einsum(v11, [0, 1], v10, [1, 2], [0, 2])
    v14 = tables[6]
    v15 = np.einsum(v14, [1, 2], v13, [0, 1], [0, 1, 2])
    v17 = np.einsum(v15, [0, 1, 2], v3, [0, 1], [0, 2])
    v18 = tables[8]
    v19 = tables[4]
    v20 = np.einsum(v19, [0, 1], v18, [1, 2], [0, 1, 2])
    v22 = np.einsum(v20, [0, 1, 2], v17, [0, 2], [0, 1])
    v23 = v22.sum(axis=1)
    v24 = v20.sum(axis=1)
    v25 = np.einsum(v14, [1, 2], v24, [0, 2], [0, 1, 2])
    v27 = np.einsum(v25, [0, 1, 2], v13, [0, 1], [0, 1])
    v30 = np.einsum(v2, [0, 1, 2], v27, [0, 2], [1])
    v32 = np.einsum(v25, [0, 1, 2], v3, [0, 1], [0, 1])
    v34 = np.einsum(v11, [0, 1], v32, [0, 2], [1, 2])
    v35 = np.einsum(v6, [0, 1, 2], v9, [1], [0, 1, 2])
    v37 = np.einsum(v35, [0, 1, 2], v34, [0, 1], [0, 2])
    v38 = v37.sum(axis=1)
    v40 = np.einsum(v34, [0, 1], v7, [0, 1], [1])
    v41 = np.einsum(v40, [0], v9, [0], [0])
    v43 = np.einsum(v8, [0, 1], v40, [0], [1])
    v44 = v22.sum(axis=0)
    v45 = v37.sum(axis=0)
    v47 = np.einsum(v17, [0, 1], v24, [0, 1], [1])
    out = {}
    out['x0'] = v23 / v23.sum()
    out['x1'] = v30 / v30.sum()
    out['x2'] = v38 / v38.sum()
    out['x3'] = v41 / v41.sum()
    out['x4'] = v43 / v43.sum()
    out['x5'] = v44 / v44.sum()
    out['x6'] = v45 / v45.sum()
    out['x7'] = v47 / v47.sum()
    return out
