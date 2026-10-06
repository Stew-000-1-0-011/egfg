"""egfg program for sparse10_k3 (cost model: 723 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[5]
    v1 = v0.sum(axis=1)
    v2 = tables[8]
    v3 = v2.sum(axis=1)
    v4 = tables[1]
    v6 = np.einsum(v4, [0, 1], v3, [1], [0])
    v7 = np.einsum(v6, [0], v1, [0], [0])
    v8 = tables[10]
    v9 = tables[6]
    v10 = np.einsum(v9, [1, 2], v8, [0, 2], [0, 1, 2])
    v11 = v10.sum(axis=2)
    v12 = tables[7]
    v13 = v12.sum(axis=1)
    v14 = np.einsum(v13, [0], v11, [0, 1], [0, 1])
    v15 = tables[4]
    v16 = v15.sum(axis=1)
    v17 = tables[9]
    v18 = tables[0]
    v19 = np.einsum(v18, [0, 1], v17, [1, 2], [0, 1, 2])
    v21 = np.einsum(v19, [0, 1, 2], v16, [1], [0, 2])
    v22 = tables[2]
    v23 = np.einsum(v22, [0, 1], v21, [0, 1], [0, 1])
    v25 = np.einsum(v23, [0, 1], v14, [1, 2], [0, 2])
    v26 = tables[3]
    v28 = np.einsum(v26, [0, 1], v25, [0, 1], [0])
    v29 = np.einsum(v28, [0], v7, [0], [0])
    v30 = np.einsum(v26, [0, 1], v7, [0], [0, 1])
    v31 = np.einsum(v22, [0, 1], v30, [0, 2], [0, 1, 2])
    v33 = np.einsum(v31, [0, 1, 2], v14, [1, 2], [0, 1])
    v36 = np.einsum(v19, [0, 1, 2], v33, [0, 2], [1])
    v37 = np.einsum(v36, [0], v16, [0], [0])
    v38 = np.einsum(v28, [0], v1, [0], [0])
    v40 = np.einsum(v4, [0, 1], v38, [0], [1])
    v41 = np.einsum(v40, [0], v3, [0], [0])
    v43 = np.einsum(v31, [0, 1, 2], v21, [0, 1], [1, 2])
    v44 = np.einsum(v43, [0, 1], v11, [0, 1], [0, 1])
    v45 = v44.sum(axis=1)
    v46 = np.einsum(v45, [0], v13, [0], [0])
    v48 = np.einsum(v13, [0], v44, [0, 1], [1])
    v50 = np.einsum(v15, [0, 1], v36, [0], [1])
    v51 = np.einsum(v28, [0], v6, [0], [0])
    v53 = np.einsum(v0, [0, 1], v51, [0], [1])
    v54 = np.einsum(v10, [0, 1, 2], v13, [0], [0, 1, 2])
    v57 = np.einsum(v54, [0, 1, 2], v43, [0, 1], [2])
    v59 = np.einsum(v12, [0, 1], v45, [0], [1])
    v61 = np.einsum(v2, [0, 1], v40, [0], [1])
    out = {}
    out['x0'] = v29 / v29.sum()
    out['x1'] = v37 / v37.sum()
    out['x2'] = v41 / v41.sum()
    out['x3'] = v46 / v46.sum()
    out['x4'] = v48 / v48.sum()
    out['x5'] = v50 / v50.sum()
    out['x6'] = v53 / v53.sum()
    out['x7'] = v57 / v57.sum()
    out['x8'] = v59 / v59.sum()
    out['x9'] = v61 / v61.sum()
    return out
