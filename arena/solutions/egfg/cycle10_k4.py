"""egfg program for cycle10_k4 (cost model: 2496 operations)."""
import numpy as np


def infer(tables):
    v0 = np.transpose(tables[9], (1, 0))
    v1 = tables[8]
    v2 = np.einsum(v1, [1, 2], v0, [0, 2], [0, 1, 2])
    v3 = v2.sum(axis=2)
    v4 = tables[2]
    v5 = tables[1]
    v6 = np.einsum(v5, [0, 1], v4, [1, 2], [0, 1, 2])
    v7 = v6.sum(axis=1)
    v8 = tables[3]
    v10 = np.einsum(v8, [1, 2], v7, [0, 1], [0, 2])
    v11 = tables[7]
    v12 = tables[6]
    v13 = np.einsum(v12, [0, 1], v11, [1, 2], [0, 1, 2])
    v14 = v13.sum(axis=1)
    v15 = tables[5]
    v16 = tables[4]
    v17 = np.einsum(v16, [0, 1], v15, [1, 2], [0, 1, 2])
    v18 = v17.sum(axis=1)
    v20 = np.einsum(v18, [0, 1], v14, [1, 2], [0, 2])
    v22 = np.einsum(v20, [1, 2], v10, [0, 1], [0, 2])
    v23 = tables[0]
    v25 = np.einsum(v23, [0, 1], v22, [1, 2], [0, 2])
    v26 = np.einsum(v25, [0, 1], v3, [0, 1], [0, 1])
    v27 = v26.sum(axis=1)
    v29 = np.einsum(v23, [0, 1], v3, [0, 2], [1, 2])
    v31 = np.einsum(v20, [1, 2], v29, [0, 2], [0, 1])
    v33 = np.einsum(v8, [1, 2], v31, [0, 2], [0, 1])
    v35 = np.einsum(v6, [0, 1, 2], v33, [0, 2], [0, 1])
    v36 = v35.sum(axis=1)
    v37 = v35.sum(axis=0)
    v39 = np.einsum(v33, [0, 1], v7, [0, 1], [1])
    v41 = np.einsum(v29, [0, 2], v10, [0, 1], [1, 2])
    v43 = np.einsum(v14, [1, 2], v41, [0, 2], [0, 1])
    v45 = np.einsum(v17, [0, 1, 2], v43, [0, 2], [0, 1])
    v46 = v45.sum(axis=1)
    v47 = v45.sum(axis=0)
    v49 = np.einsum(v18, [0, 1], v41, [0, 2], [1, 2])
    v51 = np.einsum(v13, [0, 1, 2], v49, [0, 2], [0, 1])
    v52 = v51.sum(axis=1)
    v53 = v51.sum(axis=0)
    v54 = v26.sum(axis=0)
    v57 = np.einsum(v2, [0, 1, 2], v25, [0, 1], [2])
    out = {}
    out['x0'] = v27 / v27.sum()
    out['x1'] = v36 / v36.sum()
    out['x2'] = v37 / v37.sum()
    out['x3'] = v39 / v39.sum()
    out['x4'] = v46 / v46.sum()
    out['x5'] = v47 / v47.sum()
    out['x6'] = v52 / v52.sum()
    out['x7'] = v53 / v53.sum()
    out['x8'] = v54 / v54.sum()
    out['x9'] = v57 / v57.sum()
    return out
