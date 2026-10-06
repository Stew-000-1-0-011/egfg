"""egfg program for tree12_k3 (cost model: 399 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[9]
    v1 = v0.sum(axis=1)
    v2 = tables[6]
    v3 = v2.sum(axis=1)
    v4 = np.transpose(tables[10], (1, 0))
    v5 = v4.sum(axis=0)
    v6 = tables[2]
    v8 = np.einsum(v6, [0, 1], v5, [1], [0])
    v9 = tables[8]
    v10 = v9.sum(axis=1)
    v11 = tables[4]
    v13 = np.einsum(v11, [0, 1], v10, [1], [0])
    v14 = tables[7]
    v15 = v14.sum(axis=1)
    v16 = tables[5]
    v18 = np.einsum(v16, [0, 1], v15, [1], [0])
    v19 = np.einsum(v18, [0], v13, [0], [0])
    v20 = tables[3]
    v22 = np.einsum(v20, [0, 1], v19, [1], [0])
    v23 = np.einsum(v22, [0], v8, [0], [0])
    v24 = tables[1]
    v26 = np.einsum(v24, [0, 1], v23, [1], [0])
    v27 = np.einsum(v26, [0], v3, [0], [0])
    v28 = tables[0]
    v30 = np.einsum(v28, [0, 1], v27, [1], [0])
    v31 = np.einsum(v30, [0], v1, [0], [0])
    v33 = np.einsum(v28, [0, 1], v1, [0], [1])
    v34 = np.einsum(v3, [0], v33, [0], [0])
    v35 = np.einsum(v26, [0], v34, [0], [0])
    v37 = np.einsum(v0, [0, 1], v30, [0], [1])
    v39 = np.einsum(v24, [0, 1], v34, [0], [1])
    v40 = np.einsum(v22, [0], v39, [0], [0])
    v42 = np.einsum(v6, [0, 1], v40, [0], [1])
    v44 = np.einsum(v4, [0, 1], v42, [1], [0])
    v45 = np.einsum(v40, [0], v8, [0], [0])
    v46 = np.einsum(v42, [0], v5, [0], [0])
    v47 = np.einsum(v39, [0], v8, [0], [0])
    v49 = np.einsum(v20, [0, 1], v47, [0], [1])
    v50 = np.einsum(v49, [0], v19, [0], [0])
    v51 = np.einsum(v18, [0], v49, [0], [0])
    v53 = np.einsum(v11, [0, 1], v51, [0], [1])
    v54 = np.einsum(v53, [0], v10, [0], [0])
    v55 = np.einsum(v13, [0], v49, [0], [0])
    v57 = np.einsum(v16, [0, 1], v55, [0], [1])
    v58 = np.einsum(v57, [0], v15, [0], [0])
    v59 = np.einsum(v26, [0], v33, [0], [0])
    v61 = np.einsum(v2, [0, 1], v59, [0], [1])
    v63 = np.einsum(v14, [0, 1], v57, [0], [1])
    v65 = np.einsum(v9, [0, 1], v53, [0], [1])
    out = {}
    out['x0'] = v31 / v31.sum()
    out['x1'] = v35 / v35.sum()
    out['x10'] = v37 / v37.sum()
    out['x11'] = v44 / v44.sum()
    out['x2'] = v45 / v45.sum()
    out['x3'] = v46 / v46.sum()
    out['x4'] = v50 / v50.sum()
    out['x5'] = v54 / v54.sum()
    out['x6'] = v58 / v58.sum()
    out['x7'] = v61 / v61.sum()
    out['x8'] = v63 / v63.sum()
    out['x9'] = v65 / v65.sum()
    return out
