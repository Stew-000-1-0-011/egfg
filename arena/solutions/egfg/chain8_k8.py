"""egfg program for chain8_k8 (cost model: 1936 operations)."""
import numpy as np


def infer(tables):
    v0 = tables[14]
    v1 = tables[6]
    v2 = np.einsum(v1, [0, 1], v0, [1], [0, 1])
    v3 = v2.sum(axis=1)
    v4 = tables[13]
    v5 = np.einsum(v4, [0], v3, [0], [0])
    v6 = tables[5]
    v8 = np.einsum(v6, [0, 1], v5, [1], [0])
    v9 = tables[12]
    v10 = np.einsum(v9, [0], v8, [0], [0])
    v11 = tables[4]
    v13 = np.einsum(v11, [0, 1], v10, [1], [0])
    v14 = tables[11]
    v15 = np.einsum(v14, [0], v13, [0], [0])
    v16 = tables[3]
    v18 = np.einsum(v16, [0, 1], v15, [1], [0])
    v19 = tables[10]
    v20 = np.einsum(v19, [0], v18, [0], [0])
    v21 = tables[2]
    v23 = np.einsum(v21, [0, 1], v20, [1], [0])
    v24 = tables[9]
    v25 = np.einsum(v24, [0], v23, [0], [0])
    v26 = tables[1]
    v28 = np.einsum(v26, [0, 1], v25, [1], [0])
    v29 = tables[8]
    v30 = np.einsum(v29, [0], v28, [0], [0])
    v31 = tables[7]
    v32 = tables[0]
    v33 = np.einsum(v32, [0, 1], v31, [0], [0, 1])
    v35 = np.einsum(v33, [0, 1], v30, [1], [0])
    v36 = v33.sum(axis=0)
    v37 = np.einsum(v29, [0], v36, [0], [0])
    v38 = np.einsum(v28, [0], v37, [0], [0])
    v40 = np.einsum(v26, [0, 1], v37, [0], [1])
    v41 = np.einsum(v24, [0], v40, [0], [0])
    v42 = np.einsum(v23, [0], v41, [0], [0])
    v44 = np.einsum(v21, [0, 1], v41, [0], [1])
    v45 = np.einsum(v19, [0], v44, [0], [0])
    v46 = np.einsum(v18, [0], v45, [0], [0])
    v48 = np.einsum(v16, [0, 1], v45, [0], [1])
    v49 = np.einsum(v14, [0], v48, [0], [0])
    v50 = np.einsum(v13, [0], v49, [0], [0])
    v52 = np.einsum(v11, [0, 1], v49, [0], [1])
    v53 = np.einsum(v9, [0], v52, [0], [0])
    v54 = np.einsum(v8, [0], v53, [0], [0])
    v56 = np.einsum(v6, [0, 1], v53, [0], [1])
    v57 = np.einsum(v4, [0], v56, [0], [0])
    v58 = np.einsum(v3, [0], v57, [0], [0])
    v60 = np.einsum(v2, [0, 1], v57, [0], [1])
    out = {}
    out['x0'] = v35 / v35.sum()
    out['x1'] = v38 / v38.sum()
    out['x2'] = v42 / v42.sum()
    out['x3'] = v46 / v46.sum()
    out['x4'] = v50 / v50.sum()
    out['x5'] = v54 / v54.sum()
    out['x6'] = v58 / v58.sum()
    out['x7'] = v60 / v60.sum()
    return out
