import numpy as np

# sparse8: condition on the separator (x0, x3). Given it, the graph splits into three
# independent paths x0-x1-x3, x0-x2-x6-x3, x0-x5-x7-x3 (+ pendant x4 on x3).
# Paths are batched as P1 @ P2 @ P3 (identity pads the short one). W(x0,x3) is the exact
# pair marginal; each path's interior pair marginal is P2 * (P1^T (W/G) P3^T).
_red = np.add.reduce
_prod = np.multiply.reduce
_mm = np.matmul


def infer(tables):
    t3 = tables[3]
    P1 = np.array((tables[0], tables[1], tables[4]))
    P2 = np.array((tables[2], tables[5], tables[8]))
    P3 = np.array((((1.0, 0.0), (0.0, 1.0)), tables[7].T, tables[6].T))
    G = _mm(_mm(P1, P2), P3)
    g3 = _red(t3, 1)
    W = _prod(G, 0)
    W *= g3
    W /= W.sum()
    pair = P2 * _mm(_mm(P1.swapaxes(1, 2), W / G), P3.swapaxes(1, 2))
    a = _red(pair, 2)
    b = _red(pair, 1)
    m3 = b[0]
    return {'x0': _red(W, 1), 'x1': a[0], 'x2': a[1], 'x3': m3, 'x4': (m3 / g3).dot(t3),
            'x5': a[2], 'x6': b[1], 'x7': b[2]}

