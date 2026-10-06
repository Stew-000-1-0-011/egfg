import numpy as np

# sparse10: pendant subtrees (x5 on x1, x6 on x0, x2-x9 on x0, x8 on x3) are absorbed as
# unary messages (folded into the x0-x1-x3 path matrices). Conditioning on the separator (x0, x3) leaves two independent paths
# x0-x1-x3 and x0-x4-x7-x3 plus the direct factor (x0, x3). The paths are batched as
# P1 @ P2 @ P3 (identity pads the short one); W(x0,x3) is the exact pair marginal and each
# path's interior pair marginal is P2 * (P1^T (W/G) P3^T). Pendants get downward messages.
_red = np.add.reduce
_prod = np.multiply.reduce
_mm = np.matmul


def infer(tables):
    t1 = tables[1]; t4 = tables[4]; t5 = tables[5]; t7 = tables[7]; t8 = tables[8]
    one = _ones(3)
    g1 = t4.dot(one)                 # x5 -> x1
    m6 = t5.dot(one)                 # x6 -> x0
    r9 = t8.dot(one)                 # x9 -> x2
    m2 = t1.dot(r9)                  # x2 -> x0
    g3 = t7.dot(one)                 # x8 -> x3
    g0 = m6 * m2
    P1 = np.array((tables[0] * g0[:, None], tables[3]))
    P2 = np.array(((g1[:, None] * g3) * tables[9], tables[6]))
    P3 = np.array((((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), tables[10].T))
    G = _mm(_mm(P1, P2), P3)
    W = G[0] * G[1]
    W *= tables[2]
    W /= W.sum()
    pair = P2 * _mm(_mm(P1.swapaxes(1, 2), W / G), P3.swapaxes(1, 2))
    a = _red(pair, 2)
    b = _red(pair, 1)
    b0 = _red(W, 1)
    b3 = b[0]
    b1 = a[0]
    d2 = (b0 / m2).dot(t1)
    return {'x0': b0, 'x1': b1, 'x2': d2 * r9, 'x3': b3, 'x4': a[1], 'x5': (b1 / g1).dot(t4),
            'x6': (b0 / m6).dot(t5), 'x7': b[1], 'x8': (b3 / g3).dot(t7), 'x9': d2.dot(t8)}


_ones = np.ones
