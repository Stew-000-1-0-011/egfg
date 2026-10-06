import numpy as np

# Star: center x0, leaves x1..x6 with factors T_i(x0, x_i).
# Leaf->center messages are row sums; center belief is their product; center->leaf
# messages are belief / own message (tables are positive). All marginals share the
# same normaliser Z = sum(center belief), which is folded into the downward messages.
_NL = ('x1', 'x2', 'x3', 'x4', 'x5', 'x6')
_red = np.add.reduce
_prod = np.multiply.reduce
_mm = np.matmul


def infer(tables):
    S = np.array((tables[0], tables[1], tables[2], tables[3], tables[4], tables[5]))
    M = _red(S, 2)
    c = _prod(M, 0)
    c /= c.sum()
    L = _mm((c / M)[:, None], S)[:, 0]
    d = dict(zip(_NL, L))
    d["x0"] = c
    return d
