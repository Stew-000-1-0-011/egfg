import numpy as np

# 2x3 binary grid: the joint has only 64 states, so build it with a single einsum
# (no summed indices), then stack the 6 axis-permuted views and reduce once.
_N = ('x0', 'x1', 'x2', 'x3', 'x4', 'x5')
_red = np.add.reduce
_ein = np.einsum


def infer(tables):
    J = _ein('ab,ad,bc,be,cf,de,ef->abcdef', tables[0], tables[1], tables[2], tables[3],
             tables[4], tables[5], tables[6])
    tr = J.transpose
    R = _red(np.array((J, tr(1, 0, 2, 3, 4, 5), tr(2, 0, 1, 3, 4, 5), tr(3, 0, 1, 2, 4, 5),
                       tr(4, 0, 1, 2, 3, 5), tr(5, 0, 1, 2, 3, 4))).reshape(6, 2, 32), 2)
    R /= R[0].sum()
    return dict(zip(_N, R))
