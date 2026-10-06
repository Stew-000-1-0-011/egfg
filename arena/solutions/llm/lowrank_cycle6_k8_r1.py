import numpy as np

# Each factor T_j(x_j, x_{j+1}) has rank 1: T_j = a_j b_j^T with a_j ∝ T_j[:,0], b_j ∝ T_j[0,:].
# The joint factorizes completely, so p(x_j) ∝ a_j(x_j) * b_{j-1}(x_j).
_N = ('x0', 'x1', 'x2', 'x3', 'x4', 'x5')
_red = np.add.reduce


def infer(tables):
    t5 = tables[5]
    S = np.array((t5, tables[0], tables[1], tables[2], tables[3], tables[4], t5))
    R = S[1:, :, 0] * S[:-1, 0]
    R /= _red(R, 1, keepdims=True)
    return dict(zip(_N, R))
