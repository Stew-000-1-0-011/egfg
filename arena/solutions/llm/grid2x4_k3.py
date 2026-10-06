import numpy as np

# 2x4 grid: exact chain over column super-variables (x_j, x_{j+4}) with 9 states.
# Transition M_j = diag(vec V_j) (Top_j (x) Bot_j); forward/backward .dot messages; the
# column beliefs are reshaped to (3,3) and summed to give top and bottom marginals.
_NT = ('x0', 'x1', 'x2', 'x3')
_NB = ('x4', 'x5', 'x6', 'x7')
_red = np.add.reduce
_ein = np.einsum


def infer(tables):
    Vs = np.array((tables[1], tables[3], tables[5]))
    Tp = np.array((tables[0], tables[2], tables[4]))
    Bt = np.array((tables[7], tables[8], tables[9]))
    M0, M1, M2, = _ein('jab,jac,jbd->jabcd', Vs, Tp, Bt).reshape(3, 9, 9)
    r3 = tables[6].reshape(9)
    r2 = M2.dot(r3)
    r1 = M1.dot(r2)
    r0 = M0.dot(r1)
    l1 = _red(M0, 0)
    l2 = l1.dot(M1)
    l3 = l2.dot(M2)
    B = np.array((r0, r1 * l1, r2 * l2, r3 * l3)).reshape(4, 3, 3)
    B /= r0.sum()
    d = dict(zip(_NT, _red(B, 2)))
    d.update(zip(_NB, _red(B, 1)))
    return d
