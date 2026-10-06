import numpy as np

# Chain of ternary factors F_i(x_i, x_{i+1}, x_{i+2}): exact chain over pair super-variables
# s_i = (x_i, x_{i+1}) (9 states). F_i becomes a 9x9 transition with the shared variable on
# the diagonal (written through a writable einsum view); then .dot forward/backward messages.
_N = ('x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6')
_red = np.add.reduce
_ein = np.einsum
_zeros = np.zeros


def infer(tables):
    F = np.array((tables[0], tables[1], tables[2], tables[3], tables[4], tables[5]))
    M = _zeros((6, 3, 3, 3, 3))
    _ein('iabbc->iabc', M)[...] = F
    M0, M1, M2, M3, M4, M5, = M.reshape(6, 9, 9)
    r5 = _red(M5, 1)
    r4 = M4.dot(r5)
    r3 = M3.dot(r4)
    r2 = M2.dot(r3)
    r1 = M1.dot(r2)
    r0 = M0.dot(r1)
    l1 = _red(M0, 0)
    l2 = l1.dot(M1)
    l3 = l2.dot(M2)
    l4 = l3.dot(M3)
    l5 = l4.dot(M4)
    l6 = l5.dot(M5)
    z = 1.0 / r0.sum()
    B = np.array((r0, r1 * l1, r2 * l2, r3 * l3, r4 * l4, r5 * l5, l6)).reshape(7, 3, 3)
    B *= z
    d = dict(zip(_N, _red(B, 2)))
    d['x7'] = _red(B[6], 0)
    return d
