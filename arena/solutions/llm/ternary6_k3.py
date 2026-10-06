import numpy as np

# Chain of ternary factors F_i(x_i, x_{i+1}, x_{i+2}): exact chain over pair super-variables
# s_i = (x_i, x_{i+1}) (9 states). F_i becomes a 9x9 transition with the shared variable on
# the diagonal (written through a writable einsum view); then .dot forward/backward messages.
_N = ('x0', 'x1', 'x2', 'x3', 'x4')
_red = np.add.reduce
_ein = np.einsum
_zeros = np.zeros


def infer(tables):
    F = np.array((tables[0], tables[1], tables[2], tables[3]))
    M = _zeros((4, 3, 3, 3, 3))
    _ein('iabbc->iabc', M)[...] = F
    M0, M1, M2, M3, = M.reshape(4, 9, 9)
    r3 = _red(M3, 1)
    r2 = M2.dot(r3)
    r1 = M1.dot(r2)
    r0 = M0.dot(r1)
    l1 = _red(M0, 0)
    l2 = l1.dot(M1)
    l3 = l2.dot(M2)
    l4 = l3.dot(M3)
    z = 1.0 / r0.sum()
    B = np.array((r0, r1 * l1, r2 * l2, r3 * l3, l4)).reshape(5, 3, 3)
    B *= z
    d = dict(zip(_N, _red(B, 2)))
    d['x5'] = _red(B[4], 0)
    return d
