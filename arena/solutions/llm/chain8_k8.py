import numpy as np

# Chain with unary factors: absorb unaries into pairwise matrices M_i = diag(u_i) P_i,
# then forward (left) and backward (right) vector messages with plain .dot calls.
_N = tuple('x%d' % i for i in range(8))
_red = np.add.reduce


def infer(tables):
    P = np.array((tables[0], tables[1], tables[2], tables[3], tables[4], tables[5], tables[6]))
    U = np.array((tables[7], tables[8], tables[9], tables[10], tables[11], tables[12], tables[13], tables[14]))
    M = U[:7, :, None] * P
    M0, M1, M2, M3, M4, M5, M6, = M
    l1 = _red(M0, 0)
    l2 = l1.dot(M1)
    l3 = l2.dot(M2)
    l4 = l3.dot(M3)
    l5 = l4.dot(M4)
    l6 = l5.dot(M5)
    l7 = l6.dot(M6)
    r7 = U[7]
    r6 = M6.dot(r7)
    r5 = M5.dot(r6)
    r4 = M4.dot(r5)
    r3 = M3.dot(r4)
    r2 = M2.dot(r3)
    r1 = M1.dot(r2)
    r0 = M0.dot(r1)
    R = np.array((r0, r1, r2, r3, r4, r5, r6, r7))
    R[1:] *= (l1, l2, l3, l4, l5, l6, l7)
    R /= r0.sum()
    return dict(zip(_N, R))
