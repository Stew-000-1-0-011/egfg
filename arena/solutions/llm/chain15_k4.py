import numpy as np

# Chain with unary factors: absorb unaries into pairwise matrices M_i = diag(u_i) P_i,
# then forward (left) and backward (right) vector messages with plain .dot calls.
_N = tuple('x%d' % i for i in range(15))
_red = np.add.reduce


def infer(tables):
    P = np.array((tables[0], tables[1], tables[2], tables[3], tables[4], tables[5], tables[6], tables[7], tables[8], tables[9], tables[10], tables[11], tables[12], tables[13]))
    U = np.array((tables[14], tables[15], tables[16], tables[17], tables[18], tables[19], tables[20], tables[21], tables[22], tables[23], tables[24], tables[25], tables[26], tables[27], tables[28]))
    M = U[:14, :, None] * P
    M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, = M
    l1 = _red(M0, 0)
    l2 = l1.dot(M1)
    l3 = l2.dot(M2)
    l4 = l3.dot(M3)
    l5 = l4.dot(M4)
    l6 = l5.dot(M5)
    l7 = l6.dot(M6)
    l8 = l7.dot(M7)
    l9 = l8.dot(M8)
    l10 = l9.dot(M9)
    l11 = l10.dot(M10)
    l12 = l11.dot(M11)
    l13 = l12.dot(M12)
    l14 = l13.dot(M13)
    r14 = U[14]
    r13 = M13.dot(r14)
    r12 = M12.dot(r13)
    r11 = M11.dot(r12)
    r10 = M10.dot(r11)
    r9 = M9.dot(r10)
    r8 = M8.dot(r9)
    r7 = M7.dot(r8)
    r6 = M6.dot(r7)
    r5 = M5.dot(r6)
    r4 = M4.dot(r5)
    r3 = M3.dot(r4)
    r2 = M2.dot(r3)
    r1 = M1.dot(r2)
    r0 = M0.dot(r1)
    R = np.array((r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14))
    R[1:] *= (l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14)
    R /= r0.sum()
    return dict(zip(_N, R))
