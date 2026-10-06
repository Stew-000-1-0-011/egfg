import numpy as np

# Cycle x0-x1-...-x7-x0 with pairwise factors T_i(x_i, x_{i+1}).
# p(x_i) ∝ diag(T_i T_{i+1} ... T_{i-1}). All 8 cyclic products are formed by
# doubling over a stacked (wrapped) sequence with two batched matmuls.
_N = ('x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7')
_red = np.add.reduce
_mm = np.matmul
_ein = np.einsum


def infer(tables):
    t0 = tables[0]; t1 = tables[1]; t2 = tables[2]; t3 = tables[3]
    t4 = tables[4]; t5 = tables[5]; t6 = tables[6]
    S = np.array((t0, t1, t2, t3, t4, t5, t6, tables[7], t0, t1, t2, t3, t4, t5, t6))
    X2 = _mm(S[:-1], S[1:])          # X2[i] = T_i T_{i+1}, i = 0..13
    X4 = _mm(X2[:-2], X2[2:])        # X4[i] = T_i..T_{i+3}, i = 0..11
    R = _ein('iab,iba->ia', X4[:8], X4[4:])
    R /= R[0].sum()
    return dict(zip(_N, R))
