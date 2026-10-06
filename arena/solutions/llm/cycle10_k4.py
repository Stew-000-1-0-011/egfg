import numpy as np

# Cycle x0-...-x9-x0 with pairwise factors T_i(x_i, x_{i+1}).
# p(x_i) ∝ diag(T_i ... T_{i-1}); all 10 cyclic products via batched doubling.
_N = ('x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9')
_red = np.add.reduce
_mm = np.matmul
_ein = np.einsum


def infer(tables):
    t0 = tables[0]; t1 = tables[1]; t2 = tables[2]; t3 = tables[3]; t4 = tables[4]
    t5 = tables[5]; t6 = tables[6]; t7 = tables[7]; t8 = tables[8]
    S = np.array((t0, t1, t2, t3, t4, t5, t6, t7, t8, tables[9], t0, t1, t2, t3, t4, t5, t6, t7, t8))
    X2 = _mm(S[:-1], S[1:])          # X2[i] = T_i T_{i+1}, i = 0..17
    X4 = _mm(X2[:14], X2[2:16])      # X4[i] = T_i..T_{i+3}, i = 0..13
    X8 = _mm(X4[:10], X4[4:])        # X8[i] = T_i..T_{i+7}, i = 0..9
    R = _ein('iab,iba->ia', X8, X2[8:])
    R /= R[0].sum()
    return dict(zip(_N, R))
