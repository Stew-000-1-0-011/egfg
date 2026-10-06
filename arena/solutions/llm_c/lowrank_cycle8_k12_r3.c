#pragma GCC optimize("fp-contract=fast")
#include <string.h>
#define K 12
#define N 8
/* Cycle with factor j = T_j(x_j, x_{j+1}), each of rank 3.
   Skeleton decomposition: T_j = C_j W_j^{-1} R_j with C_j = T_j[:,0:3], R_j = T_j[0:3,:],
   W_j = T_j[0:3,0:3]. W^{-1} is replaced by adj(W) (the 1/det factors are common to every
   marginal and cancel in normalization).
   marg_i(a) = [T_i T_{i+1} ... T_{i-1}]_{aa}
             = sum_p C_i[a,p] Qt_i[a,p],  Qt_i = R_{i-1}^T (H_i adj_{i-1})^T,
   H_i = G_i G_{i+1} ... G_{i-2}, G_j = adj_j (R_j C_{j+1}).
   3x3 matrices are stored as 3 rows of v4 (lane 3 unused). */
typedef double v4 __attribute__((vector_size(32)));
static inline v4 ld4(const double *p) { v4 v; memcpy(&v, p, sizeof v); return v; }
typedef struct { v4 r[3]; } M3;
static inline M3 mm(M3 A, M3 B) {
  M3 C;
  for (int i = 0; i < 3; i++) C.r[i] = A.r[i][0]*B.r[0] + A.r[i][1]*B.r[1] + A.r[i][2]*B.r[2];
  return C;
}
static inline M3 adj3(const double *T) {
  double w00=T[0], w01=T[1], w02=T[2], w10=T[K], w11=T[K+1], w12=T[K+2], w20=T[2*K], w21=T[2*K+1], w22=T[2*K+2];
  M3 a;
  a.r[0] = (v4){w11*w22 - w12*w21, w02*w21 - w01*w22, w01*w12 - w02*w11, 0};
  a.r[1] = (v4){w12*w20 - w10*w22, w00*w22 - w02*w20, w02*w10 - w00*w12, 0};
  a.r[2] = (v4){w10*w21 - w11*w20, w01*w20 - w00*w21, w00*w11 - w01*w10, 0};
  return a;
}
static inline v4 hsum4(v4 p0, v4 p1, v4 p2, v4 p3) {
  v4 u0 = __builtin_shuffle(p0, p1, (long long __attribute__((vector_size(32)))){0,4,2,6})
        + __builtin_shuffle(p0, p1, (long long __attribute__((vector_size(32)))){1,5,3,7});
  v4 u1 = __builtin_shuffle(p2, p3, (long long __attribute__((vector_size(32)))){0,4,2,6})
        + __builtin_shuffle(p2, p3, (long long __attribute__((vector_size(32)))){1,5,3,7});
  return __builtin_shuffle(u0, u1, (long long __attribute__((vector_size(32)))){0,1,4,5})
       + __builtin_shuffle(u0, u1, (long long __attribute__((vector_size(32)))){2,3,6,7});
}
void infer(const double *const *tables, double *out) {
  M3 A[N], G[N], Pre[N+1], Suf[N], H[N];
  for (int j = 0; j < N; j++) {
    const double *T = tables[j], *U = tables[(j+1) % N];
    A[j] = adj3(T);
    M3 Kj;
    for (int p = 0; p < 3; p++) {
      v4 s0 = T[p*K]*ld4(U), s1 = T[p*K+1]*ld4(U+K), s2 = T[p*K+2]*ld4(U+2*K);
      for (int y = 3; y < K; y += 3) {
        s0 += T[p*K+y]*ld4(U+y*K); s1 += T[p*K+y+1]*ld4(U+(y+1)*K); s2 += T[p*K+y+2]*ld4(U+(y+2)*K);
      }
      Kj.r[p] = s0 + s1 + s2;
    }
    G[j] = mm(A[j], Kj);
  }
  Pre[1] = G[0];
  for (int j = 1; j < N; j++) Pre[j+1] = mm(Pre[j], G[j]);
  Suf[N-1] = G[N-1];
  for (int j = N-2; j >= 1; j--) Suf[j] = mm(G[j], Suf[j+1]);
  H[0] = Pre[N-1];
  H[1] = Suf[1];
  for (int i = 2; i < N; i++) H[i] = mm(Suf[i], Pre[i-1]);
  for (int i = 0; i < N; i++) {
    const double *C = tables[i], *Rm = tables[(i+N-1) % N];
    M3 L = mm(H[i], A[(i+N-1) % N]);
    /* LT[q] = column q of L (over p), lane 3 = 0 */
    v4 LT0 = {L.r[0][0], L.r[1][0], L.r[2][0], 0};
    v4 LT1 = {L.r[0][1], L.r[1][1], L.r[2][1], 0};
    v4 LT2 = {L.r[0][2], L.r[1][2], L.r[2][2], 0};
    v4 m[3];
    for (int g = 0; g < 3; g++) {
      v4 pr[4];
      for (int t = 0; t < 4; t++) {
        int a = 4*g + t;
        v4 qt = Rm[a]*LT0 + Rm[K+a]*LT1 + Rm[2*K+a]*LT2;
        pr[t] = ld4(C + a*K) * qt;
      }
      m[g] = hsum4(pr[0], pr[1], pr[2], pr[3]);
    }
    v4 s4 = m[0] + m[1] + m[2];
    double r = 1.0 / (s4[0] + s4[1] + s4[2] + s4[3]);
    for (int g = 0; g < 3; g++) { v4 o = m[g]*r; memcpy(out + i*K + 4*g, &o, sizeof o); }
  }
}
