#pragma GCC optimize("fp-contract=fast")
#include <string.h>
/* Cycle of N variables, factor i = T_i(x_i, x_{i+1 mod N}).
   Pt[i] = (T_0...T_{i-1})^T  (rows x_i, cols x_0),  S[i] = T_i...T_{N-1}  (rows x_i, cols x_0).
   marg_i(a) = sum_{x0} S_i[a][x0] Pt_i[a][x0]. Both recursions use scalars of T from memory
   broadcast against 4-wide rows. */
#define K 4
#define N 8
typedef double vK __attribute__((vector_size(32)));
typedef long long iK __attribute__((vector_size(32)));
static inline vK ld(const double *p) { vK v; memcpy(&v, p, sizeof v); return v; }
static inline void st(double *p, vK v) { memcpy(p, &v, sizeof v); }
static inline vK hsum(vK p0, vK p1, vK p2, vK p3) {
  vK u0 = __builtin_shuffle(p0, p1, (iK){0,4,2,6}) + __builtin_shuffle(p0, p1, (iK){1,5,3,7});
  vK u1 = __builtin_shuffle(p2, p3, (iK){0,4,2,6}) + __builtin_shuffle(p2, p3, (iK){1,5,3,7});
  return __builtin_shuffle(u0, u1, (iK){0,1,4,5}) + __builtin_shuffle(u0, u1, (iK){2,3,6,7});
}
typedef struct { vK r[K]; } M4;
/* Pt_{i+1}[a] = sum_b T[b][a] Pt_i[b] */
static inline M4 stepP(const double *T, M4 P) {
  M4 o;
  for (int a = 0; a < K; a++) o.r[a] = (T[a]*P.r[0] + T[K+a]*P.r[1]) + (T[2*K+a]*P.r[2] + T[3*K+a]*P.r[3]);
  return o;
}
/* S_i[a] = sum_b T[a][b] S_{i+1}[b] */
static inline M4 stepS(const double *T, M4 S) {
  M4 o;
  for (int a = 0; a < K; a++) o.r[a] = (T[a*K]*S.r[0] + T[a*K+1]*S.r[1]) + (T[a*K+2]*S.r[2] + T[a*K+3]*S.r[3]);
  return o;
}
static inline vK dg(M4 S, M4 P) { return hsum(S.r[0]*P.r[0], S.r[1]*P.r[1], S.r[2]*P.r[2], S.r[3]*P.r[3]); }
void infer(const double *const *tables, double *out) {
  M4 P[N], S[N];
  {
    const double *T0 = tables[0];
    vK r0 = ld(T0), r1 = ld(T0+K), r2 = ld(T0+2*K), r3 = ld(T0+3*K);
    vK a0 = __builtin_shuffle(r0, r1, (iK){0,4,2,6}), a1 = __builtin_shuffle(r0, r1, (iK){1,5,3,7});
    vK a2 = __builtin_shuffle(r2, r3, (iK){0,4,2,6}), a3 = __builtin_shuffle(r2, r3, (iK){1,5,3,7});
    P[1].r[0] = __builtin_shuffle(a0, a2, (iK){0,1,4,5});
    P[1].r[1] = __builtin_shuffle(a1, a3, (iK){0,1,4,5});
    P[1].r[2] = __builtin_shuffle(a0, a2, (iK){2,3,6,7});
    P[1].r[3] = __builtin_shuffle(a1, a3, (iK){2,3,6,7});
    const double *TL = tables[N-1];
    for (int a = 0; a < K; a++) S[N-1].r[a] = ld(TL + a*K);
  }
  for (int i = 1; i < N-1; i++) {
    P[i+1] = stepP(tables[i], P[i]);
    S[N-1-i] = stepS(tables[N-1-i], S[N-i]);
  }
  static const int pos[N] = {0,1,2,3,4,5,6,7};
  vK m[N];
  m[0] = (P[1].r[0]*S[1].r[0] + P[1].r[1]*S[1].r[1]) + (P[1].r[2]*S[1].r[2] + P[1].r[3]*S[1].r[3]);
  for (int i = 1; i < N; i++) m[i] = dg(S[i], P[i]);
  /* all marginals sum to Z = trace of the cycle product */
  const double rz = 1.0 / ((m[0][0] + m[0][1]) + (m[0][2] + m[0][3]));
  for (int i = 0; i < N; i++) st(out + pos[i]*K, m[i] * rz);
}
