#pragma GCC optimize("fp-contract=fast")
#include <string.h>
/* Chain of ternary factors f_i(x_i, x_{i+1}, x_{i+2}): junction chain with separators
   (x_{i+1}, x_{i+2}). Messages are 3x3, stored as 3 rows of 4-lane vectors (rows = first
   separator variable, lanes = second). F[i]: into clique i over (x_i, x_{i+1});
   B[i]: into clique i over (x_{i+1}, x_{i+2}). Marginals come from separators; all
   unnormalized marginals sum to Z, so one reciprocal normalizes everything. */
#define K 3
#define NV 6
#define M (NV-2)
typedef double v4 __attribute__((vector_size(32)));
typedef long long i4 __attribute__((vector_size(32)));
static inline v4 ld4(const double *p) { v4 v; memcpy(&v, p, sizeof v); return v; }
static inline v4 ld(const double *p) { return __builtin_shuffle(ld4(p), (v4){0}, (i4){0,1,2,4}); }
static inline v4 ldl(const double *p) { return __builtin_shuffle(ld4(p-1), (v4){0}, (i4){1,2,3,4}); }
static inline void st3(double *p, v4 v) { memcpy(p, &v, 3*sizeof(double)); }
static inline v4 hsum(v4 p0, v4 p1, v4 p2, v4 p3) {
  v4 u0 = __builtin_shuffle(p0, p1, (i4){0,4,2,6}) + __builtin_shuffle(p0, p1, (i4){1,5,3,7});
  v4 u1 = __builtin_shuffle(p2, p3, (i4){0,4,2,6}) + __builtin_shuffle(p2, p3, (i4){1,5,3,7});
  return __builtin_shuffle(u0, u1, (i4){0,1,4,5}) + __builtin_shuffle(u0, u1, (i4){2,3,6,7});
}
/* row (a,b) of a 3x3x3 table, lane 3 = 0 */
static inline v4 row(const double *f, int ab) { return ab == 8 ? ldl(f + 24) : ld(f + 3*ab); }
/* same, lane 3 arbitrary (finite) */
static inline v4 rowx(const double *f, int ab) { return ab == 8 ? ldl(f + 24) : ld4(f + 3*ab); }
typedef struct { v4 r[3]; } S3;
void infer(const double *const *tables, double *out) {
  S3 F[M], B[M];
  {
    const double *f = tables[0];
    for (int b = 0; b < 3; b++) F[1].r[b] = rowx(f, b) + rowx(f, 3+b) + rowx(f, 6+b);
    const double *g = tables[M-1];
    for (int y = 0; y < 3; y++) B[M-2].r[y] = hsum(row(g, 3*y), row(g, 3*y+1), row(g, 3*y+2), (v4){0});
  }
  for (int st = 1; st < M-1; st++) {
    int i = st;
    const double *f = tables[i];
    /* F[i+1](b,c) = sum_a F[i](a,b) f[a][b][c] */
    for (int b = 0; b < 3; b++)
      F[i+1].r[b] = F[i].r[0][b]*rowx(f, b) + F[i].r[1][b]*rowx(f, 3+b) + F[i].r[2][b]*rowx(f, 6+b);
    int j = M-2-st;
    const double *g = tables[j+1];
    /* B[j](y,z) = sum_w g[y][z][w] B[j+1](z,w) */
    for (int y = 0; y < 3; y++)
      B[j].r[y] = hsum(rowx(g, 3*y)*B[j+1].r[0], rowx(g, 3*y+1)*B[j+1].r[1], rowx(g, 3*y+2)*B[j+1].r[2], (v4){0});
  }
  /* first separator gives Z */
  v4 mu0 = F[1].r[0]*B[0].r[0], mu1 = F[1].r[1]*B[0].r[1], mu2 = F[1].r[2]*B[0].r[2];
  v4 cs = (mu0 + mu1) + mu2;
  const double rz = 1.0 / ((cs[0] + cs[1]) + cs[2]);
  st3(out + 1*K, hsum(mu0, mu1, mu2, (v4){0}) * rz);
  if (M == 2) st3(out + 2*K, cs * rz);
  /* x0: t(a) = sum_{b,c} f0[a][b][c] B0(b,c) */
  {
    const double *f = tables[0];
    v4 q[3];
    for (int a = 0; a < 3; a++) q[a] = rowx(f, 3*a)*B[0].r[0] + rowx(f, 3*a+1)*B[0].r[1] + rowx(f, 3*a+2)*B[0].r[2];
    st3(out, hsum(q[0], q[1], q[2], (v4){0}) * rz);
  }
  for (int i = 1; i < M-1; i++) {
    v4 m0 = F[i+1].r[0]*B[i].r[0], m1 = F[i+1].r[1]*B[i].r[1], m2 = F[i+1].r[2]*B[i].r[2];
    st3(out + (i+1)*K, hsum(m0, m1, m2, (v4){0}) * rz);
    if (i == M-2) st3(out + (i+2)*K, ((m0 + m1) + m2) * rz);
  }
  /* x_{NV-1}: t(c) = sum_{a,b} F(a,b) f[a][b][c] */
  {
    const double *f = tables[M-1];
    v4 t = (v4){0};
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) t += F[M-1].r[a][b]*row(f, 3*a+b);
    st3(out + (NV-1)*K, t * rz);
  }
}
