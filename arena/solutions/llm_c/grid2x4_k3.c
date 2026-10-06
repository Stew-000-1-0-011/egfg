#pragma GCC optimize("fp-contract=fast")
#include <string.h>
/* 2x4 ladder: top x_c, bottom x_{c+4} (c = 0..3). Column c is a supernode (t,b) with unary
   V_c(t,b); transitions are separable: H_c(t,t') * B_c(b,b'). Forward/backward over the
   4 supernodes; supernode beliefs are stored as 3 rows (t) of 4-lane vectors over b
   (lane 3 is kept at 0). */
#define K 3
#define NC 4
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
static const int HI[NC-1] = {0, 2, 4};
static const int BI[NC-1] = {7, 8, 9};
static const int VI[NC] = {1, 3, 5, 6};
typedef struct { v4 r[3]; } S3;
static inline S3 ldm(const double *T) { S3 m; m.r[0] = ld(T); m.r[1] = ld(T+3); m.r[2] = ldl(T+6); return m; }
static inline S3 ldmT(const double *T) {
  S3 m;
  for (int j = 0; j < 3; j++) m.r[j] = (v4){T[j], T[3+j], T[6+j], 0};
  return m;
}
void infer(const double *const *tables, double *out) {
  S3 al[NC], be[NC];
  al[0] = ldm(tables[VI[0]]);
  for (int t = 0; t < 3; t++) be[NC-1].r[t] = (v4){1, 1, 1, 0};
  for (int st = 0; st < NC-1; st++) {
    {
      int c = st;
      const double *H = tables[HI[c]];
      S3 B = ldm(tables[BI[c]]), V = ldm(tables[VI[c+1]]);
      for (int t2 = 0; t2 < 3; t2++) {
        v4 tau = H[t2]*al[c].r[0] + H[3+t2]*al[c].r[1] + H[6+t2]*al[c].r[2];
        v4 sg = tau[0]*B.r[0] + tau[1]*B.r[1] + tau[2]*B.r[2];
        al[c+1].r[t2] = sg * V.r[t2];
      }
    }
    {
      int c = NC-2-st;
      const double *H = tables[HI[c]];
      S3 BT = ldmT(tables[BI[c]]), V = ldm(tables[VI[c+1]]);
      v4 g0 = V.r[0]*be[c+1].r[0], g1 = V.r[1]*be[c+1].r[1], g2 = V.r[2]*be[c+1].r[2];
      for (int t = 0; t < 3; t++) {
        v4 rho = H[3*t]*g0 + H[3*t+1]*g1 + H[3*t+2]*g2;
        be[c].r[t] = rho[0]*BT.r[0] + rho[1]*BT.r[1] + rho[2]*BT.r[2];
      }
    }
  }
  v4 top[NC], bot[NC];
  for (int c = 0; c < NC; c++) {
    v4 m0 = al[c].r[0]*be[c].r[0], m1 = al[c].r[1]*be[c].r[1], m2 = al[c].r[2]*be[c].r[2];
    top[c] = hsum(m0, m1, m2, (v4){0});
    bot[c] = (m0 + m1) + m2;
  }
  v4 rr = 1.0 / hsum(bot[0], bot[1], bot[2], bot[3]);
  for (int c = 0; c < NC; c++) {
    st3(out + c*K, top[c]*rr[c]);
    st3(out + (c+NC)*K, bot[c]*rr[c]);
  }
}
