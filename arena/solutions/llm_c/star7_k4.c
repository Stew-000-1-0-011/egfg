#pragma GCC optimize("fp-contract=fast")
#include <string.h>
#define K 4
#define ldl ld
#define st3 st
typedef double vK __attribute__((vector_size(32)));
typedef long long iK __attribute__((vector_size(32)));
static inline vK ld(const double *p) { vK v; memcpy(&v, p, sizeof v); return v; }
static inline void st(double *p, vK v) { memcpy(p, &v, sizeof v); }
static inline vK bc(double x) { return (vK){0} + x; }

/* returns (sum p0, sum p1, sum p2, sum p3) */
static inline vK hsum(vK p0, vK p1, vK p2, vK p3) {
  vK u0 = __builtin_shuffle(p0, p1, (iK){0,4,2,6}) + __builtin_shuffle(p0, p1, (iK){1,5,3,7});
  vK u1 = __builtin_shuffle(p2, p3, (iK){0,4,2,6}) + __builtin_shuffle(p2, p3, (iK){1,5,3,7});
  return __builtin_shuffle(u0, u1, (iK){0,1,4,5}) + __builtin_shuffle(u0, u1, (iK){2,3,6,7});
}
#define HS(p) hsum(p[0], p[1], p[2], p[3])
static inline vK vm(const double *T, vK x) {
  return (x[0]*ld(T) + x[1]*ld(T+K)) + (x[2]*ld(T+2*K) + x[3]*ld(T+3*K));
}
static inline vK cs(const double *T) { return (ld(T) + ld(T+K)) + (ld(T+2*K) + ld(T+3*K)); }
static inline vK rs(const double *T) { return hsum(ld(T), ld(T+K), ld(T+2*K), ld(T+3*K)); }
static inline vK mv(const double *T, vK x) { return hsum(ld(T)*x, ld(T+K)*x, ld(T+2*K)*x, ld(T+3*K)*x); }

void infer(const double *const *tables, double *out) {
  vK m6 = rs(tables[5]);
  vK m5 = rs(tables[4]);
  vK m4 = rs(tables[3]);
  vK m3 = rs(tables[2]);
  vK m2 = rs(tables[1]);
  vK m1 = rs(tables[0]);
  vK u0 = m1 * m2 * m3 * m4 * m5 * m6;
  const double rz = 1.0 / hsum(u0, (vK){0}, (vK){0}, (vK){0})[0];
  vK pre0_0 = bc(1.0);
  vK pre0_1 = pre0_0 * m1;
  vK pre0_2 = pre0_1 * m2;
  vK pre0_3 = pre0_2 * m3;
  vK pre0_4 = pre0_3 * m4;
  vK pre0_5 = pre0_4 * m5;
  vK suf0_5 = m6;
  vK suf0_4 = suf0_5 * m5;
  vK suf0_3 = suf0_4 * m4;
  vK suf0_2 = suf0_3 * m3;
  vK suf0_1 = suf0_2 * m2;
  vK o1 = vm(tables[0], pre0_0 * suf0_1);
  vK o2 = vm(tables[1], pre0_1 * suf0_2);
  vK o3 = vm(tables[2], pre0_2 * suf0_3);
  vK o4 = vm(tables[3], pre0_3 * suf0_4);
  vK o5 = vm(tables[4], pre0_4 * suf0_5);
  vK o6 = vm(tables[5], pre0_5);
  st3(out + 0, (u0) * rz);
  st3(out + 4, (o1) * rz);
  st3(out + 8, (o2) * rz);
  st3(out + 12, (o3) * rz);
  st3(out + 16, (o4) * rz);
  st3(out + 20, (o5) * rz);
  st3(out + 24, (o6) * rz);
}
