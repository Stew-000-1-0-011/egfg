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
  vK m8 = rs(tables[6]);
  vK m7 = rs(tables[7]);
  vK m6 = rs(tables[5]);
  vK u5 = m8;
  vK m5 = mv(tables[4], u5);
  vK m4 = rs(tables[3]);
  vK u3 = m7;
  vK m3 = vm(tables[0], u3);
  vK u2 = m5 * m6;
  vK m2 = mv(tables[2], u2);
  vK u1 = m3 * m4;
  vK m1 = vm(tables[1], u1);
  vK u0 = m1 * m2;
  const double rz = 1.0 / hsum(u0, (vK){0}, (vK){0}, (vK){0})[0];
  vK o1 = mv(tables[1], m2);
  vK o2 = vm(tables[2], m1);
  vK o3 = mv(tables[0], o1 * m4);
  vK o4 = vm(tables[3], o1 * m3);
  vK o5 = vm(tables[4], o2 * m6);
  vK o6 = vm(tables[5], o2 * m5);
  vK o7 = vm(tables[7], o3);
  vK o8 = vm(tables[6], o5);
  st3(out + 8, (u0) * rz);
  st3(out + 4, (u1 * o1) * rz);
  st3(out + 12, (u2 * o2) * rz);
  st3(out + 0, (u3 * o3) * rz);
  st3(out + 16, (o4) * rz);
  st3(out + 20, (u5 * o5) * rz);
  st3(out + 24, (o6) * rz);
  st3(out + 32, (o7) * rz);
  st3(out + 28, (o8) * rz);
}
