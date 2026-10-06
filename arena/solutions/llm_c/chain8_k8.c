#pragma GCC optimize("fp-contract=fast")
#include <string.h>
#define K 8
#define ldl ld
#define st3 st
typedef double vK __attribute__((vector_size(64)));
typedef long long iK __attribute__((vector_size(64)));
static inline vK ld(const double *p) { vK v; memcpy(&v, p, sizeof v); return v; }
static inline void st(double *p, vK v) { memcpy(p, &v, sizeof v); }
static inline vK bc(double x) { return (vK){0} + x; }

static inline vK hs2(vK a, vK b) {
  return __builtin_shuffle(a, b, (iK){0,8,2,10,4,12,6,14}) + __builtin_shuffle(a, b, (iK){1,9,3,11,5,13,7,15});
}
static inline vK hs4(vK a, vK b) {
  return __builtin_shuffle(a, b, (iK){0,1,8,9,4,5,12,13}) + __builtin_shuffle(a, b, (iK){2,3,10,11,6,7,14,15});
}
static inline vK hsum(vK p0, vK p1, vK p2, vK p3, vK p4, vK p5, vK p6, vK p7) {
  vK a = hs4(hs2(p0, p1), hs2(p2, p3)), b = hs4(hs2(p4, p5), hs2(p6, p7));
  return __builtin_shuffle(a, b, (iK){0,1,2,3,8,9,10,11}) + __builtin_shuffle(a, b, (iK){4,5,6,7,12,13,14,15});
}
#define HS(p) hsum(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
static inline vK vm(const double *T, vK x) {
  return ((x[0]*ld(T) + x[1]*ld(T+K)) + (x[2]*ld(T+2*K) + x[3]*ld(T+3*K)))
       + ((x[4]*ld(T+4*K) + x[5]*ld(T+5*K)) + (x[6]*ld(T+6*K) + x[7]*ld(T+7*K)));
}
static inline vK cs(const double *T) {
  return ((ld(T) + ld(T+K)) + (ld(T+2*K) + ld(T+3*K))) + ((ld(T+4*K) + ld(T+5*K)) + (ld(T+6*K) + ld(T+7*K)));
}
static inline vK rs(const double *T) {
  return hsum(ld(T), ld(T+K), ld(T+2*K), ld(T+3*K), ld(T+4*K), ld(T+5*K), ld(T+6*K), ld(T+7*K));
}
static inline vK mv(const double *T, vK x) {
  return hsum(ld(T)*x, ld(T+K)*x, ld(T+2*K)*x, ld(T+3*K)*x, ld(T+4*K)*x, ld(T+5*K)*x, ld(T+6*K)*x, ld(T+7*K)*x);
}

void infer(const double *const *tables, double *out) {
  vK u7 = ldl(tables[14]);
  vK m7 = mv(tables[6], u7);
  vK u6 = ldl(tables[13]) * m7;
  vK m6 = mv(tables[5], u6);
  vK u5 = ldl(tables[7]);
  vK m5 = vm(tables[0], u5);
  vK u4 = ldl(tables[12]) * m6;
  vK m4 = mv(tables[4], u4);
  vK u3 = ldl(tables[8]) * m5;
  vK m3 = vm(tables[1], u3);
  vK u2 = ldl(tables[11]) * m4;
  vK m2 = mv(tables[3], u2);
  vK u1 = ldl(tables[9]) * m3;
  vK m1 = vm(tables[2], u1);
  vK u0 = ldl(tables[10]) * m1 * m2;
  const double rz = 1.0 / hsum(u0, (vK){0}, (vK){0}, (vK){0}, (vK){0}, (vK){0}, (vK){0}, (vK){0})[0];
  vK o1 = mv(tables[2], ldl(tables[10]) * m2);
  vK o2 = vm(tables[3], ldl(tables[10]) * m1);
  vK o3 = mv(tables[1], ldl(tables[9]) * o1);
  vK o4 = vm(tables[4], ldl(tables[11]) * o2);
  vK o5 = mv(tables[0], ldl(tables[8]) * o3);
  vK o6 = vm(tables[5], ldl(tables[12]) * o4);
  vK o7 = vm(tables[6], ldl(tables[13]) * o6);
  st3(out + 24, (u0) * rz);
  st3(out + 16, (u1 * o1) * rz);
  st3(out + 32, (u2 * o2) * rz);
  st3(out + 8, (u3 * o3) * rz);
  st3(out + 40, (u4 * o4) * rz);
  st3(out + 0, (u5 * o5) * rz);
  st3(out + 48, (u6 * o6) * rz);
  st3(out + 56, (u7 * o7) * rz);
}
