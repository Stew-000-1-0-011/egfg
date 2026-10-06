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
  vK u14 = ldl(tables[28]);
  vK m14 = mv(tables[13], u14);
  vK u13 = ldl(tables[14]);
  vK m13 = vm(tables[0], u13);
  vK u12 = ldl(tables[27]) * m14;
  vK m12 = mv(tables[12], u12);
  vK u11 = ldl(tables[15]) * m13;
  vK m11 = vm(tables[1], u11);
  vK u10 = ldl(tables[26]) * m12;
  vK m10 = mv(tables[11], u10);
  vK u9 = ldl(tables[16]) * m11;
  vK m9 = vm(tables[2], u9);
  vK u8 = ldl(tables[25]) * m10;
  vK m8 = mv(tables[10], u8);
  vK u7 = ldl(tables[17]) * m9;
  vK m7 = vm(tables[3], u7);
  vK u6 = ldl(tables[24]) * m8;
  vK m6 = mv(tables[9], u6);
  vK u5 = ldl(tables[18]) * m7;
  vK m5 = vm(tables[4], u5);
  vK u4 = ldl(tables[23]) * m6;
  vK m4 = mv(tables[8], u4);
  vK u3 = ldl(tables[19]) * m5;
  vK m3 = vm(tables[5], u3);
  vK u2 = ldl(tables[22]) * m4;
  vK m2 = mv(tables[7], u2);
  vK u1 = ldl(tables[20]) * m3;
  vK m1 = vm(tables[6], u1);
  vK u0 = ldl(tables[21]) * m1 * m2;
  const double rz = 1.0 / hsum(u0, (vK){0}, (vK){0}, (vK){0})[0];
  vK o1 = mv(tables[6], ldl(tables[21]) * m2);
  vK o2 = vm(tables[7], ldl(tables[21]) * m1);
  vK o3 = mv(tables[5], ldl(tables[20]) * o1);
  vK o4 = vm(tables[8], ldl(tables[22]) * o2);
  vK o5 = mv(tables[4], ldl(tables[19]) * o3);
  vK o6 = vm(tables[9], ldl(tables[23]) * o4);
  vK o7 = mv(tables[3], ldl(tables[18]) * o5);
  vK o8 = vm(tables[10], ldl(tables[24]) * o6);
  vK o9 = mv(tables[2], ldl(tables[17]) * o7);
  vK o10 = vm(tables[11], ldl(tables[25]) * o8);
  vK o11 = mv(tables[1], ldl(tables[16]) * o9);
  vK o12 = vm(tables[12], ldl(tables[26]) * o10);
  vK o13 = mv(tables[0], ldl(tables[15]) * o11);
  vK o14 = vm(tables[13], ldl(tables[27]) * o12);
  st3(out + 48, (u0) * rz);
  st3(out + 44, (u1 * o1) * rz);
  st3(out + 52, (u2 * o2) * rz);
  st3(out + 40, (u3 * o3) * rz);
  st3(out + 56, (u4 * o4) * rz);
  st3(out + 36, (u5 * o5) * rz);
  st3(out + 8, (u6 * o6) * rz);
  st3(out + 32, (u7 * o7) * rz);
  st3(out + 12, (u8 * o8) * rz);
  st3(out + 28, (u9 * o9) * rz);
  st3(out + 16, (u10 * o10) * rz);
  st3(out + 4, (u11 * o11) * rz);
  st3(out + 20, (u12 * o12) * rz);
  st3(out + 0, (u13 * o13) * rz);
  st3(out + 24, (u14 * o14) * rz);
}
