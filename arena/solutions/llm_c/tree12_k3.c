#pragma GCC optimize("fp-contract=fast")
#include <string.h>
#define K 3

static inline void mv(const double *restrict T, const double *restrict x, double *restrict y) {
  /* y[i] = sum_j T[i][j] x[j] */
  for (int i = 0; i < K; i++) { double s = 0; for (int j = 0; j < K; j++) s += T[i*K+j]*x[j]; y[i] = s; }
}
static inline void vm(const double *restrict T, const double *restrict x, double *restrict y) {
  /* y[j] = sum_i x[i] T[i][j] */
  for (int j = 0; j < K; j++) y[j] = x[0]*T[j];
  for (int i = 1; i < K; i++) for (int j = 0; j < K; j++) y[j] += x[i]*T[i*K+j];
}
static inline void rs(const double *restrict T, double *restrict y) {
  for (int i = 0; i < K; i++) { double s = 0; for (int j = 0; j < K; j++) s += T[i*K+j]; y[i] = s; }
}
static inline void cs(const double *restrict T, double *restrict y) {
  for (int j = 0; j < K; j++) y[j] = T[j];
  for (int i = 1; i < K; i++) for (int j = 0; j < K; j++) y[j] += T[i*K+j];
}
static inline void nz(const double *restrict a, const double *restrict b, double r, double *restrict o) {
  for (int i = 0; i < K; i++) o[i] = a[i]*b[i]*r;
}
static inline void nz1(const double *restrict a, double r, double *restrict o) {
  for (int i = 0; i < K; i++) o[i] = a[i]*r;
}
static inline void mul(double *restrict a, const double *restrict b) { for (int i = 0; i < K; i++) a[i] *= b[i]; }
static inline void mul3(const double *restrict a, const double *restrict b, double *restrict o) { for (int i = 0; i < K; i++) o[i] = a[i]*b[i]; }

void infer(const double *const *tables, double *out) {
  double m11[K];
  rs(tables[7], m11);
  double m10[K];
  rs(tables[8], m10);
  double m9[K];
  rs(tables[9], m9);
  double u8[K];
  memcpy(u8, m11, sizeof(double)*K);
  double m8[K];
  mv(tables[5], u8, m8);
  double u7[K];
  memcpy(u7, m10, sizeof(double)*K);
  double m7[K];
  mv(tables[4], u7, m7);
  double m6[K];
  rs(tables[10], m6);
  double m5[K];
  rs(tables[6], m5);
  double u4[K];
  memcpy(u4, m9, sizeof(double)*K);
  double m4[K];
  vm(tables[0], u4, m4);
  double u3[K];
  mul3(m7, m8, u3);
  double m3[K];
  mv(tables[3], u3, m3);
  double u2[K];
  memcpy(u2, m6, sizeof(double)*K);
  double m2[K];
  mv(tables[2], u2, m2);
  double u1[K];
  mul3(m4, m5, u1);
  double m1[K];
  vm(tables[1], u1, m1);
  double u0[K];
  mul3(m1, m2, u0);
  mul(u0, m3);
  double zs = 0; for (int i = 0; i < K; i++) zs += u0[i];
  const double rz = 1.0/zs;
  nz1(u0, rz, out+12);
  double o1[K];
  double x1[K];
  mul3(m2, m3, x1);
  mv(tables[1], x1, o1);
  double o2[K];
  double x2[K];
  mul3(m1, m3, x2);
  vm(tables[2], x2, o2);
  double o3[K];
  double x3[K];
  mul3(m1, m2, x3);
  vm(tables[3], x3, o3);
  nz(u1, o1, rz, out+3);
  double o4[K];
  double x4[K];
  mul3(o1, m5, x4);
  mv(tables[0], x4, o4);
  double o5[K];
  double x5[K];
  mul3(o1, m4, x5);
  vm(tables[6], x5, o5);
  nz(u2, o2, rz, out+15);
  double o6[K];
  vm(tables[10], o2, o6);
  nz(u3, o3, rz, out+18);
  double o7[K];
  double x7[K];
  mul3(o3, m8, x7);
  vm(tables[4], x7, o7);
  double o8[K];
  double x8[K];
  mul3(o3, m7, x8);
  vm(tables[5], x8, o8);
  nz(u4, o4, rz, out+0);
  double o9[K];
  vm(tables[9], o4, o9);
  nz1(o5, rz, out+27);
  nz1(o6, rz, out+9);
  nz(u7, o7, rz, out+21);
  double o10[K];
  vm(tables[8], o7, o10);
  nz(u8, o8, rz, out+24);
  double o11[K];
  vm(tables[7], o8, o11);
  nz1(o9, rz, out+6);
  nz1(o10, rz, out+33);
  nz1(o11, rz, out+30);
}
