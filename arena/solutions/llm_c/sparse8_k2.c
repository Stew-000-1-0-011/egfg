#pragma GCC optimize("fp-contract=fast")
#include <string.h>
/* Edges: f0(0,1) f1(0,2) f2(1,3) f3(3,4) f4(0,5) f5(2,6) f6(3,7) f7(3,6) f8(5,7).
   x4 is a leaf on x3; the rest is three parallel paths between x0 and x3:
   0-1-3 (f0 f2), 0-2-6-3 (f1 f5 f7^T), 0-5-7-3 (f4 f8 f6^T). */
#define K 2
static inline void sc(const double *t, double r, double *o) { o[0] = t[0]*r; o[1] = t[1]*r; }
/* C = A B */
static inline void mm(const double *A, const double *B, double *C) {
  for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) C[i*K+j] = A[i*K]*B[j] + A[i*K+1]*B[K+j];
}
/* C = A B^T */
static inline void mmT(const double *A, const double *B, double *C) {
  for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) C[i*K+j] = A[i*K]*B[j*K] + A[i*K+1]*B[j*K+1];
}
/* marginal of middle node v of path E1 (a,v) then E2 (v,d) with outside O(a,d):
   t[v] = sum_{a,d} E1[a][v] E2[v][d] O[a][d] */
static inline void mid(const double *E1, const double *E2, const double *O, double *t) {
  for (int v = 0; v < K; v++) {
    double s = 0;
    for (int a = 0; a < K; a++) s += E1[a*K+v]*(E2[v*K]*O[a*K] + E2[v*K+1]*O[a*K+1]);
    t[v] = s;
  }
}
/* same but E2 given transposed: E2T[d][v] */
static inline void midT(const double *E1, const double *E2T, const double *O, double *t) {
  for (int v = 0; v < K; v++) {
    double s = 0;
    for (int a = 0; a < K; a++) s += E1[a*K+v]*(E2T[v]*O[a*K] + E2T[K+v]*O[a*K+1]);
    t[v] = s;
  }
}
void infer(const double *const *tb, double *out) {
  const double *f0=tb[0], *f1=tb[1], *f2=tb[2], *f3=tb[3], *f4=tb[4], *f5=tb[5], *f6=tb[6], *f7=tb[7], *f8=tb[8];
  double u3[K] = { f3[0]+f3[1], f3[2]+f3[3] };
  double M1[4], A2[4], Bq2[4], M2[4], A3[4], Bq3[4], M3[4];
  mm(f0, f2, M1);
  mm(f1, f5, A2); mmT(A2, f7, M2); mmT(f5, f7, Bq2);
  mm(f4, f8, A3); mmT(A3, f6, M3); mmT(f8, f6, Bq3);
  double O1[4], O2[4], O3[4], J[4], c3[K];
  for (int s = 0; s < 4; s++) {
    double u = u3[s & 1];
    O1[s] = M2[s]*M3[s]*u; O2[s] = M1[s]*M3[s]*u; O3[s] = M1[s]*M2[s]*u;
    J[s] = O1[s]*M1[s];
  }
  double t[K];
  t[0] = J[0]+J[1]; t[1] = J[2]+J[3];
  const double rz = 1.0/(t[0] + t[1]);  /* every marginal sums to Z */
  sc(t, rz, out + 0*K);
  t[0] = J[0]+J[2]; t[1] = J[1]+J[3]; sc(t, rz, out + 3*K);
  c3[0] = M1[0]*M2[0]*M3[0] + M1[2]*M2[2]*M3[2];
  c3[1] = M1[1]*M2[1]*M3[1] + M1[3]*M2[3]*M3[3];
  for (int w = 0; w < K; w++) t[w] = f3[w]*c3[0] + f3[K+w]*c3[1];
  sc(t, rz, out + 4*K);
  mid(f0, f2, O1, t); sc(t, rz, out + 1*K);
  mid(f1, Bq2, O2, t); sc(t, rz, out + 2*K);
  midT(A2, f7, O2, t); sc(t, rz, out + 6*K);
  mid(f4, Bq3, O3, t); sc(t, rz, out + 5*K);
  midT(A3, f6, O3, t); sc(t, rz, out + 7*K);
}
