#pragma GCC optimize("fp-contract=fast")
#include <string.h>
/* 2xNC ladder: top x_c, bottom x_{c+NC}. Treat column c as a supernode (t,b) with unary V_c,
   chain forward/backward with separable transitions H_c (top) and B_c (bottom). */
#define K 2
#define NC 3
static const int HI[NC-1] = {0,2};
static const int BI[NC-1] = {5,6};
static const int VI[NC] = {1,3,4};
#define KK (K*K)
void infer(const double *const *tables, double *out) {
  double al[NC][KK], be[NC][KK];
  memcpy(al[0], tables[VI[0]], sizeof(double)*KK);
  for (int s = 0; s < KK; s++) be[NC-1][s] = 1.0;
  for (int st = 0; st < NC-1; st++) {
    /* forward step c = st */
    {
      int c = st;
      const double *H = tables[HI[c]], *B = tables[BI[c]], *V = tables[VI[c+1]];
      double tau[KK];
      for (int t2 = 0; t2 < K; t2++) for (int b = 0; b < K; b++) {
        double s = 0; for (int t = 0; t < K; t++) s += H[t*K+t2]*al[c][t*K+b]; tau[t2*K+b] = s; }
      for (int t2 = 0; t2 < K; t2++) for (int b2 = 0; b2 < K; b2++) {
        double s = 0; for (int b = 0; b < K; b++) s += tau[t2*K+b]*B[b*K+b2];
        al[c+1][t2*K+b2] = s * V[t2*K+b2]; }
    }
    /* backward step c = NC-2-st: be[c] from be[c+1] */
    {
      int c = NC-2-st;
      const double *H = tables[HI[c]], *B = tables[BI[c]], *V = tables[VI[c+1]];
      double g[KK], tau[KK];
      for (int s = 0; s < KK; s++) g[s] = V[s]*be[c+1][s];
      /* tau(t2,b) = sum_b2 B[b][b2] g(t2,b2) */
      for (int t2 = 0; t2 < K; t2++) for (int b = 0; b < K; b++) {
        double s = 0; for (int b2 = 0; b2 < K; b2++) s += B[b*K+b2]*g[t2*K+b2]; tau[t2*K+b] = s; }
      for (int t = 0; t < K; t++) for (int b = 0; b < K; b++) {
        double s = 0; for (int t2 = 0; t2 < K; t2++) s += H[t*K+t2]*tau[t2*K+b]; be[c][t*K+b] = s; }
    }
  }
  /* every supernode belief sums to Z: take it from the middle column */
  double z = 0;
  for (int s = 0; s < KK; s++) z += al[NC/2][s]*be[NC/2][s];
  const double r = 1.0/z;
  for (int c = 0; c < NC; c++) {
    double mu[KK];
    for (int s = 0; s < KK; s++) mu[s] = al[c][s]*be[c][s];
    for (int t = 0; t < K; t++) { double s = 0; for (int b = 0; b < K; b++) s += mu[t*K+b]; out[c*K+t] = s*r; }
    for (int b = 0; b < K; b++) { double s = 0; for (int t = 0; t < K; t++) s += mu[t*K+b]; out[(c+NC)*K+b] = s*r; }
  }
}
