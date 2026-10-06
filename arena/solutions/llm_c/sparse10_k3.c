#pragma GCC optimize("fp-contract=fast")
#include <string.h>
/* Edges: f0(0,1) f1(0,2) f2(0,3) f3(0,4) f4(1,5) f5(0,6) f6(4,7) f7(3,8) f8(2,9) f9(1,3) f10(3,7).
   Leaves 5,6,8,9 (and 2 after 9) are absorbed into unaries; 1 and 4 are eliminated into
   psi(0,3) and phi(0,7); the remaining triangle (0,3,7) is handled exactly; outside messages
   are pushed back to every eliminated variable. */
#define K 3
static inline void sc(const double *t, double r, double *o) { for (int a = 0; a < K; a++) o[a] = t[a]*r; }
void infer(const double *const *tb, double *out) {
  const double *f0=tb[0], *f1=tb[1], *f2=tb[2], *f3=tb[3], *f4=tb[4], *f5=tb[5], *f6=tb[6],
               *f7=tb[7], *f8=tb[8], *f9=tb[9], *f10=tb[10];
  double m5[K], m6[K], m8[K], m9[K], m2[K], u0[K];
  for (int a = 0; a < K; a++) {
    m5[a] = f4[a*K] + f4[a*K+1] + f4[a*K+2];
    m6[a] = f5[a*K] + f5[a*K+1] + f5[a*K+2];
    m8[a] = f7[a*K] + f7[a*K+1] + f7[a*K+2];
    m9[a] = f8[a*K] + f8[a*K+1] + f8[a*K+2];
  }
  for (int a = 0; a < K; a++) { m2[a] = f1[a*K]*m9[0] + f1[a*K+1]*m9[1] + f1[a*K+2]*m9[2]; u0[a] = m6[a]*m2[a]; }
  const double *u1 = m5, *u3 = m8;
  double psi[K*K], phi[K*K], S[K*K];
  for (int a = 0; a < K; a++) for (int d = 0; d < K; d++) {
    double s = 0, p = 0;
    for (int v = 0; v < K; v++) { s += f0[a*K+v]*u1[v]*f9[v*K+d]; p += f3[a*K+v]*f6[v*K+d]; }
    psi[a*K+d] = s; phi[a*K+d] = p;
  }
  for (int a = 0; a < K; a++) for (int d = 0; d < K; d++) {
    double s = 0; for (int g = 0; g < K; g++) s += f10[d*K+g]*phi[a*K+g]; S[a*K+d] = s;
  }
  double X[K] = {0}, Y[K] = {0}, O1[K*K], W[K*K], O4[K*K];
  for (int a = 0; a < K; a++) for (int d = 0; d < K; d++) {
    double e = f2[a*K+d];
    double F = e*psi[a*K+d]*S[a*K+d];
    X[a] += u3[d]*F; Y[d] += u0[a]*F;
    double uu = u0[a]*u3[d]*e;
    O1[a*K+d] = uu*S[a*K+d];
    W[a*K+d] = uu*psi[a*K+d];
  }
  for (int a = 0; a < K; a++) for (int g = 0; g < K; g++) {
    double s = 0; for (int d = 0; d < K; d++) s += W[a*K+d]*f10[d*K+g]; O4[a*K+g] = s;
  }
  double t[K], B1[K], B2[K], c[K];
  /* x0, x3 */
  for (int a = 0; a < K; a++) t[a] = u0[a]*X[a];
  /* every marginal sums to Z */
  const double rz = 1.0/(t[0] + t[1] + t[2]);
  sc(t, rz, out + 0*K);
  for (int d = 0; d < K; d++) t[d] = u3[d]*Y[d];
  sc(t, rz, out + 3*K);
  /* x7 */
  for (int g = 0; g < K; g++) t[g] = phi[g]*O4[g] + phi[K+g]*O4[K+g] + phi[2*K+g]*O4[2*K+g];
  sc(t, rz, out + 7*K);
  /* x1, x5 */
  for (int v = 0; v < K; v++) {
    double s = 0;
    for (int a = 0; a < K; a++) for (int d = 0; d < K; d++) s += f0[a*K+v]*f9[v*K+d]*O1[a*K+d];
    B1[v] = s; t[v] = s*u1[v];
  }
  sc(t, rz, out + 1*K);
  for (int w = 0; w < K; w++) t[w] = f4[w]*B1[0] + f4[K+w]*B1[1] + f4[2*K+w]*B1[2];
  sc(t, rz, out + 5*K);
  /* x4 */
  for (int w = 0; w < K; w++) {
    double s = 0;
    for (int a = 0; a < K; a++) for (int g = 0; g < K; g++) s += f3[a*K+w]*f6[w*K+g]*O4[a*K+g];
    t[w] = s;
  }
  sc(t, rz, out + 4*K);
  /* x2, x9 */
  for (int a = 0; a < K; a++) c[a] = m6[a]*X[a];
  for (int v = 0; v < K; v++) { B2[v] = f1[v]*c[0] + f1[K+v]*c[1] + f1[2*K+v]*c[2]; t[v] = B2[v]*m9[v]; }
  sc(t, rz, out + 2*K);
  for (int w = 0; w < K; w++) t[w] = f8[w]*B2[0] + f8[K+w]*B2[1] + f8[2*K+w]*B2[2];
  sc(t, rz, out + 9*K);
  /* x6 */
  for (int a = 0; a < K; a++) c[a] = m2[a]*X[a];
  for (int z = 0; z < K; z++) t[z] = f5[z]*c[0] + f5[K+z]*c[1] + f5[2*K+z]*c[2];
  sc(t, rz, out + 6*K);
  /* x8 */
  for (int w = 0; w < K; w++) t[w] = f7[w]*Y[0] + f7[K+w]*Y[1] + f7[2*K+w]*Y[2];
  sc(t, rz, out + 8*K);
}
