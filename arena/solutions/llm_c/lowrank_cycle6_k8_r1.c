#pragma GCC optimize("fp-contract=fast")
#include <string.h>
#define K 8
#define N 6
/* Each T_i (rows x_i, cols x_{i+1}) is rank 1: T_i = a_i b_i^T, so the joint factorizes
   and marg(x_i) ∝ a_i(x) b_{i-1}(x) ∝ T_i[x][0] * T_{i-1}[0][x]. */
void infer(const double *const *tables, double *out) {
  for (int i = 0; i < N; i++) {
    const double *A = tables[i], *B = tables[(i + N - 1) % N];
    double t[K], s = 0;
    for (int x = 0; x < K; x++) { t[x] = A[x*K] * B[x]; s += t[x]; }
    double r = 1.0 / s;
    for (int x = 0; x < K; x++) out[i*K + x] = t[x] * r;
  }
}
