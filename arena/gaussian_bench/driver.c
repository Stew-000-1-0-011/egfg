/* Runs a generated filter (gccodegen) on a problem: prints the results of the first Tc steps,
 * then the time per step (median over 7 runs of the whole sequence). */
#define _POSIX_C_SOURCE 199309L
#include <time.h>

#include "io.h"

void egfg_setup(const double *const *params);
void egfg_head(const double *const *ys, double *out);
void egfg_step(const double *const *ys, double *out);

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static double run_all(const Problem *P, double *out, long reps) {
    double sink = 0;
    for (long r = 0; r < reps; r++) {
        egfg_head((const double *const *)P->ys, out);
        for (int t = 1; t < P->T; t++) egfg_step((const double *const *)(P->ys + t * P->nobs), out);
        sink += out[0];
        __asm__ volatile("" : : "g"(out) : "memory");
    }
    return sink;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    Problem P = read_problem(argv[1]);
    const double **params = malloc(sizeof(double *) * P.nf);
    for (int i = 0; i < P.nf; i++) params[i] = P.f[i].data;
    egfg_setup(params);
    double *out = malloc(sizeof(double) * P.out_size);
    for (int t = 0; t < P.Tc; t++) {
        if (t == 0) egfg_head((const double *const *)P.ys, out);
        else egfg_step((const double *const *)(P.ys + t * P.nobs), out);
        for (int k = 0; k < P.out_size; k++) printf("%.17g ", out[k]);
    }
    printf("\n");
    long reps = 1;
    for (;;) {
        double t0 = now();
        run_all(&P, out, reps);
        if (now() - t0 >= 0.01) break;
        reps *= 2;
    }
    double s[7];
    for (int k = 0; k < 7; k++) {
        double t0 = now();
        run_all(&P, out, reps);
        s[k] = (now() - t0) / (reps * (double)P.T);
    }
    qsort(s, 7, sizeof(double), cmp);
    printf("%.6e\n", s[3]);
    return 0;
}
