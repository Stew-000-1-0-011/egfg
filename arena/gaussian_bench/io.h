/* Reading a linear-Gaussian filtering problem written by run.py (text):
 *   nblocks, then per block its dimension (blocks in name order)
 *   nfactors, then per factor: kind (0 prior, 1 cond, 2 obs), child block (-1 for obs),
 *     nparents, per parent: block and time (-1 or 0), then A_1.., b, Q (row-major)
 *   T, then per step and observation factor the observed vector
 *   Tc: the number of steps whose results are printed for checking */
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int kind, child, np, rows;
    int parent[8], ptime[8];
    double *data; /* A_1.., b, Q */
    double *A[8], *b, *Q;
} Fac;

typedef struct {
    int nb, nf, nobs, T, Tc, out_size;
    int dim[64];
    Fac *f;
    double **ys; /* ys[t * nobs + k] */
} Problem;

static Problem read_problem(const char *path) {
    Problem P;
    FILE *in = fopen(path, "r");
    if (!in || fscanf(in, "%d", &P.nb) != 1) exit(2);
    P.out_size = 0;
    for (int i = 0; i < P.nb; i++) {
        if (fscanf(in, "%d", &P.dim[i]) != 1) exit(2);
        P.out_size += P.dim[i] + P.dim[i] * P.dim[i];
    }
    if (fscanf(in, "%d", &P.nf) != 1) exit(2);
    P.f = (Fac *)calloc(P.nf, sizeof(Fac));
    P.nobs = 0;
    for (int i = 0; i < P.nf; i++) {
        Fac *f = &P.f[i];
        if (fscanf(in, "%d %d %d", &f->kind, &f->child, &f->np) != 3) exit(2);
        int cols = 0;
        for (int k = 0; k < f->np; k++) {
            if (fscanf(in, "%d %d", &f->parent[k], &f->ptime[k]) != 2) exit(2);
            cols += P.dim[f->parent[k]];
        }
        if (fscanf(in, "%d", &f->rows) != 1) exit(2);
        int n = f->rows * cols + f->rows + f->rows * f->rows;
        f->data = (double *)malloc(sizeof(double) * n);
        for (int k = 0; k < n; k++)
            if (fscanf(in, "%lf", &f->data[k]) != 1) exit(2);
        double *p = f->data;
        for (int k = 0; k < f->np; k++) {
            f->A[k] = p;
            p += f->rows * P.dim[f->parent[k]];
        }
        f->b = p;
        f->Q = p + f->rows;
        if (f->kind == 2) P.nobs++;
    }
    if (fscanf(in, "%d", &P.T) != 1) exit(2);
    P.ys = (double **)malloc(sizeof(double *) * P.T * P.nobs);
    for (int t = 0; t < P.T; t++) {
        int k = 0;
        for (int i = 0; i < P.nf; i++) {
            if (P.f[i].kind != 2) continue;
            double *y = (double *)malloc(sizeof(double) * P.f[i].rows);
            for (int r = 0; r < P.f[i].rows; r++)
                if (fscanf(in, "%lf", &y[r]) != 1) exit(2);
            P.ys[t * P.nobs + k++] = y;
        }
    }
    if (fscanf(in, "%d", &P.Tc) != 1) exit(2);
    fclose(in);
    return P;
}
