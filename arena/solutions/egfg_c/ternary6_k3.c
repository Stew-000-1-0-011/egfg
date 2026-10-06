/* egfg program for ternary6_k3 (cost model: 450 operations, overhead 0) */
#include <stddef.h>
#include <math.h>

static double t0[9];
static double t1[9];
static double t2[9];
static double t3[9];
static double t4[3];
static double t5[3];
static double t6[9];
static double t7[9];
static double t8[9];
static double t9[3];
static double t10[3];
static double t11[9];
static double t12[3];
static double t13[3];

void infer(const double *const *tables, double *out) {
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 3; i5++) {
                acc += tables[3][i3*9 + i4*3 + i5];
            }
            t0[i3*3 + i4] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i3 = 0; i3 < 3; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 3; i4++) {
                acc += tables[2][i2*9 + i3*3 + i4] * t0[i3*3 + i4];
            }
            t1[i2*3 + i3] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i2 = 0; i2 < 3; i2++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 3; i3++) {
                acc += tables[1][i1*9 + i2*3 + i3] * t1[i2*3 + i3];
            }
            t2[i1*3 + i2] = acc;
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 3; i2++) {
                acc += tables[0][i0*9 + i1*3 + i2] * t2[i1*3 + i2];
            }
            t3[i0*3 + i1] = acc;
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            acc += t3[i0*3 + i1];
        }
        t4[i0] = acc;
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += t3[i0*3 + i1];
        }
        t5[i1] = acc;
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i2 = 0; i2 < 3; i2++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 3; i0++) {
                acc += tables[0][i0*9 + i1*3 + i2];
            }
            t6[i1*3 + i2] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i3 = 0; i3 < 3; i3++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 3; i1++) {
                acc += tables[1][i1*9 + i2*3 + i3] * t6[i1*3 + i2];
            }
            t7[i2*3 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i3 = 0; i3 < 3; i3++) {
            t8[i2*3 + i3] = t7[i2*3 + i3] * t1[i2*3 + i3];
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += t8[i2*3 + i3];
        }
        t9[i2] = acc;
    }
    for (int i3 = 0; i3 < 3; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            acc += t8[i2*3 + i3];
        }
        t10[i3] = acc;
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 3; i2++) {
                acc += tables[2][i2*9 + i3*3 + i4] * t7[i2*3 + i3];
            }
            t11[i3*3 + i4] = acc;
        }
    }
    for (int i4 = 0; i4 < 3; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += t11[i3*3 + i4] * t0[i3*3 + i4];
        }
        t12[i4] = acc;
    }
    for (int i5 = 0; i5 < 3; i5++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            for (int i4 = 0; i4 < 3; i4++) {
                acc += tables[3][i3*9 + i4*3 + i5] * t11[i3*3 + i4];
            }
        }
        t13[i5] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 3; j++) z += t4[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 3; j++) out[0 + j] = t4[j] * iz;
    for (int j = 0; j < 3; j++) out[3 + j] = t5[j] * iz;
    for (int j = 0; j < 3; j++) out[6 + j] = t9[j] * iz;
    for (int j = 0; j < 3; j++) out[9 + j] = t10[j] * iz;
    for (int j = 0; j < 3; j++) out[12 + j] = t12[j] * iz;
    for (int j = 0; j < 3; j++) out[15 + j] = t13[j] * iz;
}
