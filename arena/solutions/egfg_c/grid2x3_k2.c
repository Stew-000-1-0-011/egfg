/* egfg program for grid2x3_k2 (cost model: 168 operations, overhead 64) */
#include <stddef.h>

static double t0[8];
static double t1[4];
static double t2[8];
static double t3[4];
static double t4[4];
static double t5[8];
static double t6[4];
static double t7[2];
static double t8[4];
static double t9[4];
static double t10[4];
static double t11[4];
static double t12[2];
static double t13[2];
static double t14[2];
static double t15[2];
static double t16[2];

void infer(const double *const *tables, double *out) {
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i2 = 0; i2 < 2; i2++) {
            for (int i5 = 0; i5 < 2; i5++) {
                t0[i1*4 + i2*2 + i5] = tables[2][i1*2 + i2] * tables[4][i2*2 + i5];
            }
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i5 = 0; i5 < 2; i5++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 2; i2++) {
                acc += t0[i1*4 + i2*2 + i5];
            }
            t1[i1*2 + i5] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i4 = 0; i4 < 2; i4++) {
            for (int i5 = 0; i5 < 2; i5++) {
                t2[i1*4 + i4*2 + i5] = tables[3][i1*2 + i4] * tables[6][i4*2 + i5];
            }
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i4 = 0; i4 < 2; i4++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 2; i5++) {
                acc += t2[i1*4 + i4*2 + i5] * t1[i1*2 + i5];
            }
            t3[i1*2 + i4] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i4 = 0; i4 < 2; i4++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 2; i1++) {
                acc += tables[0][i0*2 + i1] * t3[i1*2 + i4];
            }
            t4[i0*2 + i4] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i3 = 0; i3 < 2; i3++) {
            for (int i4 = 0; i4 < 2; i4++) {
                t5[i0*4 + i3*2 + i4] = tables[1][i0*2 + i3] * tables[5][i3*2 + i4];
            }
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i3 = 0; i3 < 2; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 2; i4++) {
                acc += t5[i0*4 + i3*2 + i4] * t4[i0*2 + i4];
            }
            t6[i0*2 + i3] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 2; i3++) {
            acc += t6[i0*2 + i3];
        }
        t7[i0] = acc;
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i4 = 0; i4 < 2; i4++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 2; i3++) {
                acc += t5[i0*4 + i3*2 + i4];
            }
            t8[i0*2 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i4 = 0; i4 < 2; i4++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 2; i0++) {
                acc += tables[0][i0*2 + i1] * t8[i0*2 + i4];
            }
            t9[i1*2 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i5 = 0; i5 < 2; i5++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 2; i4++) {
                acc += t2[i1*4 + i4*2 + i5] * t9[i1*2 + i4];
            }
            t10[i1*2 + i5] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i2 = 0; i2 < 2; i2++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 2; i5++) {
                acc += t0[i1*4 + i2*2 + i5] * t10[i1*2 + i5];
            }
            t11[i1*2 + i2] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 2; i2++) {
            acc += t11[i1*2 + i2];
        }
        t12[i1] = acc;
    }
    for (int i2 = 0; i2 < 2; i2++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 2; i1++) {
            acc += t11[i1*2 + i2];
        }
        t13[i2] = acc;
    }
    for (int i3 = 0; i3 < 2; i3++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 2; i0++) {
            acc += t6[i0*2 + i3];
        }
        t14[i3] = acc;
    }
    for (int i4 = 0; i4 < 2; i4++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 2; i0++) {
            acc += t4[i0*2 + i4] * t8[i0*2 + i4];
        }
        t15[i4] = acc;
    }
    for (int i5 = 0; i5 < 2; i5++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 2; i1++) {
            acc += t10[i1*2 + i5] * t1[i1*2 + i5];
        }
        t16[i5] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 2; j++) z += t7[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 2; j++) out[0 + j] = t7[j] * iz;
    for (int j = 0; j < 2; j++) out[2 + j] = t12[j] * iz;
    for (int j = 0; j < 2; j++) out[4 + j] = t13[j] * iz;
    for (int j = 0; j < 2; j++) out[6 + j] = t14[j] * iz;
    for (int j = 0; j < 2; j++) out[8 + j] = t15[j] * iz;
    for (int j = 0; j < 2; j++) out[10 + j] = t16[j] * iz;
}
