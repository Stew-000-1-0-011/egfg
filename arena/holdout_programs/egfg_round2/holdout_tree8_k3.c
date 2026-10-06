/* egfg program for holdout_tree8_k3 (cost model: 246 operations, overhead 64) */
#include <stddef.h>
#include <math.h>

static double t0[3];
static double t1[3];
static double t2[3];
static double t3[3];
static double t4[3];
static double t5[3];
static double t6[3];
static double t7[3];
static double t8[3];
static double t9[3];
static double t10[3];
static double t11[3];
static double t12[3];
static double t13[3];
static double t14[3];
static double t15[3];
static double t16[3];
static double t17[3];
static double t18[3];
static double t19[3];
static double t20[3];
static double t21[3];
static double t22[3];
static double t23[3];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            acc += tables[0][i0*3 + i1];
        }
        t0[i0] = acc;
    }
    for (int i3 = 0; i3 < 3; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            acc += tables[3][i3*3 + i4];
        }
        t1[i3] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += tables[2][i2*3 + i3] * t1[i3];
        }
        t2[i2] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += tables[5][i2*3 + i6];
        }
        t3[i2] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        t4[i2] = t3[i2] * t2[i2];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            acc += tables[1][i0*3 + i2] * t4[i2];
        }
        t5[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t6[i0] = t5[i0] * t0[i0];
    }
    for (int i5 = 0; i5 < 3; i5++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 3; i7++) {
            acc += tables[6][i5*3 + i7];
        }
        t7[i5] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            acc += tables[4][i0*3 + i5] * t7[i5];
        }
        t8[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t9[i0] = t8[i0] * t6[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t10[i0] = t5[i0] * t8[i0];
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[0][i0*3 + i1] * t10[i0];
        }
        t11[i1] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t12[i0] = t8[i0] * t0[i0];
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[1][i0*3 + i2] * t12[i0];
        }
        t13[i2] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        t14[i2] = t13[i2] * t4[i2];
    }
    for (int i2 = 0; i2 < 3; i2++) {
        t15[i2] = t3[i2] * t13[i2];
    }
    for (int i3 = 0; i3 < 3; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            acc += tables[2][i2*3 + i3] * t15[i2];
        }
        t16[i3] = acc;
    }
    for (int i3 = 0; i3 < 3; i3++) {
        t17[i3] = t16[i3] * t1[i3];
    }
    for (int i4 = 0; i4 < 3; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += tables[3][i3*3 + i4] * t16[i3];
        }
        t18[i4] = acc;
    }
    for (int i5 = 0; i5 < 3; i5++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[4][i0*3 + i5] * t6[i0];
        }
        t19[i5] = acc;
    }
    for (int i5 = 0; i5 < 3; i5++) {
        t20[i5] = t19[i5] * t7[i5];
    }
    for (int i2 = 0; i2 < 3; i2++) {
        t21[i2] = t13[i2] * t2[i2];
    }
    for (int i6 = 0; i6 < 3; i6++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            acc += t21[i2] * tables[5][i2*3 + i6];
        }
        t22[i6] = acc;
    }
    for (int i7 = 0; i7 < 3; i7++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            acc += tables[6][i5*3 + i7] * t19[i5];
        }
        t23[i7] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 3; j++) z += t9[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 3; j++) out[0 + j] = t9[j] * iz;
    for (int j = 0; j < 3; j++) out[3 + j] = t11[j] * iz;
    for (int j = 0; j < 3; j++) out[6 + j] = t14[j] * iz;
    for (int j = 0; j < 3; j++) out[9 + j] = t17[j] * iz;
    for (int j = 0; j < 3; j++) out[12 + j] = t18[j] * iz;
    for (int j = 0; j < 3; j++) out[15 + j] = t20[j] * iz;
    for (int j = 0; j < 3; j++) out[18 + j] = t22[j] * iz;
    for (int j = 0; j < 3; j++) out[21 + j] = t23[j] * iz;
}
