/* egfg program for star7_k4 (cost model: 348 operations, overhead 0) */
#include <stddef.h>
#include <math.h>

static double t0[4];
static double t1[4];
static double t2[4];
static double t3[4];
static double t4[4];
static double t5[4];
static double t6[4];
static double t7[4];
static double t8[4];
static double t9[4];
static double t10[4];
static double t11[4];
static double t12[4];
static double t13[4];
static double t14[4];
static double t15[4];
static double t16[4];
static double t17[4];
static double t18[4];
static double t19[4];
static double t20[4];
static double t21[4];
static double t22[4];
static double t23[4];
static double t24[4];
static double t25[4];
static double t26[4];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 4; i5++) {
            acc += tables[4][i0*4 + i5];
        }
        t0[i0] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 4; i4++) {
            acc += tables[3][i0*4 + i4];
        }
        t1[i0] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t2[i0] = t1[i0] * t0[i0];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 4; i6++) {
            acc += tables[5][i0*4 + i6];
        }
        t3[i0] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            acc += tables[2][i0*4 + i3];
        }
        t4[i0] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t5[i0] = t4[i0] * t3[i0];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t6[i0] = t5[i0] * t2[i0];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 4; i2++) {
            acc += tables[1][i0*4 + i2];
        }
        t7[i0] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            acc += tables[0][i0*4 + i1];
        }
        t8[i0] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t9[i0] = t8[i0] * t7[i0];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t10[i0] = t9[i0] * t6[i0];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t11[i0] = t7[i0] * t6[i0];
    }
    for (int i1 = 0; i1 < 4; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += tables[0][i0*4 + i1] * t11[i0];
        }
        t12[i1] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t13[i0] = t8[i0] * t6[i0];
    }
    for (int i2 = 0; i2 < 4; i2++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += tables[1][i0*4 + i2] * t13[i0];
        }
        t14[i2] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t15[i0] = t2[i0] * t9[i0];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t16[i0] = t3[i0] * t15[i0];
    }
    for (int i3 = 0; i3 < 4; i3++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += tables[2][i0*4 + i3] * t16[i0];
        }
        t17[i3] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t18[i0] = t5[i0] * t9[i0];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t19[i0] = t0[i0] * t18[i0];
    }
    for (int i4 = 0; i4 < 4; i4++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += tables[3][i0*4 + i4] * t19[i0];
        }
        t20[i4] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t21[i0] = t4[i0] * t1[i0];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t22[i0] = t21[i0] * t9[i0];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t23[i0] = t3[i0] * t22[i0];
    }
    for (int i5 = 0; i5 < 4; i5++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += tables[4][i0*4 + i5] * t23[i0];
        }
        t24[i5] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t25[i0] = t0[i0] * t22[i0];
    }
    for (int i6 = 0; i6 < 4; i6++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += tables[5][i0*4 + i6] * t25[i0];
        }
        t26[i6] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 4; j++) z += t10[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 4; j++) out[0 + j] = t10[j] * iz;
    for (int j = 0; j < 4; j++) out[4 + j] = t12[j] * iz;
    for (int j = 0; j < 4; j++) out[8 + j] = t14[j] * iz;
    for (int j = 0; j < 4; j++) out[12 + j] = t17[j] * iz;
    for (int j = 0; j < 4; j++) out[16 + j] = t20[j] * iz;
    for (int j = 0; j < 4; j++) out[20 + j] = t24[j] * iz;
    for (int j = 0; j < 4; j++) out[24 + j] = t26[j] * iz;
}
