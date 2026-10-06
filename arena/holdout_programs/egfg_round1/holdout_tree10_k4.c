/* egfg program for holdout_tree10_k4 (cost model: 552 operations, overhead 64) */
#include <stddef.h>

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
static double t27[4];
static double t28[4];
static double t29[4];
static double t30[4];
static double t31[4];

void infer(const double *const *tables, double *out) {
    for (int i2 = 0; i2 < 4; i2++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 4; i4++) {
            acc += tables[3][i2*4 + i4];
        }
        t0[i2] = acc;
    }
    for (int i2 = 0; i2 < 4; i2++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 4; i5++) {
            acc += tables[4][i2*4 + i5];
        }
        t1[i2] = acc;
    }
    for (int i2 = 0; i2 < 4; i2++) {
        t2[i2] = t1[i2] * t0[i2];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 4; i2++) {
            acc += tables[1][i0*4 + i2] * t2[i2];
        }
        t3[i0] = acc;
    }
    for (int i3 = 0; i3 < 4; i3++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 4; i6++) {
            acc += tables[5][i3*4 + i6];
        }
        t4[i3] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 4; i9++) {
            acc += tables[8][i7*4 + i9];
        }
        t5[i7] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 4; i8++) {
            acc += tables[7][i7*4 + i8];
        }
        t6[i7] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        t7[i7] = t6[i7] * t5[i7];
    }
    for (int i3 = 0; i3 < 4; i3++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 4; i7++) {
            acc += tables[6][i3*4 + i7] * t7[i7];
        }
        t8[i3] = acc;
    }
    for (int i3 = 0; i3 < 4; i3++) {
        t9[i3] = t8[i3] * t4[i3];
    }
    for (int i1 = 0; i1 < 4; i1++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            acc += t9[i3] * tables[2][i1*4 + i3];
        }
        t10[i1] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            acc += tables[0][i0*4 + i1] * t10[i1];
        }
        t11[i0] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        t12[i0] = t11[i0] * t3[i0];
    }
    for (int i1 = 0; i1 < 4; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += tables[0][i0*4 + i1] * t3[i0];
        }
        t13[i1] = acc;
    }
    for (int i1 = 0; i1 < 4; i1++) {
        t14[i1] = t10[i1] * t13[i1];
    }
    for (int i2 = 0; i2 < 4; i2++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += tables[1][i0*4 + i2] * t11[i0];
        }
        t15[i2] = acc;
    }
    for (int i2 = 0; i2 < 4; i2++) {
        t16[i2] = t15[i2] * t2[i2];
    }
    for (int i3 = 0; i3 < 4; i3++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            acc += tables[2][i1*4 + i3] * t13[i1];
        }
        t17[i3] = acc;
    }
    for (int i3 = 0; i3 < 4; i3++) {
        t18[i3] = t17[i3] * t9[i3];
    }
    for (int i2 = 0; i2 < 4; i2++) {
        t19[i2] = t15[i2] * t1[i2];
    }
    for (int i4 = 0; i4 < 4; i4++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 4; i2++) {
            acc += tables[3][i2*4 + i4] * t19[i2];
        }
        t20[i4] = acc;
    }
    for (int i2 = 0; i2 < 4; i2++) {
        t21[i2] = t15[i2] * t0[i2];
    }
    for (int i5 = 0; i5 < 4; i5++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 4; i2++) {
            acc += tables[4][i2*4 + i5] * t21[i2];
        }
        t22[i5] = acc;
    }
    for (int i3 = 0; i3 < 4; i3++) {
        t23[i3] = t8[i3] * t17[i3];
    }
    for (int i6 = 0; i6 < 4; i6++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            acc += tables[5][i3*4 + i6] * t23[i3];
        }
        t24[i6] = acc;
    }
    for (int i3 = 0; i3 < 4; i3++) {
        t25[i3] = t4[i3] * t17[i3];
    }
    for (int i7 = 0; i7 < 4; i7++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            acc += tables[6][i3*4 + i7] * t25[i3];
        }
        t26[i7] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        t27[i7] = t26[i7] * t7[i7];
    }
    for (int i7 = 0; i7 < 4; i7++) {
        t28[i7] = t5[i7] * t26[i7];
    }
    for (int i8 = 0; i8 < 4; i8++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 4; i7++) {
            acc += tables[7][i7*4 + i8] * t28[i7];
        }
        t29[i8] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        t30[i7] = t6[i7] * t26[i7];
    }
    for (int i9 = 0; i9 < 4; i9++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 4; i7++) {
            acc += t30[i7] * tables[8][i7*4 + i9];
        }
        t31[i9] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 4; j++) z += t12[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 4; j++) out[0 + j] = t12[j] * iz;
    for (int j = 0; j < 4; j++) out[4 + j] = t14[j] * iz;
    for (int j = 0; j < 4; j++) out[8 + j] = t16[j] * iz;
    for (int j = 0; j < 4; j++) out[12 + j] = t18[j] * iz;
    for (int j = 0; j < 4; j++) out[16 + j] = t20[j] * iz;
    for (int j = 0; j < 4; j++) out[20 + j] = t22[j] * iz;
    for (int j = 0; j < 4; j++) out[24 + j] = t24[j] * iz;
    for (int j = 0; j < 4; j++) out[28 + j] = t27[j] * iz;
    for (int j = 0; j < 4; j++) out[32 + j] = t29[j] * iz;
    for (int j = 0; j < 4; j++) out[36 + j] = t31[j] * iz;
}
