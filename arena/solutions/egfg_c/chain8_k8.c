/* egfg program for chain8_k8 (cost model: 1936 operations, overhead 0) */
#include <stddef.h>

static double t0[64];
static double t1[8];
static double t2[8];
static double t3[8];
static double t4[8];
static double t5[8];
static double t6[8];
static double t7[8];
static double t8[8];
static double t9[8];
static double t10[8];
static double t11[8];
static double t12[8];
static double t13[64];
static double t14[8];
static double t15[8];
static double t16[8];
static double t17[8];
static double t18[8];
static double t19[8];
static double t20[8];
static double t21[8];
static double t22[8];
static double t23[8];
static double t24[8];
static double t25[8];
static double t26[8];
static double t27[8];
static double t28[8];
static double t29[8];
static double t30[8];
static double t31[8];
static double t32[8];
static double t33[8];

void infer(const double *const *tables, double *out) {
    for (int i6 = 0; i6 < 8; i6++) {
        for (int i7 = 0; i7 < 8; i7++) {
            t0[i6*8 + i7] = tables[6][i6*8 + i7] * tables[14][i7];
        }
    }
    for (int i6 = 0; i6 < 8; i6++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 8; i7++) {
            acc += t0[i6*8 + i7];
        }
        t1[i6] = acc;
    }
    for (int i6 = 0; i6 < 8; i6++) {
        t2[i6] = tables[13][i6] * t1[i6];
    }
    for (int i5 = 0; i5 < 8; i5++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 8; i6++) {
            acc += tables[5][i5*8 + i6] * t2[i6];
        }
        t3[i5] = acc;
    }
    for (int i5 = 0; i5 < 8; i5++) {
        t4[i5] = tables[12][i5] * t3[i5];
    }
    for (int i4 = 0; i4 < 8; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 8; i5++) {
            acc += tables[4][i4*8 + i5] * t4[i5];
        }
        t5[i4] = acc;
    }
    for (int i4 = 0; i4 < 8; i4++) {
        t6[i4] = tables[11][i4] * t5[i4];
    }
    for (int i3 = 0; i3 < 8; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 8; i4++) {
            acc += tables[3][i3*8 + i4] * t6[i4];
        }
        t7[i3] = acc;
    }
    for (int i3 = 0; i3 < 8; i3++) {
        t8[i3] = tables[10][i3] * t7[i3];
    }
    for (int i2 = 0; i2 < 8; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 8; i3++) {
            acc += tables[2][i2*8 + i3] * t8[i3];
        }
        t9[i2] = acc;
    }
    for (int i2 = 0; i2 < 8; i2++) {
        t10[i2] = tables[9][i2] * t9[i2];
    }
    for (int i1 = 0; i1 < 8; i1++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 8; i2++) {
            acc += tables[1][i1*8 + i2] * t10[i2];
        }
        t11[i1] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        t12[i1] = tables[8][i1] * t11[i1];
    }
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i1 = 0; i1 < 8; i1++) {
            t13[i0*8 + i1] = tables[0][i0*8 + i1] * tables[7][i0];
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 8; i1++) {
            acc += t13[i0*8 + i1] * t12[i1];
        }
        t14[i0] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 8; i0++) {
            acc += t13[i0*8 + i1];
        }
        t15[i1] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        t16[i1] = tables[8][i1] * t15[i1];
    }
    for (int i1 = 0; i1 < 8; i1++) {
        t17[i1] = t11[i1] * t16[i1];
    }
    for (int i2 = 0; i2 < 8; i2++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 8; i1++) {
            acc += tables[1][i1*8 + i2] * t16[i1];
        }
        t18[i2] = acc;
    }
    for (int i2 = 0; i2 < 8; i2++) {
        t19[i2] = tables[9][i2] * t18[i2];
    }
    for (int i2 = 0; i2 < 8; i2++) {
        t20[i2] = t9[i2] * t19[i2];
    }
    for (int i3 = 0; i3 < 8; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 8; i2++) {
            acc += tables[2][i2*8 + i3] * t19[i2];
        }
        t21[i3] = acc;
    }
    for (int i3 = 0; i3 < 8; i3++) {
        t22[i3] = tables[10][i3] * t21[i3];
    }
    for (int i3 = 0; i3 < 8; i3++) {
        t23[i3] = t7[i3] * t22[i3];
    }
    for (int i4 = 0; i4 < 8; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 8; i3++) {
            acc += tables[3][i3*8 + i4] * t22[i3];
        }
        t24[i4] = acc;
    }
    for (int i4 = 0; i4 < 8; i4++) {
        t25[i4] = tables[11][i4] * t24[i4];
    }
    for (int i4 = 0; i4 < 8; i4++) {
        t26[i4] = t5[i4] * t25[i4];
    }
    for (int i5 = 0; i5 < 8; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 8; i4++) {
            acc += tables[4][i4*8 + i5] * t25[i4];
        }
        t27[i5] = acc;
    }
    for (int i5 = 0; i5 < 8; i5++) {
        t28[i5] = tables[12][i5] * t27[i5];
    }
    for (int i5 = 0; i5 < 8; i5++) {
        t29[i5] = t3[i5] * t28[i5];
    }
    for (int i6 = 0; i6 < 8; i6++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 8; i5++) {
            acc += tables[5][i5*8 + i6] * t28[i5];
        }
        t30[i6] = acc;
    }
    for (int i6 = 0; i6 < 8; i6++) {
        t31[i6] = tables[13][i6] * t30[i6];
    }
    for (int i6 = 0; i6 < 8; i6++) {
        t32[i6] = t1[i6] * t31[i6];
    }
    for (int i7 = 0; i7 < 8; i7++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 8; i6++) {
            acc += t0[i6*8 + i7] * t31[i6];
        }
        t33[i7] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 8; j++) z += t14[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 8; j++) out[0 + j] = t14[j] * iz;
    for (int j = 0; j < 8; j++) out[8 + j] = t17[j] * iz;
    for (int j = 0; j < 8; j++) out[16 + j] = t20[j] * iz;
    for (int j = 0; j < 8; j++) out[24 + j] = t23[j] * iz;
    for (int j = 0; j < 8; j++) out[32 + j] = t26[j] * iz;
    for (int j = 0; j < 8; j++) out[40 + j] = t29[j] * iz;
    for (int j = 0; j < 8; j++) out[48 + j] = t32[j] * iz;
    for (int j = 0; j < 8; j++) out[56 + j] = t33[j] * iz;
}
