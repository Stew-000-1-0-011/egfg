/* egfg program for tree12_k3 (cost model: 399 operations, overhead 0) */
#include <stddef.h>

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
static double t24[3];
static double t25[3];
static double t26[3];
static double t27[3];
static double t28[3];
static double t29[3];
static double t30[3];
static double t31[3];
static double t32[3];
static double t33[3];
static double t34[3];
static double t35[3];
static double t36[3];
static double t37[3];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            acc += tables[9][i0*3 + i2];
        }
        t0[i0] = acc;
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 3; i9++) {
            acc += tables[6][i1*3 + i9];
        }
        t1[i1] = acc;
    }
    for (int i5 = 0; i5 < 3; i5++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += tables[10][i3 + i5*3];
        }
        t2[i5] = acc;
    }
    for (int i4 = 0; i4 < 3; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            acc += tables[2][i4*3 + i5] * t2[i5];
        }
        t3[i4] = acc;
    }
    for (int i7 = 0; i7 < 3; i7++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 3; i11++) {
            acc += tables[8][i7*3 + i11];
        }
        t4[i7] = acc;
    }
    for (int i6 = 0; i6 < 3; i6++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 3; i7++) {
            acc += tables[4][i6*3 + i7] * t4[i7];
        }
        t5[i6] = acc;
    }
    for (int i8 = 0; i8 < 3; i8++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 3; i10++) {
            acc += tables[7][i8*3 + i10];
        }
        t6[i8] = acc;
    }
    for (int i6 = 0; i6 < 3; i6++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 3; i8++) {
            acc += tables[5][i6*3 + i8] * t6[i8];
        }
        t7[i6] = acc;
    }
    for (int i6 = 0; i6 < 3; i6++) {
        t8[i6] = t7[i6] * t5[i6];
    }
    for (int i4 = 0; i4 < 3; i4++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += tables[3][i4*3 + i6] * t8[i6];
        }
        t9[i4] = acc;
    }
    for (int i4 = 0; i4 < 3; i4++) {
        t10[i4] = t9[i4] * t3[i4];
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            acc += tables[1][i1*3 + i4] * t10[i4];
        }
        t11[i1] = acc;
    }
    for (int i1 = 0; i1 < 3; i1++) {
        t12[i1] = t11[i1] * t1[i1];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            acc += tables[0][i0*3 + i1] * t12[i1];
        }
        t13[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t14[i0] = t13[i0] * t0[i0];
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[0][i0*3 + i1] * t0[i0];
        }
        t15[i1] = acc;
    }
    for (int i1 = 0; i1 < 3; i1++) {
        t16[i1] = t1[i1] * t15[i1];
    }
    for (int i1 = 0; i1 < 3; i1++) {
        t17[i1] = t11[i1] * t16[i1];
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[9][i0*3 + i2] * t13[i0];
        }
        t18[i2] = acc;
    }
    for (int i4 = 0; i4 < 3; i4++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            acc += tables[1][i1*3 + i4] * t16[i1];
        }
        t19[i4] = acc;
    }
    for (int i4 = 0; i4 < 3; i4++) {
        t20[i4] = t9[i4] * t19[i4];
    }
    for (int i5 = 0; i5 < 3; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            acc += tables[2][i4*3 + i5] * t20[i4];
        }
        t21[i5] = acc;
    }
    for (int i3 = 0; i3 < 3; i3++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            acc += tables[10][i3 + i5*3] * t21[i5];
        }
        t22[i3] = acc;
    }
    for (int i4 = 0; i4 < 3; i4++) {
        t23[i4] = t20[i4] * t3[i4];
    }
    for (int i5 = 0; i5 < 3; i5++) {
        t24[i5] = t21[i5] * t2[i5];
    }
    for (int i4 = 0; i4 < 3; i4++) {
        t25[i4] = t19[i4] * t3[i4];
    }
    for (int i6 = 0; i6 < 3; i6++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            acc += tables[3][i4*3 + i6] * t25[i4];
        }
        t26[i6] = acc;
    }
    for (int i6 = 0; i6 < 3; i6++) {
        t27[i6] = t26[i6] * t8[i6];
    }
    for (int i6 = 0; i6 < 3; i6++) {
        t28[i6] = t7[i6] * t26[i6];
    }
    for (int i7 = 0; i7 < 3; i7++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += tables[4][i6*3 + i7] * t28[i6];
        }
        t29[i7] = acc;
    }
    for (int i7 = 0; i7 < 3; i7++) {
        t30[i7] = t29[i7] * t4[i7];
    }
    for (int i6 = 0; i6 < 3; i6++) {
        t31[i6] = t5[i6] * t26[i6];
    }
    for (int i8 = 0; i8 < 3; i8++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += tables[5][i6*3 + i8] * t31[i6];
        }
        t32[i8] = acc;
    }
    for (int i8 = 0; i8 < 3; i8++) {
        t33[i8] = t32[i8] * t6[i8];
    }
    for (int i1 = 0; i1 < 3; i1++) {
        t34[i1] = t11[i1] * t15[i1];
    }
    for (int i9 = 0; i9 < 3; i9++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            acc += tables[6][i1*3 + i9] * t34[i1];
        }
        t35[i9] = acc;
    }
    for (int i10 = 0; i10 < 3; i10++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 3; i8++) {
            acc += tables[7][i8*3 + i10] * t32[i8];
        }
        t36[i10] = acc;
    }
    for (int i11 = 0; i11 < 3; i11++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 3; i7++) {
            acc += tables[8][i7*3 + i11] * t29[i7];
        }
        t37[i11] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 3; j++) z += t14[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 3; j++) out[0 + j] = t14[j] * iz;
    for (int j = 0; j < 3; j++) out[3 + j] = t17[j] * iz;
    for (int j = 0; j < 3; j++) out[6 + j] = t18[j] * iz;
    for (int j = 0; j < 3; j++) out[9 + j] = t22[j] * iz;
    for (int j = 0; j < 3; j++) out[12 + j] = t23[j] * iz;
    for (int j = 0; j < 3; j++) out[15 + j] = t24[j] * iz;
    for (int j = 0; j < 3; j++) out[18 + j] = t27[j] * iz;
    for (int j = 0; j < 3; j++) out[21 + j] = t30[j] * iz;
    for (int j = 0; j < 3; j++) out[24 + j] = t33[j] * iz;
    for (int j = 0; j < 3; j++) out[27 + j] = t35[j] * iz;
    for (int j = 0; j < 3; j++) out[30 + j] = t36[j] * iz;
    for (int j = 0; j < 3; j++) out[33 + j] = t37[j] * iz;
}
