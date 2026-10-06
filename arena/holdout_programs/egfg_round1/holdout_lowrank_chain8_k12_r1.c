/* egfg program for holdout_lowrank_chain8_k12_r1 (cost model: 3816 operations, overhead 64) */
#include <stddef.h>

static double t0[12];
static double t1[12];
static double t2[12];
static double t3[12];
static double t4[12];
static double t5[12];
static double t6[12];
static double t7[12];
static double t8[12];
static double t9[12];
static double t10[12];
static double t11[12];
static double t12[12];
static double t13[12];
static double t14[12];
static double t15[12];
static double t16[12];
static double t17[12];
static double t18[12];
static double t19[12];

void infer(const double *const *tables, double *out) {
    for (int i6 = 0; i6 < 12; i6++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 12; i7++) {
            acc += tables[6][i6*12 + i7];
        }
        t0[i6] = acc;
    }
    for (int i5 = 0; i5 < 12; i5++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 12; i6++) {
            acc += tables[5][i5*12 + i6] * t0[i6];
        }
        t1[i5] = acc;
    }
    for (int i4 = 0; i4 < 12; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 12; i5++) {
            acc += tables[4][i4*12 + i5] * t1[i5];
        }
        t2[i4] = acc;
    }
    for (int i3 = 0; i3 < 12; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 12; i4++) {
            acc += tables[3][i3*12 + i4] * t2[i4];
        }
        t3[i3] = acc;
    }
    for (int i2 = 0; i2 < 12; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 12; i3++) {
            acc += tables[2][i2*12 + i3] * t3[i3];
        }
        t4[i2] = acc;
    }
    for (int i1 = 0; i1 < 12; i1++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 12; i2++) {
            acc += tables[1][i1*12 + i2] * t4[i2];
        }
        t5[i1] = acc;
    }
    for (int i0 = 0; i0 < 12; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 12; i1++) {
            acc += tables[0][i0*12 + i1] * t5[i1];
        }
        t6[i0] = acc;
    }
    for (int i1 = 0; i1 < 12; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 12; i0++) {
            acc += tables[0][i0*12 + i1];
        }
        t7[i1] = acc;
    }
    for (int i1 = 0; i1 < 12; i1++) {
        t8[i1] = t5[i1] * t7[i1];
    }
    for (int i2 = 0; i2 < 12; i2++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 12; i1++) {
            acc += tables[1][i1*12 + i2] * t7[i1];
        }
        t9[i2] = acc;
    }
    for (int i2 = 0; i2 < 12; i2++) {
        t10[i2] = t4[i2] * t9[i2];
    }
    for (int i3 = 0; i3 < 12; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 12; i2++) {
            acc += tables[2][i2*12 + i3] * t9[i2];
        }
        t11[i3] = acc;
    }
    for (int i3 = 0; i3 < 12; i3++) {
        t12[i3] = t3[i3] * t11[i3];
    }
    for (int i4 = 0; i4 < 12; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 12; i3++) {
            acc += tables[3][i3*12 + i4] * t11[i3];
        }
        t13[i4] = acc;
    }
    for (int i4 = 0; i4 < 12; i4++) {
        t14[i4] = t2[i4] * t13[i4];
    }
    for (int i5 = 0; i5 < 12; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 12; i4++) {
            acc += tables[4][i4*12 + i5] * t13[i4];
        }
        t15[i5] = acc;
    }
    for (int i5 = 0; i5 < 12; i5++) {
        t16[i5] = t1[i5] * t15[i5];
    }
    for (int i6 = 0; i6 < 12; i6++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 12; i5++) {
            acc += tables[5][i5*12 + i6] * t15[i5];
        }
        t17[i6] = acc;
    }
    for (int i6 = 0; i6 < 12; i6++) {
        t18[i6] = t0[i6] * t17[i6];
    }
    for (int i7 = 0; i7 < 12; i7++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 12; i6++) {
            acc += tables[6][i6*12 + i7] * t17[i6];
        }
        t19[i7] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 12; j++) z += t6[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 12; j++) out[0 + j] = t6[j] * iz;
    for (int j = 0; j < 12; j++) out[12 + j] = t8[j] * iz;
    for (int j = 0; j < 12; j++) out[24 + j] = t10[j] * iz;
    for (int j = 0; j < 12; j++) out[36 + j] = t12[j] * iz;
    for (int j = 0; j < 12; j++) out[48 + j] = t14[j] * iz;
    for (int j = 0; j < 12; j++) out[60 + j] = t16[j] * iz;
    for (int j = 0; j < 12; j++) out[72 + j] = t18[j] * iz;
    for (int j = 0; j < 12; j++) out[84 + j] = t19[j] * iz;
}
