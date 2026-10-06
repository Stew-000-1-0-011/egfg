/* egfg program for ternary8_k3 (cost model: 693 operations, overhead 0) */
#include <stddef.h>
#include <math.h>

static double t0[9];
static double t1[9];
static double t2[9];
static double t3[9];
static double t4[9];
static double t5[3];
static double t6[9];
static double t7[9];
static double t8[3];
static double t9[3];
static double t10[9];
static double t11[9];
static double t12[9];
static double t13[3];
static double t14[3];
static double t15[9];
static double t16[9];
static double t17[9];
static double t18[3];
static double t19[3];
static double t20[3];

void infer(const double *const *tables, double *out) {
    for (int i5 = 0; i5 < 3; i5++) {
        for (int i6 = 0; i6 < 3; i6++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 3; i7++) {
                acc += tables[5][i5*9 + i6*3 + i7];
            }
            t0[i5*3 + i6] = acc;
        }
    }
    for (int i4 = 0; i4 < 3; i4++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 3; i6++) {
                acc += tables[4][i4*9 + i5*3 + i6] * t0[i5*3 + i6];
            }
            t1[i4*3 + i5] = acc;
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 3; i5++) {
                acc += tables[3][i3*9 + i4*3 + i5] * t1[i4*3 + i5];
            }
            t2[i3*3 + i4] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i3 = 0; i3 < 3; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 3; i4++) {
                acc += tables[2][i2*9 + i3*3 + i4] * t2[i3*3 + i4];
            }
            t3[i2*3 + i3] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i2 = 0; i2 < 3; i2++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 3; i3++) {
                acc += tables[1][i1*9 + i2*3 + i3] * t3[i2*3 + i3];
            }
            t4[i1*3 + i2] = acc;
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 3; i2++) {
                acc += tables[0][i0*9 + i1*3 + i2] * t4[i1*3 + i2];
            }
        }
        t5[i0] = acc;
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
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i2 = 0; i2 < 3; i2++) {
            t7[i1*3 + i2] = t6[i1*3 + i2] * t4[i1*3 + i2];
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            acc += t7[i1*3 + i2];
        }
        t8[i1] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            acc += t7[i1*3 + i2];
        }
        t9[i2] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i3 = 0; i3 < 3; i3++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 3; i1++) {
                acc += tables[1][i1*9 + i2*3 + i3] * t6[i1*3 + i2];
            }
            t10[i2*3 + i3] = acc;
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 3; i2++) {
                acc += tables[2][i2*9 + i3*3 + i4] * t10[i2*3 + i3];
            }
            t11[i3*3 + i4] = acc;
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            t12[i3*3 + i4] = t11[i3*3 + i4] * t2[i3*3 + i4];
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            acc += t12[i3*3 + i4];
        }
        t13[i3] = acc;
    }
    for (int i4 = 0; i4 < 3; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += t12[i3*3 + i4];
        }
        t14[i4] = acc;
    }
    for (int i4 = 0; i4 < 3; i4++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 3; i3++) {
                acc += tables[3][i3*9 + i4*3 + i5] * t11[i3*3 + i4];
            }
            t15[i4*3 + i5] = acc;
        }
    }
    for (int i5 = 0; i5 < 3; i5++) {
        for (int i6 = 0; i6 < 3; i6++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 3; i4++) {
                acc += tables[4][i4*9 + i5*3 + i6] * t15[i4*3 + i5];
            }
            t16[i5*3 + i6] = acc;
        }
    }
    for (int i5 = 0; i5 < 3; i5++) {
        for (int i6 = 0; i6 < 3; i6++) {
            t17[i5*3 + i6] = t16[i5*3 + i6] * t0[i5*3 + i6];
        }
    }
    for (int i5 = 0; i5 < 3; i5++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += t17[i5*3 + i6];
        }
        t18[i5] = acc;
    }
    for (int i6 = 0; i6 < 3; i6++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            acc += t17[i5*3 + i6];
        }
        t19[i6] = acc;
    }
    for (int i7 = 0; i7 < 3; i7++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            for (int i6 = 0; i6 < 3; i6++) {
                acc += tables[5][i5*9 + i6*3 + i7] * t16[i5*3 + i6];
            }
        }
        t20[i7] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 3; j++) z += t5[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 3; j++) out[0 + j] = t5[j] * iz;
    for (int j = 0; j < 3; j++) out[3 + j] = t8[j] * iz;
    for (int j = 0; j < 3; j++) out[6 + j] = t9[j] * iz;
    for (int j = 0; j < 3; j++) out[9 + j] = t13[j] * iz;
    for (int j = 0; j < 3; j++) out[12 + j] = t14[j] * iz;
    for (int j = 0; j < 3; j++) out[15 + j] = t18[j] * iz;
    for (int j = 0; j < 3; j++) out[18 + j] = t19[j] * iz;
    for (int j = 0; j < 3; j++) out[21 + j] = t20[j] * iz;
}
