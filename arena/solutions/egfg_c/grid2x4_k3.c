/* egfg program for grid2x4_k3 (cost model: 801 operations, overhead 64) */
#include <stddef.h>

static double t0[27];
static double t1[9];
static double t2[27];
static double t3[9];
static double t4[9];
static double t5[9];
static double t6[27];
static double t7[9];
static double t8[27];
static double t9[9];
static double t10[3];
static double t11[9];
static double t12[9];
static double t13[9];
static double t14[3];
static double t15[9];
static double t16[3];
static double t17[9];
static double t18[9];
static double t19[9];
static double t20[3];
static double t21[3];
static double t22[3];
static double t23[3];
static double t24[3];

void infer(const double *const *tables, double *out) {
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i6 = 0; i6 < 3; i6++) {
            for (int i7 = 0; i7 < 3; i7++) {
                t0[i3*9 + i6*3 + i7] = tables[6][i3*3 + i7] * tables[9][i6*3 + i7];
            }
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i6 = 0; i6 < 3; i6++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 3; i7++) {
                acc += t0[i3*9 + i6*3 + i7];
            }
            t1[i3*3 + i6] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i3 = 0; i3 < 3; i3++) {
            for (int i6 = 0; i6 < 3; i6++) {
                t2[i2*9 + i3*3 + i6] = tables[4][i2*3 + i3] * tables[5][i2*3 + i6];
            }
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i6 = 0; i6 < 3; i6++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 3; i3++) {
                acc += t2[i2*9 + i3*3 + i6] * t1[i3*3 + i6];
            }
            t3[i2*3 + i6] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 3; i6++) {
                acc += tables[8][i5*3 + i6] * t3[i2*3 + i6];
            }
            t4[i2*3 + i5] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 3; i2++) {
                acc += tables[2][i1*3 + i2] * t4[i2*3 + i5];
            }
            t5[i1*3 + i5] = acc;
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i5 = 0; i5 < 3; i5++) {
                t6[i0*9 + i1*3 + i5] = tables[0][i0*3 + i1] * tables[3][i1*3 + i5];
            }
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 3; i1++) {
                acc += t6[i0*9 + i1*3 + i5] * t5[i1*3 + i5];
            }
            t7[i0*3 + i5] = acc;
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i4 = 0; i4 < 3; i4++) {
            for (int i5 = 0; i5 < 3; i5++) {
                t8[i0*9 + i4*3 + i5] = tables[1][i0*3 + i4] * tables[7][i4*3 + i5];
            }
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 3; i5++) {
                acc += t8[i0*9 + i4*3 + i5] * t7[i0*3 + i5];
            }
            t9[i0*3 + i4] = acc;
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            acc += t9[i0*3 + i4];
        }
        t10[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 3; i4++) {
                acc += t8[i0*9 + i4*3 + i5];
            }
            t11[i0*3 + i5] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 3; i0++) {
                acc += t6[i0*9 + i1*3 + i5] * t11[i0*3 + i5];
            }
            t12[i1*3 + i5] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i5 = 0; i5 < 3; i5++) {
            t13[i1*3 + i5] = t5[i1*3 + i5] * t12[i1*3 + i5];
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            acc += t13[i1*3 + i5];
        }
        t14[i1] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 3; i1++) {
                acc += tables[2][i1*3 + i2] * t12[i1*3 + i5];
            }
            t15[i2*3 + i5] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            acc += t4[i2*3 + i5] * t15[i2*3 + i5];
        }
        t16[i2] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i6 = 0; i6 < 3; i6++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 3; i5++) {
                acc += tables[8][i5*3 + i6] * t15[i2*3 + i5];
            }
            t17[i2*3 + i6] = acc;
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i6 = 0; i6 < 3; i6++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 3; i2++) {
                acc += t2[i2*9 + i3*3 + i6] * t17[i2*3 + i6];
            }
            t18[i3*3 + i6] = acc;
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i6 = 0; i6 < 3; i6++) {
            t19[i3*3 + i6] = t18[i3*3 + i6] * t1[i3*3 + i6];
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += t19[i3*3 + i6];
        }
        t20[i3] = acc;
    }
    for (int i4 = 0; i4 < 3; i4++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += t9[i0*3 + i4];
        }
        t21[i4] = acc;
    }
    for (int i5 = 0; i5 < 3; i5++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            acc += t13[i1*3 + i5];
        }
        t22[i5] = acc;
    }
    for (int i6 = 0; i6 < 3; i6++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += t19[i3*3 + i6];
        }
        t23[i6] = acc;
    }
    for (int i7 = 0; i7 < 3; i7++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            for (int i6 = 0; i6 < 3; i6++) {
                acc += t0[i3*9 + i6*3 + i7] * t18[i3*3 + i6];
            }
        }
        t24[i7] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 3; j++) z += t10[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 3; j++) out[0 + j] = t10[j] * iz;
    for (int j = 0; j < 3; j++) out[3 + j] = t14[j] * iz;
    for (int j = 0; j < 3; j++) out[6 + j] = t16[j] * iz;
    for (int j = 0; j < 3; j++) out[9 + j] = t20[j] * iz;
    for (int j = 0; j < 3; j++) out[12 + j] = t21[j] * iz;
    for (int j = 0; j < 3; j++) out[15 + j] = t22[j] * iz;
    for (int j = 0; j < 3; j++) out[18 + j] = t23[j] * iz;
    for (int j = 0; j < 3; j++) out[21 + j] = t24[j] * iz;
}
