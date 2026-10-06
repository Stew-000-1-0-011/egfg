/* egfg program for sparse10_k3 (cost model: 723 operations, overhead 0) */
#include <stddef.h>

static double t0[3];
static double t1[3];
static double t2[3];
static double t3[3];
static double t4[27];
static double t5[9];
static double t6[3];
static double t7[9];
static double t8[3];
static double t9[27];
static double t10[9];
static double t11[9];
static double t12[9];
static double t13[3];
static double t14[3];
static double t15[9];
static double t16[27];
static double t17[9];
static double t18[3];
static double t19[3];
static double t20[3];
static double t21[3];
static double t22[3];
static double t23[9];
static double t24[9];
static double t25[3];
static double t26[3];
static double t27[3];
static double t28[3];
static double t29[3];
static double t30[3];
static double t31[27];
static double t32[3];
static double t33[3];
static double t34[3];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += tables[5][i0*3 + i6];
        }
        t0[i0] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 3; i9++) {
            acc += tables[8][i2*3 + i9];
        }
        t1[i2] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            acc += tables[1][i0*3 + i2] * t1[i2];
        }
        t2[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t3[i0] = t2[i0] * t0[i0];
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            for (int i7 = 0; i7 < 3; i7++) {
                t4[i3*9 + i4*3 + i7] = tables[6][i4*3 + i7] * tables[10][i3*3 + i7];
            }
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 3; i7++) {
                acc += t4[i3*9 + i4*3 + i7];
            }
            t5[i3*3 + i4] = acc;
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 3; i8++) {
            acc += tables[7][i3*3 + i8];
        }
        t6[i3] = acc;
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            t7[i3*3 + i4] = t6[i3] * t5[i3*3 + i4];
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            acc += tables[4][i1*3 + i5];
        }
        t8[i1] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i3 = 0; i3 < 3; i3++) {
                t9[i0*9 + i1*3 + i3] = tables[0][i0*3 + i1] * tables[9][i1*3 + i3];
            }
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i3 = 0; i3 < 3; i3++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 3; i1++) {
                acc += t9[i0*9 + i1*3 + i3] * t8[i1];
            }
            t10[i0*3 + i3] = acc;
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i3 = 0; i3 < 3; i3++) {
            t11[i0*3 + i3] = tables[2][i0*3 + i3] * t10[i0*3 + i3];
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 3; i3++) {
                acc += t11[i0*3 + i3] * t7[i3*3 + i4];
            }
            t12[i0*3 + i4] = acc;
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            acc += tables[3][i0*3 + i4] * t12[i0*3 + i4];
        }
        t13[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t14[i0] = t13[i0] * t3[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i4 = 0; i4 < 3; i4++) {
            t15[i0*3 + i4] = tables[3][i0*3 + i4] * t3[i0];
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i3 = 0; i3 < 3; i3++) {
            for (int i4 = 0; i4 < 3; i4++) {
                t16[i0*9 + i3*3 + i4] = tables[2][i0*3 + i3] * t15[i0*3 + i4];
            }
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i3 = 0; i3 < 3; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 3; i4++) {
                acc += t16[i0*9 + i3*3 + i4] * t7[i3*3 + i4];
            }
            t17[i0*3 + i3] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            for (int i3 = 0; i3 < 3; i3++) {
                acc += t9[i0*9 + i1*3 + i3] * t17[i0*3 + i3];
            }
        }
        t18[i1] = acc;
    }
    for (int i1 = 0; i1 < 3; i1++) {
        t19[i1] = t18[i1] * t8[i1];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t20[i0] = t13[i0] * t0[i0];
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[1][i0*3 + i2] * t20[i0];
        }
        t21[i2] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        t22[i2] = t21[i2] * t1[i2];
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 3; i0++) {
                acc += t16[i0*9 + i3*3 + i4] * t10[i0*3 + i3];
            }
            t23[i3*3 + i4] = acc;
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            t24[i3*3 + i4] = t23[i3*3 + i4] * t5[i3*3 + i4];
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            acc += t24[i3*3 + i4];
        }
        t25[i3] = acc;
    }
    for (int i3 = 0; i3 < 3; i3++) {
        t26[i3] = t25[i3] * t6[i3];
    }
    for (int i4 = 0; i4 < 3; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += t6[i3] * t24[i3*3 + i4];
        }
        t27[i4] = acc;
    }
    for (int i5 = 0; i5 < 3; i5++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            acc += tables[4][i1*3 + i5] * t18[i1];
        }
        t28[i5] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t29[i0] = t13[i0] * t2[i0];
    }
    for (int i6 = 0; i6 < 3; i6++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[5][i0*3 + i6] * t29[i0];
        }
        t30[i6] = acc;
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            for (int i7 = 0; i7 < 3; i7++) {
                t31[i3*9 + i4*3 + i7] = t4[i3*9 + i4*3 + i7] * t6[i3];
            }
        }
    }
    for (int i7 = 0; i7 < 3; i7++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            for (int i4 = 0; i4 < 3; i4++) {
                acc += t31[i3*9 + i4*3 + i7] * t23[i3*3 + i4];
            }
        }
        t32[i7] = acc;
    }
    for (int i8 = 0; i8 < 3; i8++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += tables[7][i3*3 + i8] * t25[i3];
        }
        t33[i8] = acc;
    }
    for (int i9 = 0; i9 < 3; i9++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            acc += tables[8][i2*3 + i9] * t21[i2];
        }
        t34[i9] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 3; j++) z += t14[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 3; j++) out[0 + j] = t14[j] * iz;
    for (int j = 0; j < 3; j++) out[3 + j] = t19[j] * iz;
    for (int j = 0; j < 3; j++) out[6 + j] = t22[j] * iz;
    for (int j = 0; j < 3; j++) out[9 + j] = t26[j] * iz;
    for (int j = 0; j < 3; j++) out[12 + j] = t27[j] * iz;
    for (int j = 0; j < 3; j++) out[15 + j] = t28[j] * iz;
    for (int j = 0; j < 3; j++) out[18 + j] = t30[j] * iz;
    for (int j = 0; j < 3; j++) out[21 + j] = t32[j] * iz;
    for (int j = 0; j < 3; j++) out[24 + j] = t33[j] * iz;
    for (int j = 0; j < 3; j++) out[27 + j] = t34[j] * iz;
}
