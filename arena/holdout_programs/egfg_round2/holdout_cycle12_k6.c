/* egfg program for holdout_cycle12_k6 (cost model: 10044 operations, overhead 64) */
#include <stddef.h>
#include <math.h>

static double t0[216];
static double t1[36];
static double t2[36];
static double t3[216];
static double t4[36];
static double t5[36];
static double t6[216];
static double t7[36];
static double t8[36];
static double t9[216];
static double t10[36];
static double t11[36];
static double t12[36];
static double t13[36];
static double t14[36];
static double t15[6];
static double t16[36];
static double t17[6];
static double t18[36];
static double t19[36];
static double t20[6];
static double t21[6];
static double t22[36];
static double t23[36];
static double t24[36];
static double t25[6];
static double t26[6];
static double t27[36];
static double t28[36];
static double t29[36];
static double t30[6];
static double t31[6];
static double t32[36];
static double t33[6];
static double t34[36];
static double t35[36];
static double t36[6];
static double t37[6];
static double t38[6];

void infer(const double *const *tables, double *out) {
    for (int i2 = 0; i2 < 6; i2++) {
        for (int i3 = 0; i3 < 6; i3++) {
            for (int i11 = 0; i11 < 6; i11++) {
                t0[i2*36 + i3*6 + i11] = tables[9][i2 + i11*6] * tables[10][i2*6 + i3];
            }
        }
    }
    for (int i3 = 0; i3 < 6; i3++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 6; i2++) {
                acc += t0[i2*36 + i3*6 + i11];
            }
            t1[i3*6 + i11] = acc;
        }
    }
    for (int i0 = 0; i0 < 6; i0++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 6; i3++) {
                acc += tables[11][i0 + i3*6] * t1[i3*6 + i11];
            }
            t2[i0*6 + i11] = acc;
        }
    }
    for (int i9 = 0; i9 < 6; i9++) {
        for (int i10 = 0; i10 < 6; i10++) {
            for (int i11 = 0; i11 < 6; i11++) {
                t3[i9*36 + i10*6 + i11] = tables[7][i9*6 + i10] * tables[8][i10*6 + i11];
            }
        }
    }
    for (int i9 = 0; i9 < 6; i9++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 6; i10++) {
                acc += t3[i9*36 + i10*6 + i11];
            }
            t4[i9*6 + i11] = acc;
        }
    }
    for (int i8 = 0; i8 < 6; i8++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 6; i9++) {
                acc += tables[6][i8*6 + i9] * t4[i9*6 + i11];
            }
            t5[i8*6 + i11] = acc;
        }
    }
    for (int i6 = 0; i6 < 6; i6++) {
        for (int i7 = 0; i7 < 6; i7++) {
            for (int i8 = 0; i8 < 6; i8++) {
                t6[i6*36 + i7*6 + i8] = tables[4][i6*6 + i7] * tables[5][i7*6 + i8];
            }
        }
    }
    for (int i6 = 0; i6 < 6; i6++) {
        for (int i8 = 0; i8 < 6; i8++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 6; i7++) {
                acc += t6[i6*36 + i7*6 + i8];
            }
            t7[i6*6 + i8] = acc;
        }
    }
    for (int i6 = 0; i6 < 6; i6++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 6; i8++) {
                acc += t7[i6*6 + i8] * t5[i8*6 + i11];
            }
            t8[i6*6 + i11] = acc;
        }
    }
    for (int i4 = 0; i4 < 6; i4++) {
        for (int i5 = 0; i5 < 6; i5++) {
            for (int i6 = 0; i6 < 6; i6++) {
                t9[i4*36 + i5*6 + i6] = tables[2][i4*6 + i5] * tables[3][i5*6 + i6];
            }
        }
    }
    for (int i4 = 0; i4 < 6; i4++) {
        for (int i6 = 0; i6 < 6; i6++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 6; i5++) {
                acc += t9[i4*36 + i5*6 + i6];
            }
            t10[i4*6 + i6] = acc;
        }
    }
    for (int i4 = 0; i4 < 6; i4++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 6; i6++) {
                acc += t10[i4*6 + i6] * t8[i6*6 + i11];
            }
            t11[i4*6 + i11] = acc;
        }
    }
    for (int i1 = 0; i1 < 6; i1++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 6; i4++) {
                acc += tables[1][i1*6 + i4] * t11[i4*6 + i11];
            }
            t12[i1*6 + i11] = acc;
        }
    }
    for (int i0 = 0; i0 < 6; i0++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 6; i1++) {
                acc += tables[0][i0*6 + i1] * t12[i1*6 + i11];
            }
            t13[i0*6 + i11] = acc;
        }
    }
    for (int i0 = 0; i0 < 6; i0++) {
        for (int i11 = 0; i11 < 6; i11++) {
            t14[i0*6 + i11] = t13[i0*6 + i11] * t2[i0*6 + i11];
        }
    }
    for (int i0 = 0; i0 < 6; i0++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 6; i11++) {
            acc += t14[i0*6 + i11];
        }
        t15[i0] = acc;
    }
    for (int i1 = 0; i1 < 6; i1++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 6; i0++) {
                acc += tables[0][i0*6 + i1] * t2[i0*6 + i11];
            }
            t16[i1*6 + i11] = acc;
        }
    }
    for (int i1 = 0; i1 < 6; i1++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 6; i11++) {
            acc += t12[i1*6 + i11] * t16[i1*6 + i11];
        }
        t17[i1] = acc;
    }
    for (int i3 = 0; i3 < 6; i3++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 6; i0++) {
                acc += tables[11][i0 + i3*6] * t13[i0*6 + i11];
            }
            t18[i3*6 + i11] = acc;
        }
    }
    for (int i2 = 0; i2 < 6; i2++) {
        for (int i3 = 0; i3 < 6; i3++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 6; i11++) {
                acc += t0[i2*36 + i3*6 + i11] * t18[i3*6 + i11];
            }
            t19[i2*6 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 6; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 6; i3++) {
            acc += t19[i2*6 + i3];
        }
        t20[i2] = acc;
    }
    for (int i3 = 0; i3 < 6; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 6; i2++) {
            acc += t19[i2*6 + i3];
        }
        t21[i3] = acc;
    }
    for (int i4 = 0; i4 < 6; i4++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 6; i1++) {
                acc += tables[1][i1*6 + i4] * t16[i1*6 + i11];
            }
            t22[i4*6 + i11] = acc;
        }
    }
    for (int i4 = 0; i4 < 6; i4++) {
        for (int i6 = 0; i6 < 6; i6++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 6; i11++) {
                acc += t22[i4*6 + i11] * t8[i6*6 + i11];
            }
            t23[i4*6 + i6] = acc;
        }
    }
    for (int i4 = 0; i4 < 6; i4++) {
        for (int i5 = 0; i5 < 6; i5++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 6; i6++) {
                acc += t9[i4*36 + i5*6 + i6] * t23[i4*6 + i6];
            }
            t24[i4*6 + i5] = acc;
        }
    }
    for (int i4 = 0; i4 < 6; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 6; i5++) {
            acc += t24[i4*6 + i5];
        }
        t25[i4] = acc;
    }
    for (int i5 = 0; i5 < 6; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 6; i4++) {
            acc += t24[i4*6 + i5];
        }
        t26[i5] = acc;
    }
    for (int i6 = 0; i6 < 6; i6++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 6; i4++) {
                acc += t10[i4*6 + i6] * t22[i4*6 + i11];
            }
            t27[i6*6 + i11] = acc;
        }
    }
    for (int i6 = 0; i6 < 6; i6++) {
        for (int i8 = 0; i8 < 6; i8++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 6; i11++) {
                acc += t27[i6*6 + i11] * t5[i8*6 + i11];
            }
            t28[i6*6 + i8] = acc;
        }
    }
    for (int i6 = 0; i6 < 6; i6++) {
        for (int i7 = 0; i7 < 6; i7++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 6; i8++) {
                acc += t6[i6*36 + i7*6 + i8] * t28[i6*6 + i8];
            }
            t29[i6*6 + i7] = acc;
        }
    }
    for (int i6 = 0; i6 < 6; i6++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 6; i7++) {
            acc += t29[i6*6 + i7];
        }
        t30[i6] = acc;
    }
    for (int i7 = 0; i7 < 6; i7++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 6; i6++) {
            acc += t29[i6*6 + i7];
        }
        t31[i7] = acc;
    }
    for (int i8 = 0; i8 < 6; i8++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 6; i6++) {
                acc += t7[i6*6 + i8] * t27[i6*6 + i11];
            }
            t32[i8*6 + i11] = acc;
        }
    }
    for (int i8 = 0; i8 < 6; i8++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 6; i11++) {
            acc += t5[i8*6 + i11] * t32[i8*6 + i11];
        }
        t33[i8] = acc;
    }
    for (int i9 = 0; i9 < 6; i9++) {
        for (int i11 = 0; i11 < 6; i11++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 6; i8++) {
                acc += tables[6][i8*6 + i9] * t32[i8*6 + i11];
            }
            t34[i9*6 + i11] = acc;
        }
    }
    for (int i9 = 0; i9 < 6; i9++) {
        for (int i10 = 0; i10 < 6; i10++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 6; i11++) {
                acc += t3[i9*36 + i10*6 + i11] * t34[i9*6 + i11];
            }
            t35[i9*6 + i10] = acc;
        }
    }
    for (int i9 = 0; i9 < 6; i9++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 6; i10++) {
            acc += t35[i9*6 + i10];
        }
        t36[i9] = acc;
    }
    for (int i10 = 0; i10 < 6; i10++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 6; i9++) {
            acc += t35[i9*6 + i10];
        }
        t37[i10] = acc;
    }
    for (int i11 = 0; i11 < 6; i11++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 6; i0++) {
            acc += t14[i0*6 + i11];
        }
        t38[i11] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 6; j++) z += t15[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 6; j++) out[0 + j] = t15[j] * iz;
    for (int j = 0; j < 6; j++) out[6 + j] = t17[j] * iz;
    for (int j = 0; j < 6; j++) out[12 + j] = t20[j] * iz;
    for (int j = 0; j < 6; j++) out[18 + j] = t21[j] * iz;
    for (int j = 0; j < 6; j++) out[24 + j] = t25[j] * iz;
    for (int j = 0; j < 6; j++) out[30 + j] = t26[j] * iz;
    for (int j = 0; j < 6; j++) out[36 + j] = t30[j] * iz;
    for (int j = 0; j < 6; j++) out[42 + j] = t31[j] * iz;
    for (int j = 0; j < 6; j++) out[48 + j] = t33[j] * iz;
    for (int j = 0; j < 6; j++) out[54 + j] = t36[j] * iz;
    for (int j = 0; j < 6; j++) out[60 + j] = t37[j] * iz;
    for (int j = 0; j < 6; j++) out[66 + j] = t38[j] * iz;
}
