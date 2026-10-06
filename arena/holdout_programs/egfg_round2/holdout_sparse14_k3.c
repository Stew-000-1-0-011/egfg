/* egfg program for holdout_sparse14_k3 (cost model: 1125 operations, overhead 64) */
#include <stddef.h>
#include <math.h>

static double t0[3];
static double t1[3];
static double t2[3];
static double t3[3];
static double t4[3];
static double t5[3];
static double t6[27];
static double t7[9];
static double t8[27];
static double t9[9];
static double t10[9];
static double t11[9];
static double t12[9];
static double t13[9];
static double t14[27];
static double t15[9];
static double t16[27];
static double t17[3];
static double t18[3];
static double t19[3];
static double t20[3];
static double t21[3];
static double t22[9];
static double t23[9];
static double t24[9];
static double t25[9];
static double t26[9];
static double t27[9];
static double t28[9];
static double t29[3];
static double t30[3];
static double t31[3];
static double t32[3];
static double t33[3];
static double t34[3];
static double t35[3];
static double t36[3];
static double t37[9];
static double t38[9];
static double t39[3];
static double t40[3];
static double t41[3];
static double t42[3];
static double t43[3];
static double t44[3];
static double t45[9];
static double t46[3];
static double t47[3];
static double t48[3];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 3; i8++) {
            acc += tables[3][i0*3 + i8];
        }
        t0[i0] = acc;
    }
    for (int i12 = 0; i12 < 3; i12++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += tables[10][i3 + i12*3];
        }
        t1[i12] = acc;
    }
    for (int i12 = 0; i12 < 3; i12++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            acc += tables[12][i5 + i12*3];
        }
        t2[i12] = acc;
    }
    for (int i12 = 0; i12 < 3; i12++) {
        t3[i12] = t2[i12] * t1[i12];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 3; i12++) {
            acc += tables[7][i0*3 + i12] * t3[i12];
        }
        t4[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t5[i0] = t4[i0] * t0[i0];
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i7 = 0; i7 < 3; i7++) {
            for (int i13 = 0; i13 < 3; i13++) {
                t6[i6*9 + i7*3 + i13] = tables[2][i6*3 + i7] * tables[15][i7*3 + i13];
            }
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i13 = 0; i13 < 3; i13++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 3; i7++) {
                acc += t6[i6*9 + i7*3 + i13];
            }
            t7[i6*3 + i13] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i4 = 0; i4 < 3; i4++) {
            for (int i11 = 0; i11 < 3; i11++) {
                t8[i2*9 + i4*3 + i11] = tables[9][i2 + i11*3] * tables[13][i2*3 + i4];
            }
        }
    }
    for (int i4 = 0; i4 < 3; i4++) {
        for (int i11 = 0; i11 < 3; i11++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 3; i2++) {
                acc += t8[i2*9 + i4*3 + i11];
            }
            t9[i4*3 + i11] = acc;
        }
    }
    for (int i11 = 0; i11 < 3; i11++) {
        for (int i13 = 0; i13 < 3; i13++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 3; i4++) {
                acc += tables[11][i4 + i13*3] * t9[i4*3 + i11];
            }
            t10[i11*3 + i13] = acc;
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i13 = 0; i13 < 3; i13++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 3; i11++) {
                acc += tables[6][i6*3 + i11] * t10[i11*3 + i13];
            }
            t11[i6*3 + i13] = acc;
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i13 = 0; i13 < 3; i13++) {
            t12[i6*3 + i13] = t11[i6*3 + i13] * t7[i6*3 + i13];
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i9 = 0; i9 < 3; i9++) {
            double acc = 0.0;
            for (int i13 = 0; i13 < 3; i13++) {
                acc += tables[8][i9*3 + i13] * t12[i6*3 + i13];
            }
            t13[i6*3 + i9] = acc;
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i9 = 0; i9 < 3; i9++) {
            for (int i10 = 0; i10 < 3; i10++) {
                t14[i6*9 + i9*3 + i10] = tables[4][i6*3 + i9] * tables[5][i9*3 + i10];
            }
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i10 = 0; i10 < 3; i10++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 3; i9++) {
                acc += t14[i6*9 + i9*3 + i10] * t13[i6*3 + i9];
            }
            t15[i6*3 + i10] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i6 = 0; i6 < 3; i6++) {
            for (int i10 = 0; i10 < 3; i10++) {
                t16[i1*9 + i6*3 + i10] = tables[1][i1*3 + i6] * tables[14][i1*3 + i10];
            }
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            for (int i10 = 0; i10 < 3; i10++) {
                acc += t16[i1*9 + i6*3 + i10] * t15[i6*3 + i10];
            }
        }
        t17[i1] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            acc += tables[0][i0*3 + i1] * t17[i1];
        }
        t18[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t19[i0] = t18[i0] * t5[i0];
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[0][i0*3 + i1] * t5[i0];
        }
        t20[i1] = acc;
    }
    for (int i1 = 0; i1 < 3; i1++) {
        t21[i1] = t17[i1] * t20[i1];
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i10 = 0; i10 < 3; i10++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 3; i1++) {
                acc += t16[i1*9 + i6*3 + i10] * t20[i1];
            }
            t22[i6*3 + i10] = acc;
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i9 = 0; i9 < 3; i9++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 3; i10++) {
                acc += t14[i6*9 + i9*3 + i10] * t22[i6*3 + i10];
            }
            t23[i6*3 + i9] = acc;
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i13 = 0; i13 < 3; i13++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 3; i9++) {
                acc += tables[8][i9*3 + i13] * t23[i6*3 + i9];
            }
            t24[i6*3 + i13] = acc;
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i13 = 0; i13 < 3; i13++) {
            t25[i6*3 + i13] = t24[i6*3 + i13] * t7[i6*3 + i13];
        }
    }
    for (int i11 = 0; i11 < 3; i11++) {
        for (int i13 = 0; i13 < 3; i13++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 3; i6++) {
                acc += tables[6][i6*3 + i11] * t25[i6*3 + i13];
            }
            t26[i11*3 + i13] = acc;
        }
    }
    for (int i4 = 0; i4 < 3; i4++) {
        for (int i11 = 0; i11 < 3; i11++) {
            double acc = 0.0;
            for (int i13 = 0; i13 < 3; i13++) {
                acc += tables[11][i4 + i13*3] * t26[i11*3 + i13];
            }
            t27[i4*3 + i11] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 3; i11++) {
                acc += t8[i2*9 + i4*3 + i11] * t27[i4*3 + i11];
            }
            t28[i2*3 + i4] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            acc += t28[i2*3 + i4];
        }
        t29[i2] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t30[i0] = t18[i0] * t0[i0];
    }
    for (int i12 = 0; i12 < 3; i12++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[7][i0*3 + i12] * t30[i0];
        }
        t31[i12] = acc;
    }
    for (int i12 = 0; i12 < 3; i12++) {
        t32[i12] = t31[i12] * t2[i12];
    }
    for (int i3 = 0; i3 < 3; i3++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 3; i12++) {
            acc += tables[10][i3 + i12*3] * t32[i12];
        }
        t33[i3] = acc;
    }
    for (int i4 = 0; i4 < 3; i4++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            acc += t28[i2*3 + i4];
        }
        t34[i4] = acc;
    }
    for (int i12 = 0; i12 < 3; i12++) {
        t35[i12] = t31[i12] * t1[i12];
    }
    for (int i5 = 0; i5 < 3; i5++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 3; i12++) {
            acc += tables[12][i5 + i12*3] * t35[i12];
        }
        t36[i5] = acc;
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i13 = 0; i13 < 3; i13++) {
            t37[i6*3 + i13] = t11[i6*3 + i13] * t24[i6*3 + i13];
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i7 = 0; i7 < 3; i7++) {
            double acc = 0.0;
            for (int i13 = 0; i13 < 3; i13++) {
                acc += t6[i6*9 + i7*3 + i13] * t37[i6*3 + i13];
            }
            t38[i6*3 + i7] = acc;
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 3; i7++) {
            acc += t38[i6*3 + i7];
        }
        t39[i6] = acc;
    }
    for (int i7 = 0; i7 < 3; i7++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += t38[i6*3 + i7];
        }
        t40[i7] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t41[i0] = t4[i0] * t18[i0];
    }
    for (int i8 = 0; i8 < 3; i8++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[3][i0*3 + i8] * t41[i0];
        }
        t42[i8] = acc;
    }
    for (int i9 = 0; i9 < 3; i9++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += t13[i6*3 + i9] * t23[i6*3 + i9];
        }
        t43[i9] = acc;
    }
    for (int i10 = 0; i10 < 3; i10++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += t22[i6*3 + i10] * t15[i6*3 + i10];
        }
        t44[i10] = acc;
    }
    for (int i11 = 0; i11 < 3; i11++) {
        for (int i13 = 0; i13 < 3; i13++) {
            t45[i11*3 + i13] = t26[i11*3 + i13] * t10[i11*3 + i13];
        }
    }
    for (int i11 = 0; i11 < 3; i11++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 3; i13++) {
            acc += t45[i11*3 + i13];
        }
        t46[i11] = acc;
    }
    for (int i12 = 0; i12 < 3; i12++) {
        t47[i12] = t31[i12] * t3[i12];
    }
    for (int i13 = 0; i13 < 3; i13++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 3; i11++) {
            acc += t45[i11*3 + i13];
        }
        t48[i13] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 3; j++) z += t19[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 3; j++) out[0 + j] = t19[j] * iz;
    for (int j = 0; j < 3; j++) out[3 + j] = t21[j] * iz;
    for (int j = 0; j < 3; j++) out[6 + j] = t29[j] * iz;
    for (int j = 0; j < 3; j++) out[9 + j] = t33[j] * iz;
    for (int j = 0; j < 3; j++) out[12 + j] = t34[j] * iz;
    for (int j = 0; j < 3; j++) out[15 + j] = t36[j] * iz;
    for (int j = 0; j < 3; j++) out[18 + j] = t39[j] * iz;
    for (int j = 0; j < 3; j++) out[21 + j] = t40[j] * iz;
    for (int j = 0; j < 3; j++) out[24 + j] = t42[j] * iz;
    for (int j = 0; j < 3; j++) out[27 + j] = t43[j] * iz;
    for (int j = 0; j < 3; j++) out[30 + j] = t44[j] * iz;
    for (int j = 0; j < 3; j++) out[33 + j] = t46[j] * iz;
    for (int j = 0; j < 3; j++) out[36 + j] = t47[j] * iz;
    for (int j = 0; j < 3; j++) out[39 + j] = t48[j] * iz;
}
