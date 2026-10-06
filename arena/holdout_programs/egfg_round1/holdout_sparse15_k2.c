/* egfg program for holdout_sparse15_k2 (cost model: 446 operations, overhead 0) */
#include <stddef.h>

static double t0[2];
static double t1[2];
static double t2[2];
static double t3[2];
static double t4[8];
static double t5[4];
static double t6[4];
static double t7[4];
static double t8[8];
static double t9[4];
static double t10[2];
static double t11[8];
static double t12[4];
static double t13[2];
static double t14[8];
static double t15[4];
static double t16[4];
static double t17[4];
static double t18[8];
static double t19[2];
static double t20[2];
static double t21[4];
static double t22[2];
static double t23[2];
static double t24[2];
static double t25[4];
static double t26[4];
static double t27[2];
static double t28[2];
static double t29[4];
static double t30[4];
static double t31[4];
static double t32[4];
static double t33[4];
static double t34[2];
static double t35[2];
static double t36[2];
static double t37[2];
static double t38[2];
static double t39[2];
static double t40[4];
static double t41[2];
static double t42[2];
static double t43[4];
static double t44[2];
static double t45[2];
static double t46[4];
static double t47[2];
static double t48[2];
static double t49[2];
static double t50[2];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 2; i0++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 2; i2++) {
            acc += tables[9][i0*2 + i2];
        }
        t0[i0] = acc;
    }
    for (int i12 = 0; i12 < 2; i12++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 2; i6++) {
            acc += tables[13][i6 + i12*2];
        }
        t1[i12] = acc;
    }
    for (int i0 = 0; i0 < 2; i0++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 2; i12++) {
            acc += tables[6][i0*2 + i12] * t1[i12];
        }
        t2[i0] = acc;
    }
    for (int i0 = 0; i0 < 2; i0++) {
        t3[i0] = t2[i0] * t0[i0];
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i5 = 0; i5 < 2; i5++) {
            for (int i14 = 0; i14 < 2; i14++) {
                t4[i4*4 + i5*2 + i14] = tables[12][i5 + i14*2] * tables[15][i4*2 + i5];
            }
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i14 = 0; i14 < 2; i14++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 2; i5++) {
                acc += t4[i4*4 + i5*2 + i14];
            }
            t5[i4*2 + i14] = acc;
        }
    }
    for (int i11 = 0; i11 < 2; i11++) {
        for (int i14 = 0; i14 < 2; i14++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 2; i4++) {
                acc += tables[11][i4 + i11*2] * t5[i4*2 + i14];
            }
            t6[i11*2 + i14] = acc;
        }
    }
    for (int i9 = 0; i9 < 2; i9++) {
        for (int i11 = 0; i11 < 2; i11++) {
            double acc = 0.0;
            for (int i14 = 0; i14 < 2; i14++) {
                acc += tables[8][i9*2 + i14] * t6[i11*2 + i14];
            }
            t7[i9*2 + i11] = acc;
        }
    }
    for (int i8 = 0; i8 < 2; i8++) {
        for (int i9 = 0; i9 < 2; i9++) {
            for (int i11 = 0; i11 < 2; i11++) {
                t8[i8*4 + i9*2 + i11] = tables[3][i8*2 + i9] * tables[5][i8*2 + i11];
            }
        }
    }
    for (int i8 = 0; i8 < 2; i8++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 2; i11++) {
                acc += t8[i8*4 + i9*2 + i11] * t7[i9*2 + i11];
            }
            t9[i8*2 + i9] = acc;
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 2; i3++) {
            acc += tables[10][i3 + i7*2];
        }
        t10[i7] = acc;
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i8 = 0; i8 < 2; i8++) {
                t11[i1*4 + i7*2 + i8] = tables[1][i1*2 + i7] * tables[2][i7*2 + i8];
            }
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i8 = 0; i8 < 2; i8++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 2; i7++) {
                acc += t11[i1*4 + i7*2 + i8] * t10[i7];
            }
            t12[i1*2 + i8] = acc;
        }
    }
    for (int i10 = 0; i10 < 2; i10++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 2; i13++) {
            acc += tables[7][i10*2 + i13];
        }
        t13[i10] = acc;
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i8 = 0; i8 < 2; i8++) {
            for (int i10 = 0; i10 < 2; i10++) {
                t14[i1*4 + i8*2 + i10] = tables[4][i8*2 + i10] * tables[14][i1*2 + i10];
            }
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i8 = 0; i8 < 2; i8++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 2; i10++) {
                acc += t14[i1*4 + i8*2 + i10] * t13[i10];
            }
            t15[i1*2 + i8] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i8 = 0; i8 < 2; i8++) {
            t16[i1*2 + i8] = t15[i1*2 + i8] * t12[i1*2 + i8];
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 2; i8++) {
                acc += t16[i1*2 + i8] * t9[i8*2 + i9];
            }
            t17[i1*2 + i9] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i1 = 0; i1 < 2; i1++) {
            for (int i9 = 0; i9 < 2; i9++) {
                t18[i0*4 + i1*2 + i9] = tables[0][i0*2 + i1] * tables[16][i0*2 + i9];
            }
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 2; i1++) {
            for (int i9 = 0; i9 < 2; i9++) {
                acc += t18[i0*4 + i1*2 + i9] * t17[i1*2 + i9];
            }
        }
        t19[i0] = acc;
    }
    for (int i0 = 0; i0 < 2; i0++) {
        t20[i0] = t19[i0] * t3[i0];
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 2; i0++) {
                acc += t18[i0*4 + i1*2 + i9] * t3[i0];
            }
            t21[i1*2 + i9] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 2; i9++) {
            acc += t17[i1*2 + i9] * t21[i1*2 + i9];
        }
        t22[i1] = acc;
    }
    for (int i0 = 0; i0 < 2; i0++) {
        t23[i0] = t19[i0] * t2[i0];
    }
    for (int i2 = 0; i2 < 2; i2++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 2; i0++) {
            acc += tables[9][i0*2 + i2] * t23[i0];
        }
        t24[i2] = acc;
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i8 = 0; i8 < 2; i8++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 2; i9++) {
                acc += t21[i1*2 + i9] * t9[i8*2 + i9];
            }
            t25[i1*2 + i8] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i8 = 0; i8 < 2; i8++) {
            t26[i1*2 + i8] = t15[i1*2 + i8] * t25[i1*2 + i8];
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 2; i1++) {
            for (int i8 = 0; i8 < 2; i8++) {
                acc += t11[i1*4 + i7*2 + i8] * t26[i1*2 + i8];
            }
        }
        t27[i7] = acc;
    }
    for (int i3 = 0; i3 < 2; i3++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 2; i7++) {
            acc += tables[10][i3 + i7*2] * t27[i7];
        }
        t28[i3] = acc;
    }
    for (int i8 = 0; i8 < 2; i8++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 2; i1++) {
                acc += t21[i1*2 + i9] * t16[i1*2 + i8];
            }
            t29[i8*2 + i9] = acc;
        }
    }
    for (int i9 = 0; i9 < 2; i9++) {
        for (int i11 = 0; i11 < 2; i11++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 2; i8++) {
                acc += t8[i8*4 + i9*2 + i11] * t29[i8*2 + i9];
            }
            t30[i9*2 + i11] = acc;
        }
    }
    for (int i11 = 0; i11 < 2; i11++) {
        for (int i14 = 0; i14 < 2; i14++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 2; i9++) {
                acc += tables[8][i9*2 + i14] * t30[i9*2 + i11];
            }
            t31[i11*2 + i14] = acc;
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i14 = 0; i14 < 2; i14++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 2; i11++) {
                acc += tables[11][i4 + i11*2] * t31[i11*2 + i14];
            }
            t32[i4*2 + i14] = acc;
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i5 = 0; i5 < 2; i5++) {
            double acc = 0.0;
            for (int i14 = 0; i14 < 2; i14++) {
                acc += t4[i4*4 + i5*2 + i14] * t32[i4*2 + i14];
            }
            t33[i4*2 + i5] = acc;
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 2; i5++) {
            acc += t33[i4*2 + i5];
        }
        t34[i4] = acc;
    }
    for (int i5 = 0; i5 < 2; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 2; i4++) {
            acc += t33[i4*2 + i5];
        }
        t35[i5] = acc;
    }
    for (int i0 = 0; i0 < 2; i0++) {
        t36[i0] = t19[i0] * t0[i0];
    }
    for (int i12 = 0; i12 < 2; i12++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 2; i0++) {
            acc += tables[6][i0*2 + i12] * t36[i0];
        }
        t37[i12] = acc;
    }
    for (int i6 = 0; i6 < 2; i6++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 2; i12++) {
            acc += tables[13][i6 + i12*2] * t37[i12];
        }
        t38[i6] = acc;
    }
    for (int i7 = 0; i7 < 2; i7++) {
        t39[i7] = t27[i7] * t10[i7];
    }
    for (int i8 = 0; i8 < 2; i8++) {
        for (int i9 = 0; i9 < 2; i9++) {
            t40[i8*2 + i9] = t29[i8*2 + i9] * t9[i8*2 + i9];
        }
    }
    for (int i8 = 0; i8 < 2; i8++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 2; i9++) {
            acc += t40[i8*2 + i9];
        }
        t41[i8] = acc;
    }
    for (int i9 = 0; i9 < 2; i9++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 2; i8++) {
            acc += t40[i8*2 + i9];
        }
        t42[i9] = acc;
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i8 = 0; i8 < 2; i8++) {
            t43[i1*2 + i8] = t25[i1*2 + i8] * t12[i1*2 + i8];
        }
    }
    for (int i10 = 0; i10 < 2; i10++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 2; i1++) {
            for (int i8 = 0; i8 < 2; i8++) {
                acc += t14[i1*4 + i8*2 + i10] * t43[i1*2 + i8];
            }
        }
        t44[i10] = acc;
    }
    for (int i10 = 0; i10 < 2; i10++) {
        t45[i10] = t44[i10] * t13[i10];
    }
    for (int i11 = 0; i11 < 2; i11++) {
        for (int i14 = 0; i14 < 2; i14++) {
            t46[i11*2 + i14] = t31[i11*2 + i14] * t6[i11*2 + i14];
        }
    }
    for (int i11 = 0; i11 < 2; i11++) {
        double acc = 0.0;
        for (int i14 = 0; i14 < 2; i14++) {
            acc += t46[i11*2 + i14];
        }
        t47[i11] = acc;
    }
    for (int i12 = 0; i12 < 2; i12++) {
        t48[i12] = t37[i12] * t1[i12];
    }
    for (int i13 = 0; i13 < 2; i13++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 2; i10++) {
            acc += tables[7][i10*2 + i13] * t44[i10];
        }
        t49[i13] = acc;
    }
    for (int i14 = 0; i14 < 2; i14++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 2; i11++) {
            acc += t46[i11*2 + i14];
        }
        t50[i14] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 2; j++) z += t20[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 2; j++) out[0 + j] = t20[j] * iz;
    for (int j = 0; j < 2; j++) out[2 + j] = t22[j] * iz;
    for (int j = 0; j < 2; j++) out[4 + j] = t24[j] * iz;
    for (int j = 0; j < 2; j++) out[6 + j] = t28[j] * iz;
    for (int j = 0; j < 2; j++) out[8 + j] = t34[j] * iz;
    for (int j = 0; j < 2; j++) out[10 + j] = t35[j] * iz;
    for (int j = 0; j < 2; j++) out[12 + j] = t38[j] * iz;
    for (int j = 0; j < 2; j++) out[14 + j] = t39[j] * iz;
    for (int j = 0; j < 2; j++) out[16 + j] = t41[j] * iz;
    for (int j = 0; j < 2; j++) out[18 + j] = t42[j] * iz;
    for (int j = 0; j < 2; j++) out[20 + j] = t45[j] * iz;
    for (int j = 0; j < 2; j++) out[22 + j] = t47[j] * iz;
    for (int j = 0; j < 2; j++) out[24 + j] = t48[j] * iz;
    for (int j = 0; j < 2; j++) out[26 + j] = t49[j] * iz;
    for (int j = 0; j < 2; j++) out[28 + j] = t50[j] * iz;
}
