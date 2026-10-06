/* egfg program for holdout_grid2x6_k2 (cost model: 432 operations, overhead 0) */
#include <stddef.h>

static double t0[8];
static double t1[4];
static double t2[8];
static double t3[4];
static double t4[4];
static double t5[8];
static double t6[8];
static double t7[4];
static double t8[4];
static double t9[4];
static double t10[8];
static double t11[4];
static double t12[4];
static double t13[8];
static double t14[4];
static double t15[8];
static double t16[4];
static double t17[2];
static double t18[4];
static double t19[4];
static double t20[4];
static double t21[2];
static double t22[4];
static double t23[4];
static double t24[4];
static double t25[4];
static double t26[4];
static double t27[2];
static double t28[4];
static double t29[4];
static double t30[4];
static double t31[4];
static double t32[2];
static double t33[2];
static double t34[4];
static double t35[2];
static double t36[2];
static double t37[2];
static double t38[2];
static double t39[2];
static double t40[2];
static double t41[2];

void infer(const double *const *tables, double *out) {
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i6 = 0; i6 < 2; i6++) {
            for (int i7 = 0; i7 < 2; i7++) {
                t0[i3*4 + i6*2 + i7] = tables[8][i6*2 + i7] * tables[10][i3 + i7*2];
            }
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i6 = 0; i6 < 2; i6++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 2; i7++) {
                acc += t0[i3*4 + i6*2 + i7];
            }
            t1[i3*2 + i6] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            for (int i6 = 0; i6 < 2; i6++) {
                t2[i2*4 + i3*2 + i6] = tables[9][i2 + i6*2] * tables[15][i2*2 + i3];
            }
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i6 = 0; i6 < 2; i6++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 2; i3++) {
                acc += t2[i2*4 + i3*2 + i6] * t1[i3*2 + i6];
            }
            t3[i2*2 + i6] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i5 = 0; i5 < 2; i5++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 2; i6++) {
                acc += tables[6][i5*2 + i6] * t3[i2*2 + i6];
            }
            t4[i2*2 + i5] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i5 = 0; i5 < 2; i5++) {
            for (int i11 = 0; i11 < 2; i11++) {
                t5[i2*4 + i5*2 + i11] = tables[7][i5*2 + i11] * tables[14][i2 + i11*2];
            }
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i5 = 0; i5 < 2; i5++) {
            for (int i11 = 0; i11 < 2; i11++) {
                t6[i2*4 + i5*2 + i11] = t5[i2*4 + i5*2 + i11] * t4[i2*2 + i5];
            }
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i11 = 0; i11 < 2; i11++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 2; i2++) {
                acc += t6[i2*4 + i5*2 + i11];
            }
            t7[i5*2 + i11] = acc;
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i10 = 0; i10 < 2; i10++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 2; i11++) {
                acc += tables[13][i10*2 + i11] * t7[i5*2 + i11];
            }
            t8[i5*2 + i10] = acc;
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i10 = 0; i10 < 2; i10++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 2; i5++) {
                acc += tables[4][i4*2 + i5] * t8[i5*2 + i10];
            }
            t9[i4*2 + i10] = acc;
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i9 = 0; i9 < 2; i9++) {
            for (int i10 = 0; i10 < 2; i10++) {
                t10[i4*4 + i9*2 + i10] = tables[5][i4*2 + i10] * tables[12][i9*2 + i10];
            }
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 2; i10++) {
                acc += t10[i4*4 + i9*2 + i10] * t9[i4*2 + i10];
            }
            t11[i4*2 + i9] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 2; i4++) {
                acc += tables[2][i1*2 + i4] * t11[i4*2 + i9];
            }
            t12[i1*2 + i9] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i1 = 0; i1 < 2; i1++) {
            for (int i9 = 0; i9 < 2; i9++) {
                t13[i0*4 + i1*2 + i9] = tables[0][i0*2 + i1] * tables[3][i1*2 + i9];
            }
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 2; i1++) {
                acc += t13[i0*4 + i1*2 + i9] * t12[i1*2 + i9];
            }
            t14[i0*2 + i9] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i8 = 0; i8 < 2; i8++) {
            for (int i9 = 0; i9 < 2; i9++) {
                t15[i0*4 + i8*2 + i9] = tables[1][i0*2 + i8] * tables[11][i8*2 + i9];
            }
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i8 = 0; i8 < 2; i8++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 2; i9++) {
                acc += t15[i0*4 + i8*2 + i9] * t14[i0*2 + i9];
            }
            t16[i0*2 + i8] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 2; i8++) {
            acc += t16[i0*2 + i8];
        }
        t17[i0] = acc;
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 2; i8++) {
                acc += t15[i0*4 + i8*2 + i9];
            }
            t18[i0*2 + i9] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 2; i0++) {
                acc += t13[i0*4 + i1*2 + i9] * t18[i0*2 + i9];
            }
            t19[i1*2 + i9] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i9 = 0; i9 < 2; i9++) {
            t20[i1*2 + i9] = t12[i1*2 + i9] * t19[i1*2 + i9];
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 2; i9++) {
            acc += t20[i1*2 + i9];
        }
        t21[i1] = acc;
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 2; i1++) {
                acc += tables[2][i1*2 + i4] * t19[i1*2 + i9];
            }
            t22[i4*2 + i9] = acc;
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i10 = 0; i10 < 2; i10++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 2; i9++) {
                acc += t10[i4*4 + i9*2 + i10] * t22[i4*2 + i9];
            }
            t23[i4*2 + i10] = acc;
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i10 = 0; i10 < 2; i10++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 2; i4++) {
                acc += tables[4][i4*2 + i5] * t23[i4*2 + i10];
            }
            t24[i5*2 + i10] = acc;
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i11 = 0; i11 < 2; i11++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 2; i10++) {
                acc += tables[13][i10*2 + i11] * t24[i5*2 + i10];
            }
            t25[i5*2 + i11] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i11 = 0; i11 < 2; i11++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 2; i5++) {
                acc += t6[i2*4 + i5*2 + i11] * t25[i5*2 + i11];
            }
            t26[i2*2 + i11] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 2; i11++) {
            acc += t26[i2*2 + i11];
        }
        t27[i2] = acc;
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i5 = 0; i5 < 2; i5++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 2; i11++) {
                acc += t5[i2*4 + i5*2 + i11] * t25[i5*2 + i11];
            }
            t28[i2*2 + i5] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i6 = 0; i6 < 2; i6++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 2; i5++) {
                acc += tables[6][i5*2 + i6] * t28[i2*2 + i5];
            }
            t29[i2*2 + i6] = acc;
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i6 = 0; i6 < 2; i6++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 2; i2++) {
                acc += t2[i2*4 + i3*2 + i6] * t29[i2*2 + i6];
            }
            t30[i3*2 + i6] = acc;
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i6 = 0; i6 < 2; i6++) {
            t31[i3*2 + i6] = t30[i3*2 + i6] * t1[i3*2 + i6];
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 2; i6++) {
            acc += t31[i3*2 + i6];
        }
        t32[i3] = acc;
    }
    for (int i4 = 0; i4 < 2; i4++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 2; i9++) {
            acc += t11[i4*2 + i9] * t22[i4*2 + i9];
        }
        t33[i4] = acc;
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i10 = 0; i10 < 2; i10++) {
            t34[i5*2 + i10] = t8[i5*2 + i10] * t24[i5*2 + i10];
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 2; i10++) {
            acc += t34[i5*2 + i10];
        }
        t35[i5] = acc;
    }
    for (int i6 = 0; i6 < 2; i6++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 2; i3++) {
            acc += t31[i3*2 + i6];
        }
        t36[i6] = acc;
    }
    for (int i7 = 0; i7 < 2; i7++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 2; i3++) {
            for (int i6 = 0; i6 < 2; i6++) {
                acc += t0[i3*4 + i6*2 + i7] * t30[i3*2 + i6];
            }
        }
        t37[i7] = acc;
    }
    for (int i8 = 0; i8 < 2; i8++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 2; i0++) {
            acc += t16[i0*2 + i8];
        }
        t38[i8] = acc;
    }
    for (int i9 = 0; i9 < 2; i9++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 2; i1++) {
            acc += t20[i1*2 + i9];
        }
        t39[i9] = acc;
    }
    for (int i10 = 0; i10 < 2; i10++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 2; i5++) {
            acc += t34[i5*2 + i10];
        }
        t40[i10] = acc;
    }
    for (int i11 = 0; i11 < 2; i11++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 2; i2++) {
            acc += t26[i2*2 + i11];
        }
        t41[i11] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 2; j++) z += t17[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 2; j++) out[0 + j] = t17[j] * iz;
    for (int j = 0; j < 2; j++) out[2 + j] = t21[j] * iz;
    for (int j = 0; j < 2; j++) out[4 + j] = t27[j] * iz;
    for (int j = 0; j < 2; j++) out[6 + j] = t32[j] * iz;
    for (int j = 0; j < 2; j++) out[8 + j] = t33[j] * iz;
    for (int j = 0; j < 2; j++) out[10 + j] = t35[j] * iz;
    for (int j = 0; j < 2; j++) out[12 + j] = t36[j] * iz;
    for (int j = 0; j < 2; j++) out[14 + j] = t37[j] * iz;
    for (int j = 0; j < 2; j++) out[16 + j] = t38[j] * iz;
    for (int j = 0; j < 2; j++) out[18 + j] = t39[j] * iz;
    for (int j = 0; j < 2; j++) out[20 + j] = t40[j] * iz;
    for (int j = 0; j < 2; j++) out[22 + j] = t41[j] * iz;
}
