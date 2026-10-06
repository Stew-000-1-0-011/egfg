/* egfg program for holdout_cycle13_k4 (cost model: 3488 operations, overhead 64) */
#include <stddef.h>
#include <math.h>

static double t0[64];
static double t1[16];
static double t2[16];
static double t3[16];
static double t4[64];
static double t5[16];
static double t6[16];
static double t7[64];
static double t8[16];
static double t9[16];
static double t10[64];
static double t11[16];
static double t12[16];
static double t13[16];
static double t14[64];
static double t15[16];
static double t16[4];
static double t17[4];
static double t18[16];
static double t19[16];
static double t20[16];
static double t21[16];
static double t22[4];
static double t23[16];
static double t24[4];
static double t25[4];
static double t26[4];
static double t27[16];
static double t28[16];
static double t29[16];
static double t30[4];
static double t31[4];
static double t32[4];
static double t33[16];
static double t34[16];
static double t35[4];
static double t36[16];
static double t37[16];
static double t38[4];
static double t39[4];
static double t40[4];

void infer(const double *const *tables, double *out) {
    for (int i10 = 0; i10 < 4; i10++) {
        for (int i11 = 0; i11 < 4; i11++) {
            for (int i12 = 0; i12 < 4; i12++) {
                t0[i10*16 + i11*4 + i12] = tables[7][i10*4 + i11] * tables[8][i11*4 + i12];
            }
        }
    }
    for (int i10 = 0; i10 < 4; i10++) {
        for (int i12 = 0; i12 < 4; i12++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 4; i11++) {
                acc += t0[i10*16 + i11*4 + i12];
            }
            t1[i10*4 + i12] = acc;
        }
    }
    for (int i9 = 0; i9 < 4; i9++) {
        for (int i12 = 0; i12 < 4; i12++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 4; i10++) {
                acc += tables[6][i9*4 + i10] * t1[i10*4 + i12];
            }
            t2[i9*4 + i12] = acc;
        }
    }
    for (int i8 = 0; i8 < 4; i8++) {
        for (int i12 = 0; i12 < 4; i12++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 4; i9++) {
                acc += tables[5][i8*4 + i9] * t2[i9*4 + i12];
            }
            t3[i8*4 + i12] = acc;
        }
    }
    for (int i2 = 0; i2 < 4; i2++) {
        for (int i3 = 0; i3 < 4; i3++) {
            for (int i12 = 0; i12 < 4; i12++) {
                t4[i2*16 + i3*4 + i12] = tables[9][i2 + i12*4] * tables[10][i2*4 + i3];
            }
        }
    }
    for (int i3 = 0; i3 < 4; i3++) {
        for (int i12 = 0; i12 < 4; i12++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 4; i2++) {
                acc += t4[i2*16 + i3*4 + i12];
            }
            t5[i3*4 + i12] = acc;
        }
    }
    for (int i3 = 0; i3 < 4; i3++) {
        for (int i8 = 0; i8 < 4; i8++) {
            double acc = 0.0;
            for (int i12 = 0; i12 < 4; i12++) {
                acc += t5[i3*4 + i12] * t3[i8*4 + i12];
            }
            t6[i3*4 + i8] = acc;
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i3 = 0; i3 < 4; i3++) {
            for (int i4 = 0; i4 < 4; i4++) {
                t7[i0*16 + i3*4 + i4] = tables[11][i3*4 + i4] * tables[12][i0 + i4*4];
            }
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i3 = 0; i3 < 4; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 4; i4++) {
                acc += t7[i0*16 + i3*4 + i4];
            }
            t8[i0*4 + i3] = acc;
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i8 = 0; i8 < 4; i8++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 4; i3++) {
                acc += t8[i0*4 + i3] * t6[i3*4 + i8];
            }
            t9[i0*4 + i8] = acc;
        }
    }
    for (int i6 = 0; i6 < 4; i6++) {
        for (int i7 = 0; i7 < 4; i7++) {
            for (int i8 = 0; i8 < 4; i8++) {
                t10[i6*16 + i7*4 + i8] = tables[3][i6*4 + i7] * tables[4][i7*4 + i8];
            }
        }
    }
    for (int i6 = 0; i6 < 4; i6++) {
        for (int i8 = 0; i8 < 4; i8++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 4; i7++) {
                acc += t10[i6*16 + i7*4 + i8];
            }
            t11[i6*4 + i8] = acc;
        }
    }
    for (int i5 = 0; i5 < 4; i5++) {
        for (int i8 = 0; i8 < 4; i8++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 4; i6++) {
                acc += tables[2][i5*4 + i6] * t11[i6*4 + i8];
            }
            t12[i5*4 + i8] = acc;
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i5 = 0; i5 < 4; i5++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 4; i8++) {
                acc += t12[i5*4 + i8] * t9[i0*4 + i8];
            }
            t13[i0*4 + i5] = acc;
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            for (int i5 = 0; i5 < 4; i5++) {
                t14[i0*16 + i1*4 + i5] = tables[0][i0*4 + i1] * tables[1][i1*4 + i5];
            }
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 4; i5++) {
                acc += t14[i0*16 + i1*4 + i5] * t13[i0*4 + i5];
            }
            t15[i0*4 + i1] = acc;
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            acc += t15[i0*4 + i1];
        }
        t16[i0] = acc;
    }
    for (int i1 = 0; i1 < 4; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += t15[i0*4 + i1];
        }
        t17[i1] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i5 = 0; i5 < 4; i5++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 4; i1++) {
                acc += t14[i0*16 + i1*4 + i5];
            }
            t18[i0*4 + i5] = acc;
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i8 = 0; i8 < 4; i8++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 4; i5++) {
                acc += t18[i0*4 + i5] * t12[i5*4 + i8];
            }
            t19[i0*4 + i8] = acc;
        }
    }
    for (int i3 = 0; i3 < 4; i3++) {
        for (int i8 = 0; i8 < 4; i8++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 4; i0++) {
                acc += t8[i0*4 + i3] * t19[i0*4 + i8];
            }
            t20[i3*4 + i8] = acc;
        }
    }
    for (int i3 = 0; i3 < 4; i3++) {
        for (int i12 = 0; i12 < 4; i12++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 4; i8++) {
                acc += t20[i3*4 + i8] * t3[i8*4 + i12];
            }
            t21[i3*4 + i12] = acc;
        }
    }
    for (int i2 = 0; i2 < 4; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            for (int i12 = 0; i12 < 4; i12++) {
                acc += t4[i2*16 + i3*4 + i12] * t21[i3*4 + i12];
            }
        }
        t22[i2] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i3 = 0; i3 < 4; i3++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 4; i8++) {
                acc += t19[i0*4 + i8] * t6[i3*4 + i8];
            }
            t23[i0*4 + i3] = acc;
        }
    }
    for (int i3 = 0; i3 < 4; i3++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += t23[i0*4 + i3] * t8[i0*4 + i3];
        }
        t24[i3] = acc;
    }
    for (int i4 = 0; i4 < 4; i4++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            for (int i3 = 0; i3 < 4; i3++) {
                acc += t7[i0*16 + i3*4 + i4] * t23[i0*4 + i3];
            }
        }
        t25[i4] = acc;
    }
    for (int i5 = 0; i5 < 4; i5++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += t13[i0*4 + i5] * t18[i0*4 + i5];
        }
        t26[i5] = acc;
    }
    for (int i5 = 0; i5 < 4; i5++) {
        for (int i8 = 0; i8 < 4; i8++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 4; i0++) {
                acc += t18[i0*4 + i5] * t9[i0*4 + i8];
            }
            t27[i5*4 + i8] = acc;
        }
    }
    for (int i6 = 0; i6 < 4; i6++) {
        for (int i8 = 0; i8 < 4; i8++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 4; i5++) {
                acc += tables[2][i5*4 + i6] * t27[i5*4 + i8];
            }
            t28[i6*4 + i8] = acc;
        }
    }
    for (int i6 = 0; i6 < 4; i6++) {
        for (int i7 = 0; i7 < 4; i7++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 4; i8++) {
                acc += t10[i6*16 + i7*4 + i8] * t28[i6*4 + i8];
            }
            t29[i6*4 + i7] = acc;
        }
    }
    for (int i6 = 0; i6 < 4; i6++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 4; i7++) {
            acc += t29[i6*4 + i7];
        }
        t30[i6] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 4; i6++) {
            acc += t29[i6*4 + i7];
        }
        t31[i7] = acc;
    }
    for (int i8 = 0; i8 < 4; i8++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 4; i6++) {
            acc += t28[i6*4 + i8] * t11[i6*4 + i8];
        }
        t32[i8] = acc;
    }
    for (int i8 = 0; i8 < 4; i8++) {
        for (int i12 = 0; i12 < 4; i12++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 4; i3++) {
                acc += t20[i3*4 + i8] * t5[i3*4 + i12];
            }
            t33[i8*4 + i12] = acc;
        }
    }
    for (int i9 = 0; i9 < 4; i9++) {
        for (int i12 = 0; i12 < 4; i12++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 4; i8++) {
                acc += tables[5][i8*4 + i9] * t33[i8*4 + i12];
            }
            t34[i9*4 + i12] = acc;
        }
    }
    for (int i9 = 0; i9 < 4; i9++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 4; i12++) {
            acc += t2[i9*4 + i12] * t34[i9*4 + i12];
        }
        t35[i9] = acc;
    }
    for (int i10 = 0; i10 < 4; i10++) {
        for (int i12 = 0; i12 < 4; i12++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 4; i9++) {
                acc += tables[6][i9*4 + i10] * t34[i9*4 + i12];
            }
            t36[i10*4 + i12] = acc;
        }
    }
    for (int i10 = 0; i10 < 4; i10++) {
        for (int i11 = 0; i11 < 4; i11++) {
            double acc = 0.0;
            for (int i12 = 0; i12 < 4; i12++) {
                acc += t0[i10*16 + i11*4 + i12] * t36[i10*4 + i12];
            }
            t37[i10*4 + i11] = acc;
        }
    }
    for (int i10 = 0; i10 < 4; i10++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 4; i11++) {
            acc += t37[i10*4 + i11];
        }
        t38[i10] = acc;
    }
    for (int i11 = 0; i11 < 4; i11++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 4; i10++) {
            acc += t37[i10*4 + i11];
        }
        t39[i11] = acc;
    }
    for (int i12 = 0; i12 < 4; i12++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            acc += t21[i3*4 + i12] * t5[i3*4 + i12];
        }
        t40[i12] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 4; j++) z += t16[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 4; j++) out[0 + j] = t16[j] * iz;
    for (int j = 0; j < 4; j++) out[4 + j] = t17[j] * iz;
    for (int j = 0; j < 4; j++) out[8 + j] = t22[j] * iz;
    for (int j = 0; j < 4; j++) out[12 + j] = t24[j] * iz;
    for (int j = 0; j < 4; j++) out[16 + j] = t25[j] * iz;
    for (int j = 0; j < 4; j++) out[20 + j] = t26[j] * iz;
    for (int j = 0; j < 4; j++) out[24 + j] = t30[j] * iz;
    for (int j = 0; j < 4; j++) out[28 + j] = t31[j] * iz;
    for (int j = 0; j < 4; j++) out[32 + j] = t32[j] * iz;
    for (int j = 0; j < 4; j++) out[36 + j] = t35[j] * iz;
    for (int j = 0; j < 4; j++) out[40 + j] = t38[j] * iz;
    for (int j = 0; j < 4; j++) out[44 + j] = t39[j] * iz;
    for (int j = 0; j < 4; j++) out[48 + j] = t40[j] * iz;
}
