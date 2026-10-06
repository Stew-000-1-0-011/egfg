/* egfg program for holdout_ternary10_k2 (cost model: 296 operations, overhead 0) */
#include <stddef.h>
#include <math.h>

static double t0[4];
static double t1[4];
static double t2[4];
static double t3[4];
static double t4[4];
static double t5[4];
static double t6[4];
static double t7[4];
static double t8[2];
static double t9[2];
static double t10[4];
static double t11[4];
static double t12[4];
static double t13[2];
static double t14[2];
static double t15[4];
static double t16[4];
static double t17[4];
static double t18[2];
static double t19[2];
static double t20[4];
static double t21[2];
static double t22[4];
static double t23[4];
static double t24[4];
static double t25[2];
static double t26[2];
static double t27[2];

void infer(const double *const *tables, double *out) {
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i8 = 0; i8 < 2; i8++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 2; i9++) {
                acc += tables[7][i7*4 + i8*2 + i9];
            }
            t0[i7*2 + i8] = acc;
        }
    }
    for (int i6 = 0; i6 < 2; i6++) {
        for (int i7 = 0; i7 < 2; i7++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 2; i8++) {
                acc += tables[6][i6*4 + i7*2 + i8] * t0[i7*2 + i8];
            }
            t1[i6*2 + i7] = acc;
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i6 = 0; i6 < 2; i6++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 2; i7++) {
                acc += tables[5][i5*4 + i6*2 + i7] * t1[i6*2 + i7];
            }
            t2[i5*2 + i6] = acc;
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i5 = 0; i5 < 2; i5++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 2; i6++) {
                acc += tables[4][i4*4 + i5*2 + i6] * t2[i5*2 + i6];
            }
            t3[i4*2 + i5] = acc;
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i4 = 0; i4 < 2; i4++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 2; i5++) {
                acc += tables[3][i3*4 + i4*2 + i5] * t3[i4*2 + i5];
            }
            t4[i3*2 + i4] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 2; i4++) {
                acc += tables[2][i2*4 + i3*2 + i4] * t4[i3*2 + i4];
            }
            t5[i2*2 + i3] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i2 = 0; i2 < 2; i2++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 2; i3++) {
                acc += tables[1][i1*4 + i2*2 + i3] * t5[i2*2 + i3];
            }
            t6[i1*2 + i2] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i1 = 0; i1 < 2; i1++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 2; i2++) {
                acc += tables[0][i0*4 + i1*2 + i2] * t6[i1*2 + i2];
            }
            t7[i0*2 + i1] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 2; i1++) {
            acc += t7[i0*2 + i1];
        }
        t8[i0] = acc;
    }
    for (int i1 = 0; i1 < 2; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 2; i0++) {
            acc += t7[i0*2 + i1];
        }
        t9[i1] = acc;
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i2 = 0; i2 < 2; i2++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 2; i0++) {
                acc += tables[0][i0*4 + i1*2 + i2];
            }
            t10[i1*2 + i2] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 2; i1++) {
                acc += tables[1][i1*4 + i2*2 + i3] * t10[i1*2 + i2];
            }
            t11[i2*2 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            t12[i2*2 + i3] = t11[i2*2 + i3] * t5[i2*2 + i3];
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 2; i3++) {
            acc += t12[i2*2 + i3];
        }
        t13[i2] = acc;
    }
    for (int i3 = 0; i3 < 2; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 2; i2++) {
            acc += t12[i2*2 + i3];
        }
        t14[i3] = acc;
    }
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i4 = 0; i4 < 2; i4++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 2; i2++) {
                acc += tables[2][i2*4 + i3*2 + i4] * t11[i2*2 + i3];
            }
            t15[i3*2 + i4] = acc;
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i5 = 0; i5 < 2; i5++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 2; i3++) {
                acc += tables[3][i3*4 + i4*2 + i5] * t15[i3*2 + i4];
            }
            t16[i4*2 + i5] = acc;
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i5 = 0; i5 < 2; i5++) {
            t17[i4*2 + i5] = t16[i4*2 + i5] * t3[i4*2 + i5];
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 2; i5++) {
            acc += t17[i4*2 + i5];
        }
        t18[i4] = acc;
    }
    for (int i5 = 0; i5 < 2; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 2; i4++) {
            acc += t17[i4*2 + i5];
        }
        t19[i5] = acc;
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i6 = 0; i6 < 2; i6++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 2; i4++) {
                acc += tables[4][i4*4 + i5*2 + i6] * t16[i4*2 + i5];
            }
            t20[i5*2 + i6] = acc;
        }
    }
    for (int i6 = 0; i6 < 2; i6++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 2; i5++) {
            acc += t20[i5*2 + i6] * t2[i5*2 + i6];
        }
        t21[i6] = acc;
    }
    for (int i6 = 0; i6 < 2; i6++) {
        for (int i7 = 0; i7 < 2; i7++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 2; i5++) {
                acc += tables[5][i5*4 + i6*2 + i7] * t20[i5*2 + i6];
            }
            t22[i6*2 + i7] = acc;
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i8 = 0; i8 < 2; i8++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 2; i6++) {
                acc += tables[6][i6*4 + i7*2 + i8] * t22[i6*2 + i7];
            }
            t23[i7*2 + i8] = acc;
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i8 = 0; i8 < 2; i8++) {
            t24[i7*2 + i8] = t23[i7*2 + i8] * t0[i7*2 + i8];
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 2; i8++) {
            acc += t24[i7*2 + i8];
        }
        t25[i7] = acc;
    }
    for (int i8 = 0; i8 < 2; i8++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 2; i7++) {
            acc += t24[i7*2 + i8];
        }
        t26[i8] = acc;
    }
    for (int i9 = 0; i9 < 2; i9++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i8 = 0; i8 < 2; i8++) {
                acc += tables[7][i7*4 + i8*2 + i9] * t23[i7*2 + i8];
            }
        }
        t27[i9] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 2; j++) z += t8[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 2; j++) out[0 + j] = t8[j] * iz;
    for (int j = 0; j < 2; j++) out[2 + j] = t9[j] * iz;
    for (int j = 0; j < 2; j++) out[4 + j] = t13[j] * iz;
    for (int j = 0; j < 2; j++) out[6 + j] = t14[j] * iz;
    for (int j = 0; j < 2; j++) out[8 + j] = t18[j] * iz;
    for (int j = 0; j < 2; j++) out[10 + j] = t19[j] * iz;
    for (int j = 0; j < 2; j++) out[12 + j] = t21[j] * iz;
    for (int j = 0; j < 2; j++) out[14 + j] = t25[j] * iz;
    for (int j = 0; j < 2; j++) out[16 + j] = t26[j] * iz;
    for (int j = 0; j < 2; j++) out[18 + j] = t27[j] * iz;
}
