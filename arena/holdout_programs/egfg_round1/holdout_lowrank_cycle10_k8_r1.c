/* egfg program for holdout_lowrank_cycle10_k8_r1 (cost model: 19200 operations, overhead 64) */
#include <stddef.h>

static double t0[512];
static double t1[64];
static double t2[512];
static double t3[64];
static double t4[64];
static double t5[512];
static double t6[64];
static double t7[512];
static double t8[64];
static double t9[64];
static double t10[64];
static double t11[64];
static double t12[64];
static double t13[8];
static double t14[64];
static double t15[64];
static double t16[64];
static double t17[64];
static double t18[8];
static double t19[8];
static double t20[8];
static double t21[64];
static double t22[64];
static double t23[64];
static double t24[8];
static double t25[8];
static double t26[64];
static double t27[64];
static double t28[8];
static double t29[8];
static double t30[8];
static double t31[8];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i8 = 0; i8 < 8; i8++) {
            for (int i9 = 0; i9 < 8; i9++) {
                t0[i0*64 + i8*8 + i9] = tables[8][i8*8 + i9] * tables[9][i0 + i9*8];
            }
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i8 = 0; i8 < 8; i8++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 8; i9++) {
                acc += t0[i0*64 + i8*8 + i9];
            }
            t1[i0*8 + i8] = acc;
        }
    }
    for (int i1 = 0; i1 < 8; i1++) {
        for (int i2 = 0; i2 < 8; i2++) {
            for (int i3 = 0; i3 < 8; i3++) {
                t2[i1*64 + i2*8 + i3] = tables[1][i1*8 + i2] * tables[2][i2*8 + i3];
            }
        }
    }
    for (int i1 = 0; i1 < 8; i1++) {
        for (int i3 = 0; i3 < 8; i3++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 8; i2++) {
                acc += t2[i1*64 + i2*8 + i3];
            }
            t3[i1*8 + i3] = acc;
        }
    }
    for (int i1 = 0; i1 < 8; i1++) {
        for (int i4 = 0; i4 < 8; i4++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 8; i3++) {
                acc += tables[3][i3*8 + i4] * t3[i1*8 + i3];
            }
            t4[i1*8 + i4] = acc;
        }
    }
    for (int i6 = 0; i6 < 8; i6++) {
        for (int i7 = 0; i7 < 8; i7++) {
            for (int i8 = 0; i8 < 8; i8++) {
                t5[i6*64 + i7*8 + i8] = tables[6][i6*8 + i7] * tables[7][i7*8 + i8];
            }
        }
    }
    for (int i6 = 0; i6 < 8; i6++) {
        for (int i8 = 0; i8 < 8; i8++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 8; i7++) {
                acc += t5[i6*64 + i7*8 + i8];
            }
            t6[i6*8 + i8] = acc;
        }
    }
    for (int i4 = 0; i4 < 8; i4++) {
        for (int i5 = 0; i5 < 8; i5++) {
            for (int i6 = 0; i6 < 8; i6++) {
                t7[i4*64 + i5*8 + i6] = tables[4][i4*8 + i5] * tables[5][i5*8 + i6];
            }
        }
    }
    for (int i4 = 0; i4 < 8; i4++) {
        for (int i6 = 0; i6 < 8; i6++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 8; i5++) {
                acc += t7[i4*64 + i5*8 + i6];
            }
            t8[i4*8 + i6] = acc;
        }
    }
    for (int i4 = 0; i4 < 8; i4++) {
        for (int i8 = 0; i8 < 8; i8++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 8; i6++) {
                acc += t8[i4*8 + i6] * t6[i6*8 + i8];
            }
            t9[i4*8 + i8] = acc;
        }
    }
    for (int i1 = 0; i1 < 8; i1++) {
        for (int i8 = 0; i8 < 8; i8++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 8; i4++) {
                acc += t9[i4*8 + i8] * t4[i1*8 + i4];
            }
            t10[i1*8 + i8] = acc;
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i8 = 0; i8 < 8; i8++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 8; i1++) {
                acc += tables[0][i0*8 + i1] * t10[i1*8 + i8];
            }
            t11[i0*8 + i8] = acc;
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i8 = 0; i8 < 8; i8++) {
            t12[i0*8 + i8] = t11[i0*8 + i8] * t1[i0*8 + i8];
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 8; i8++) {
            acc += t12[i0*8 + i8];
        }
        t13[i0] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        for (int i8 = 0; i8 < 8; i8++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 8; i0++) {
                acc += tables[0][i0*8 + i1] * t1[i0*8 + i8];
            }
            t14[i1*8 + i8] = acc;
        }
    }
    for (int i1 = 0; i1 < 8; i1++) {
        for (int i4 = 0; i4 < 8; i4++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 8; i8++) {
                acc += t9[i4*8 + i8] * t14[i1*8 + i8];
            }
            t15[i1*8 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 8; i1++) {
        for (int i3 = 0; i3 < 8; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 8; i4++) {
                acc += tables[3][i3*8 + i4] * t15[i1*8 + i4];
            }
            t16[i1*8 + i3] = acc;
        }
    }
    for (int i1 = 0; i1 < 8; i1++) {
        for (int i2 = 0; i2 < 8; i2++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 8; i3++) {
                acc += t2[i1*64 + i2*8 + i3] * t16[i1*8 + i3];
            }
            t17[i1*8 + i2] = acc;
        }
    }
    for (int i1 = 0; i1 < 8; i1++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 8; i2++) {
            acc += t17[i1*8 + i2];
        }
        t18[i1] = acc;
    }
    for (int i2 = 0; i2 < 8; i2++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 8; i1++) {
            acc += t17[i1*8 + i2];
        }
        t19[i2] = acc;
    }
    for (int i3 = 0; i3 < 8; i3++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 8; i1++) {
            acc += t16[i1*8 + i3] * t3[i1*8 + i3];
        }
        t20[i3] = acc;
    }
    for (int i4 = 0; i4 < 8; i4++) {
        for (int i8 = 0; i8 < 8; i8++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 8; i1++) {
                acc += t14[i1*8 + i8] * t4[i1*8 + i4];
            }
            t21[i4*8 + i8] = acc;
        }
    }
    for (int i4 = 0; i4 < 8; i4++) {
        for (int i6 = 0; i6 < 8; i6++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 8; i8++) {
                acc += t6[i6*8 + i8] * t21[i4*8 + i8];
            }
            t22[i4*8 + i6] = acc;
        }
    }
    for (int i4 = 0; i4 < 8; i4++) {
        for (int i5 = 0; i5 < 8; i5++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 8; i6++) {
                acc += t7[i4*64 + i5*8 + i6] * t22[i4*8 + i6];
            }
            t23[i4*8 + i5] = acc;
        }
    }
    for (int i4 = 0; i4 < 8; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 8; i5++) {
            acc += t23[i4*8 + i5];
        }
        t24[i4] = acc;
    }
    for (int i5 = 0; i5 < 8; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 8; i4++) {
            acc += t23[i4*8 + i5];
        }
        t25[i5] = acc;
    }
    for (int i6 = 0; i6 < 8; i6++) {
        for (int i8 = 0; i8 < 8; i8++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 8; i4++) {
                acc += t8[i4*8 + i6] * t21[i4*8 + i8];
            }
            t26[i6*8 + i8] = acc;
        }
    }
    for (int i6 = 0; i6 < 8; i6++) {
        for (int i7 = 0; i7 < 8; i7++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 8; i8++) {
                acc += t5[i6*64 + i7*8 + i8] * t26[i6*8 + i8];
            }
            t27[i6*8 + i7] = acc;
        }
    }
    for (int i6 = 0; i6 < 8; i6++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 8; i7++) {
            acc += t27[i6*8 + i7];
        }
        t28[i6] = acc;
    }
    for (int i7 = 0; i7 < 8; i7++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 8; i6++) {
            acc += t27[i6*8 + i7];
        }
        t29[i7] = acc;
    }
    for (int i8 = 0; i8 < 8; i8++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 8; i0++) {
            acc += t12[i0*8 + i8];
        }
        t30[i8] = acc;
    }
    for (int i9 = 0; i9 < 8; i9++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 8; i0++) {
            for (int i8 = 0; i8 < 8; i8++) {
                acc += t0[i0*64 + i8*8 + i9] * t11[i0*8 + i8];
            }
        }
        t31[i9] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 8; j++) z += t13[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 8; j++) out[0 + j] = t13[j] * iz;
    for (int j = 0; j < 8; j++) out[8 + j] = t18[j] * iz;
    for (int j = 0; j < 8; j++) out[16 + j] = t19[j] * iz;
    for (int j = 0; j < 8; j++) out[24 + j] = t20[j] * iz;
    for (int j = 0; j < 8; j++) out[32 + j] = t24[j] * iz;
    for (int j = 0; j < 8; j++) out[40 + j] = t25[j] * iz;
    for (int j = 0; j < 8; j++) out[48 + j] = t28[j] * iz;
    for (int j = 0; j < 8; j++) out[56 + j] = t29[j] * iz;
    for (int j = 0; j < 8; j++) out[64 + j] = t30[j] * iz;
    for (int j = 0; j < 8; j++) out[72 + j] = t31[j] * iz;
}
