/* egfg program for cycle8_k4 (cost model: 1936 operations, overhead 512) */
#include <stddef.h>

static double t0[64];
static double t1[16];
static double t2[64];
static double t3[16];
static double t4[16];
static double t5[64];
static double t6[16];
static double t7[16];
static double t8[64];
static double t9[16];
static double t10[4];
static double t11[4];
static double t12[16];
static double t13[16];
static double t14[16];
static double t15[4];
static double t16[16];
static double t17[16];
static double t18[4];
static double t19[4];
static double t20[16];
static double t21[16];
static double t22[4];
static double t23[4];
static double t24[4];

void infer(const double *const *tables, double *out) {
    for (int i1 = 0; i1 < 4; i1++) {
        for (int i2 = 0; i2 < 4; i2++) {
            for (int i3 = 0; i3 < 4; i3++) {
                t0[i1*16 + i2*4 + i3] = tables[1][i1*4 + i2] * tables[2][i2*4 + i3];
            }
        }
    }
    for (int i1 = 0; i1 < 4; i1++) {
        for (int i3 = 0; i3 < 4; i3++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 4; i2++) {
                acc += t0[i1*16 + i2*4 + i3];
            }
            t1[i1*4 + i3] = acc;
        }
    }
    for (int i3 = 0; i3 < 4; i3++) {
        for (int i4 = 0; i4 < 4; i4++) {
            for (int i5 = 0; i5 < 4; i5++) {
                t2[i3*16 + i4*4 + i5] = tables[3][i3*4 + i4] * tables[4][i4*4 + i5];
            }
        }
    }
    for (int i3 = 0; i3 < 4; i3++) {
        for (int i5 = 0; i5 < 4; i5++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 4; i4++) {
                acc += t2[i3*16 + i4*4 + i5];
            }
            t3[i3*4 + i5] = acc;
        }
    }
    for (int i1 = 0; i1 < 4; i1++) {
        for (int i5 = 0; i5 < 4; i5++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 4; i3++) {
                acc += t3[i3*4 + i5] * t1[i1*4 + i3];
            }
            t4[i1*4 + i5] = acc;
        }
    }
    for (int i5 = 0; i5 < 4; i5++) {
        for (int i6 = 0; i6 < 4; i6++) {
            for (int i7 = 0; i7 < 4; i7++) {
                t5[i5*16 + i6*4 + i7] = tables[5][i5*4 + i6] * tables[6][i6*4 + i7];
            }
        }
    }
    for (int i5 = 0; i5 < 4; i5++) {
        for (int i7 = 0; i7 < 4; i7++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 4; i6++) {
                acc += t5[i5*16 + i6*4 + i7];
            }
            t6[i5*4 + i7] = acc;
        }
    }
    for (int i1 = 0; i1 < 4; i1++) {
        for (int i7 = 0; i7 < 4; i7++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 4; i5++) {
                acc += t6[i5*4 + i7] * t4[i1*4 + i5];
            }
            t7[i1*4 + i7] = acc;
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            for (int i7 = 0; i7 < 4; i7++) {
                t8[i0*16 + i1*4 + i7] = tables[0][i0*4 + i1] * tables[7][i0 + i7*4];
            }
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 4; i7++) {
                acc += t8[i0*16 + i1*4 + i7] * t7[i1*4 + i7];
            }
            t9[i0*4 + i1] = acc;
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            acc += t9[i0*4 + i1];
        }
        t10[i0] = acc;
    }
    for (int i1 = 0; i1 < 4; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += t9[i0*4 + i1];
        }
        t11[i1] = acc;
    }
    for (int i1 = 0; i1 < 4; i1++) {
        for (int i7 = 0; i7 < 4; i7++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 4; i0++) {
                acc += t8[i0*16 + i1*4 + i7];
            }
            t12[i1*4 + i7] = acc;
        }
    }
    for (int i1 = 0; i1 < 4; i1++) {
        for (int i5 = 0; i5 < 4; i5++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 4; i7++) {
                acc += t6[i5*4 + i7] * t12[i1*4 + i7];
            }
            t13[i1*4 + i5] = acc;
        }
    }
    for (int i1 = 0; i1 < 4; i1++) {
        for (int i3 = 0; i3 < 4; i3++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 4; i5++) {
                acc += t3[i3*4 + i5] * t13[i1*4 + i5];
            }
            t14[i1*4 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 4; i2++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            for (int i3 = 0; i3 < 4; i3++) {
                acc += t0[i1*16 + i2*4 + i3] * t14[i1*4 + i3];
            }
        }
        t15[i2] = acc;
    }
    for (int i3 = 0; i3 < 4; i3++) {
        for (int i5 = 0; i5 < 4; i5++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 4; i1++) {
                acc += t1[i1*4 + i3] * t13[i1*4 + i5];
            }
            t16[i3*4 + i5] = acc;
        }
    }
    for (int i3 = 0; i3 < 4; i3++) {
        for (int i4 = 0; i4 < 4; i4++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 4; i5++) {
                acc += t2[i3*16 + i4*4 + i5] * t16[i3*4 + i5];
            }
            t17[i3*4 + i4] = acc;
        }
    }
    for (int i3 = 0; i3 < 4; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 4; i4++) {
            acc += t17[i3*4 + i4];
        }
        t18[i3] = acc;
    }
    for (int i4 = 0; i4 < 4; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            acc += t17[i3*4 + i4];
        }
        t19[i4] = acc;
    }
    for (int i5 = 0; i5 < 4; i5++) {
        for (int i7 = 0; i7 < 4; i7++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 4; i1++) {
                acc += t12[i1*4 + i7] * t4[i1*4 + i5];
            }
            t20[i5*4 + i7] = acc;
        }
    }
    for (int i5 = 0; i5 < 4; i5++) {
        for (int i6 = 0; i6 < 4; i6++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 4; i7++) {
                acc += t5[i5*16 + i6*4 + i7] * t20[i5*4 + i7];
            }
            t21[i5*4 + i6] = acc;
        }
    }
    for (int i5 = 0; i5 < 4; i5++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 4; i6++) {
            acc += t21[i5*4 + i6];
        }
        t22[i5] = acc;
    }
    for (int i6 = 0; i6 < 4; i6++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 4; i5++) {
            acc += t21[i5*4 + i6];
        }
        t23[i6] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            acc += t7[i1*4 + i7] * t12[i1*4 + i7];
        }
        t24[i7] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 4; j++) z += t10[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 4; j++) out[0 + j] = t10[j] * iz;
    for (int j = 0; j < 4; j++) out[4 + j] = t11[j] * iz;
    for (int j = 0; j < 4; j++) out[8 + j] = t15[j] * iz;
    for (int j = 0; j < 4; j++) out[12 + j] = t18[j] * iz;
    for (int j = 0; j < 4; j++) out[16 + j] = t19[j] * iz;
    for (int j = 0; j < 4; j++) out[20 + j] = t22[j] * iz;
    for (int j = 0; j < 4; j++) out[24 + j] = t23[j] * iz;
    for (int j = 0; j < 4; j++) out[28 + j] = t24[j] * iz;
}
