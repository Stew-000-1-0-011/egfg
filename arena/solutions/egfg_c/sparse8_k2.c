/* egfg program for sparse8_k2 (cost model: 254 operations, overhead 512) */
#include <stddef.h>

static double t0[8];
static double t1[4];
static double t2[8];
static double t3[4];
static double t4[2];
static double t5[4];
static double t6[4];
static double t7[8];
static double t8[4];
static double t9[8];
static double t10[4];
static double t11[2];
static double t12[4];
static double t13[8];
static double t14[4];
static double t15[2];
static double t16[4];
static double t17[4];
static double t18[8];
static double t19[4];
static double t20[2];
static double t21[2];
static double t22[2];
static double t23[2];
static double t24[2];
static double t25[2];
static double t26[2];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i1 = 0; i1 < 2; i1++) {
            for (int i3 = 0; i3 < 2; i3++) {
                t0[i0*4 + i1*2 + i3] = tables[0][i0*2 + i1] * tables[2][i1*2 + i3];
            }
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i3 = 0; i3 < 2; i3++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 2; i1++) {
                acc += t0[i0*4 + i1*2 + i3];
            }
            t1[i0*2 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            for (int i6 = 0; i6 < 2; i6++) {
                t2[i2*4 + i3*2 + i6] = tables[5][i2*2 + i6] * tables[7][i3*2 + i6];
            }
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 2; i6++) {
                acc += t2[i2*4 + i3*2 + i6];
            }
            t3[i2*2 + i3] = acc;
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 2; i4++) {
            acc += tables[3][i3*2 + i4];
        }
        t4[i3] = acc;
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            t5[i2*2 + i3] = t4[i3] * t3[i2*2 + i3];
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i3 = 0; i3 < 2; i3++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 2; i2++) {
                acc += tables[1][i0*2 + i2] * t5[i2*2 + i3];
            }
            t6[i0*2 + i3] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i3 = 0; i3 < 2; i3++) {
            for (int i7 = 0; i7 < 2; i7++) {
                t7[i0*4 + i3*2 + i7] = tables[6][i3*2 + i7] * t6[i0*2 + i3];
            }
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i7 = 0; i7 < 2; i7++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 2; i3++) {
                acc += t7[i0*4 + i3*2 + i7] * t1[i0*2 + i3];
            }
            t8[i0*2 + i7] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i5 = 0; i5 < 2; i5++) {
            for (int i7 = 0; i7 < 2; i7++) {
                t9[i0*4 + i5*2 + i7] = tables[4][i0*2 + i5] * tables[8][i5*2 + i7];
            }
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i5 = 0; i5 < 2; i5++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 2; i7++) {
                acc += t9[i0*4 + i5*2 + i7] * t8[i0*2 + i7];
            }
            t10[i0*2 + i5] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 2; i5++) {
            acc += t10[i0*2 + i5];
        }
        t11[i0] = acc;
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i7 = 0; i7 < 2; i7++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 2; i5++) {
                acc += t9[i0*4 + i5*2 + i7];
            }
            t12[i0*2 + i7] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i3 = 0; i3 < 2; i3++) {
            for (int i7 = 0; i7 < 2; i7++) {
                t13[i0*4 + i3*2 + i7] = tables[6][i3*2 + i7] * t12[i0*2 + i7];
            }
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i3 = 0; i3 < 2; i3++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 2; i7++) {
                acc += t13[i0*4 + i3*2 + i7] * t6[i0*2 + i3];
            }
            t14[i0*2 + i3] = acc;
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 2; i0++) {
            for (int i3 = 0; i3 < 2; i3++) {
                acc += t0[i0*4 + i1*2 + i3] * t14[i0*2 + i3];
            }
        }
        t15[i1] = acc;
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i3 = 0; i3 < 2; i3++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 2; i7++) {
                acc += t13[i0*4 + i3*2 + i7] * t1[i0*2 + i3];
            }
            t16[i0*2 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 2; i0++) {
                acc += tables[1][i0*2 + i2] * t16[i0*2 + i3];
            }
            t17[i2*2 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            for (int i6 = 0; i6 < 2; i6++) {
                t18[i2*4 + i3*2 + i6] = t2[i2*4 + i3*2 + i6] * t4[i3];
            }
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i6 = 0; i6 < 2; i6++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 2; i3++) {
                acc += t18[i2*4 + i3*2 + i6] * t17[i2*2 + i3];
            }
            t19[i2*2 + i6] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 2; i6++) {
            acc += t19[i2*2 + i6];
        }
        t20[i2] = acc;
    }
    for (int i3 = 0; i3 < 2; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 2; i2++) {
            acc += t17[i2*2 + i3] * t3[i2*2 + i3];
        }
        t21[i3] = acc;
    }
    for (int i3 = 0; i3 < 2; i3++) {
        t22[i3] = t21[i3] * t4[i3];
    }
    for (int i4 = 0; i4 < 2; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 2; i3++) {
            acc += tables[3][i3*2 + i4] * t21[i3];
        }
        t23[i4] = acc;
    }
    for (int i5 = 0; i5 < 2; i5++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 2; i0++) {
            acc += t10[i0*2 + i5];
        }
        t24[i5] = acc;
    }
    for (int i6 = 0; i6 < 2; i6++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 2; i2++) {
            acc += t19[i2*2 + i6];
        }
        t25[i6] = acc;
    }
    for (int i7 = 0; i7 < 2; i7++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 2; i0++) {
            acc += t8[i0*2 + i7] * t12[i0*2 + i7];
        }
        t26[i7] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 2; j++) z += t11[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 2; j++) out[0 + j] = t11[j] * iz;
    for (int j = 0; j < 2; j++) out[2 + j] = t15[j] * iz;
    for (int j = 0; j < 2; j++) out[4 + j] = t20[j] * iz;
    for (int j = 0; j < 2; j++) out[6 + j] = t22[j] * iz;
    for (int j = 0; j < 2; j++) out[8 + j] = t23[j] * iz;
    for (int j = 0; j < 2; j++) out[10 + j] = t24[j] * iz;
    for (int j = 0; j < 2; j++) out[12 + j] = t25[j] * iz;
    for (int j = 0; j < 2; j++) out[14 + j] = t26[j] * iz;
}
