/* egfg program for lowrank_cycle6_k8_r1 (cost model: 8704 operations, overhead 64) */
#include <stddef.h>

static double t0[512];
static double t1[64];
static double t2[512];
static double t3[64];
static double t4[64];
static double t5[64];
static double t6[64];
static double t7[8];
static double t8[64];
static double t9[8];
static double t10[64];
static double t11[64];
static double t12[8];
static double t13[8];
static double t14[8];
static double t15[8];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i4 = 0; i4 < 8; i4++) {
            for (int i5 = 0; i5 < 8; i5++) {
                t0[i0*64 + i4*8 + i5] = tables[4][i4*8 + i5] * tables[5][i0 + i5*8];
            }
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i4 = 0; i4 < 8; i4++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 8; i5++) {
                acc += t0[i0*64 + i4*8 + i5];
            }
            t1[i0*8 + i4] = acc;
        }
    }
    for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
            for (int i4 = 0; i4 < 8; i4++) {
                t2[i2*64 + i3*8 + i4] = tables[2][i2*8 + i3] * tables[3][i3*8 + i4];
            }
        }
    }
    for (int i2 = 0; i2 < 8; i2++) {
        for (int i4 = 0; i4 < 8; i4++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 8; i3++) {
                acc += t2[i2*64 + i3*8 + i4];
            }
            t3[i2*8 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 8; i1++) {
        for (int i4 = 0; i4 < 8; i4++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 8; i2++) {
                acc += tables[1][i1*8 + i2] * t3[i2*8 + i4];
            }
            t4[i1*8 + i4] = acc;
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i4 = 0; i4 < 8; i4++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 8; i1++) {
                acc += tables[0][i0*8 + i1] * t4[i1*8 + i4];
            }
            t5[i0*8 + i4] = acc;
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i4 = 0; i4 < 8; i4++) {
            t6[i0*8 + i4] = t5[i0*8 + i4] * t1[i0*8 + i4];
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 8; i4++) {
            acc += t6[i0*8 + i4];
        }
        t7[i0] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        for (int i4 = 0; i4 < 8; i4++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 8; i0++) {
                acc += tables[0][i0*8 + i1] * t1[i0*8 + i4];
            }
            t8[i1*8 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 8; i1++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 8; i4++) {
            acc += t8[i1*8 + i4] * t4[i1*8 + i4];
        }
        t9[i1] = acc;
    }
    for (int i2 = 0; i2 < 8; i2++) {
        for (int i4 = 0; i4 < 8; i4++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 8; i1++) {
                acc += tables[1][i1*8 + i2] * t8[i1*8 + i4];
            }
            t10[i2*8 + i4] = acc;
        }
    }
    for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 8; i4++) {
                acc += t2[i2*64 + i3*8 + i4] * t10[i2*8 + i4];
            }
            t11[i2*8 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 8; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 8; i3++) {
            acc += t11[i2*8 + i3];
        }
        t12[i2] = acc;
    }
    for (int i3 = 0; i3 < 8; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 8; i2++) {
            acc += t11[i2*8 + i3];
        }
        t13[i3] = acc;
    }
    for (int i4 = 0; i4 < 8; i4++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 8; i0++) {
            acc += t6[i0*8 + i4];
        }
        t14[i4] = acc;
    }
    for (int i5 = 0; i5 < 8; i5++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 8; i0++) {
            for (int i4 = 0; i4 < 8; i4++) {
                acc += t0[i0*64 + i4*8 + i5] * t5[i0*8 + i4];
            }
        }
        t15[i5] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 8; j++) z += t7[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 8; j++) out[0 + j] = t7[j] * iz;
    for (int j = 0; j < 8; j++) out[8 + j] = t9[j] * iz;
    for (int j = 0; j < 8; j++) out[16 + j] = t12[j] * iz;
    for (int j = 0; j < 8; j++) out[24 + j] = t13[j] * iz;
    for (int j = 0; j < 8; j++) out[32 + j] = t14[j] * iz;
    for (int j = 0; j < 8; j++) out[40 + j] = t15[j] * iz;
}
