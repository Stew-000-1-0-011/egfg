/* egfg program for chain15_k4 (cost model: 1156 operations, overhead 512) */
#include <stddef.h>

static double t0[16];
static double t1[16];
static double t2[4];
static double t3[16];
static double t4[4];
static double t5[16];
static double t6[4];
static double t7[16];
static double t8[4];
static double t9[4];
static double t10[16];
static double t11[4];
static double t12[16];
static double t13[4];
static double t14[16];
static double t15[4];
static double t16[16];
static double t17[4];
static double t18[16];
static double t19[4];
static double t20[16];
static double t21[4];
static double t22[16];
static double t23[4];
static double t24[16];
static double t25[4];
static double t26[16];
static double t27[16];
static double t28[4];
static double t29[4];
static double t30[4];
static double t31[4];
static double t32[4];
static double t33[4];
static double t34[4];
static double t35[4];
static double t36[4];
static double t37[4];
static double t38[4];
static double t39[4];
static double t40[4];
static double t41[4];
static double t42[4];
static double t43[4];
static double t44[4];
static double t45[4];
static double t46[4];
static double t47[4];
static double t48[4];
static double t49[4];
static double t50[4];
static double t51[4];
static double t52[4];
static double t53[4];
static double t54[4];
static double t55[4];

void infer(const double *const *tables, double *out) {
    for (int i5 = 0; i5 < 4; i5++) {
        for (int i6 = 0; i6 < 4; i6++) {
            t0[i5*4 + i6] = tables[13][i5*4 + i6] * tables[27][i5];
        }
    }
    for (int i5 = 0; i5 < 4; i5++) {
        for (int i6 = 0; i6 < 4; i6++) {
            t1[i5*4 + i6] = t0[i5*4 + i6] * tables[28][i6];
        }
    }
    for (int i5 = 0; i5 < 4; i5++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 4; i6++) {
            acc += t1[i5*4 + i6];
        }
        t2[i5] = acc;
    }
    for (int i4 = 0; i4 < 4; i4++) {
        for (int i5 = 0; i5 < 4; i5++) {
            t3[i4*4 + i5] = tables[12][i4*4 + i5] * tables[26][i4];
        }
    }
    for (int i4 = 0; i4 < 4; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 4; i5++) {
            acc += t3[i4*4 + i5] * t2[i5];
        }
        t4[i4] = acc;
    }
    for (int i3 = 0; i3 < 4; i3++) {
        for (int i4 = 0; i4 < 4; i4++) {
            t5[i3*4 + i4] = tables[11][i3*4 + i4] * tables[25][i3];
        }
    }
    for (int i3 = 0; i3 < 4; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 4; i4++) {
            acc += t5[i3*4 + i4] * t4[i4];
        }
        t6[i3] = acc;
    }
    for (int i2 = 0; i2 < 4; i2++) {
        for (int i3 = 0; i3 < 4; i3++) {
            t7[i2*4 + i3] = tables[10][i2*4 + i3] * tables[24][i2];
        }
    }
    for (int i2 = 0; i2 < 4; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            acc += t7[i2*4 + i3] * t6[i3];
        }
        t8[i2] = acc;
    }
    for (int i14 = 0; i14 < 4; i14++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 4; i2++) {
            acc += tables[9][i2 + i14*4] * t8[i2];
        }
        t9[i14] = acc;
    }
    for (int i13 = 0; i13 < 4; i13++) {
        for (int i14 = 0; i14 < 4; i14++) {
            t10[i13*4 + i14] = tables[8][i13*4 + i14] * tables[23][i14];
        }
    }
    for (int i13 = 0; i13 < 4; i13++) {
        double acc = 0.0;
        for (int i14 = 0; i14 < 4; i14++) {
            acc += t10[i13*4 + i14] * t9[i14];
        }
        t11[i13] = acc;
    }
    for (int i12 = 0; i12 < 4; i12++) {
        for (int i13 = 0; i13 < 4; i13++) {
            t12[i12*4 + i13] = tables[7][i12*4 + i13] * tables[22][i13];
        }
    }
    for (int i12 = 0; i12 < 4; i12++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 4; i13++) {
            acc += t12[i12*4 + i13] * t11[i13];
        }
        t13[i12] = acc;
    }
    for (int i11 = 0; i11 < 4; i11++) {
        for (int i12 = 0; i12 < 4; i12++) {
            t14[i11*4 + i12] = tables[6][i11*4 + i12] * tables[21][i12];
        }
    }
    for (int i11 = 0; i11 < 4; i11++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 4; i12++) {
            acc += t14[i11*4 + i12] * t13[i12];
        }
        t15[i11] = acc;
    }
    for (int i10 = 0; i10 < 4; i10++) {
        for (int i11 = 0; i11 < 4; i11++) {
            t16[i10*4 + i11] = tables[5][i10*4 + i11] * tables[20][i11];
        }
    }
    for (int i10 = 0; i10 < 4; i10++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 4; i11++) {
            acc += t16[i10*4 + i11] * t15[i11];
        }
        t17[i10] = acc;
    }
    for (int i9 = 0; i9 < 4; i9++) {
        for (int i10 = 0; i10 < 4; i10++) {
            t18[i9*4 + i10] = tables[4][i9*4 + i10] * tables[19][i10];
        }
    }
    for (int i9 = 0; i9 < 4; i9++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 4; i10++) {
            acc += t18[i9*4 + i10] * t17[i10];
        }
        t19[i9] = acc;
    }
    for (int i8 = 0; i8 < 4; i8++) {
        for (int i9 = 0; i9 < 4; i9++) {
            t20[i8*4 + i9] = tables[3][i8*4 + i9] * tables[18][i9];
        }
    }
    for (int i8 = 0; i8 < 4; i8++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 4; i9++) {
            acc += t20[i8*4 + i9] * t19[i9];
        }
        t21[i8] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        for (int i8 = 0; i8 < 4; i8++) {
            t22[i7*4 + i8] = tables[2][i7*4 + i8] * tables[17][i8];
        }
    }
    for (int i7 = 0; i7 < 4; i7++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 4; i8++) {
            acc += t22[i7*4 + i8] * t21[i8];
        }
        t23[i7] = acc;
    }
    for (int i1 = 0; i1 < 4; i1++) {
        for (int i7 = 0; i7 < 4; i7++) {
            t24[i1*4 + i7] = tables[1][i1*4 + i7] * tables[16][i7];
        }
    }
    for (int i1 = 0; i1 < 4; i1++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 4; i7++) {
            acc += t24[i1*4 + i7] * t23[i7];
        }
        t25[i1] = acc;
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            t26[i0*4 + i1] = tables[0][i0*4 + i1] * tables[14][i0];
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            t27[i0*4 + i1] = t26[i0*4 + i1] * tables[15][i1];
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            acc += t27[i0*4 + i1] * t25[i1];
        }
        t28[i0] = acc;
    }
    for (int i1 = 0; i1 < 4; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 4; i0++) {
            acc += t27[i0*4 + i1];
        }
        t29[i1] = acc;
    }
    for (int i1 = 0; i1 < 4; i1++) {
        t30[i1] = t25[i1] * t29[i1];
    }
    for (int i7 = 0; i7 < 4; i7++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            acc += t24[i1*4 + i7] * t29[i1];
        }
        t31[i7] = acc;
    }
    for (int i8 = 0; i8 < 4; i8++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 4; i7++) {
            acc += t22[i7*4 + i8] * t31[i7];
        }
        t32[i8] = acc;
    }
    for (int i9 = 0; i9 < 4; i9++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 4; i8++) {
            acc += t20[i8*4 + i9] * t32[i8];
        }
        t33[i9] = acc;
    }
    for (int i10 = 0; i10 < 4; i10++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 4; i9++) {
            acc += t18[i9*4 + i10] * t33[i9];
        }
        t34[i10] = acc;
    }
    for (int i11 = 0; i11 < 4; i11++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 4; i10++) {
            acc += t16[i10*4 + i11] * t34[i10];
        }
        t35[i11] = acc;
    }
    for (int i12 = 0; i12 < 4; i12++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 4; i11++) {
            acc += t14[i11*4 + i12] * t35[i11];
        }
        t36[i12] = acc;
    }
    for (int i13 = 0; i13 < 4; i13++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 4; i12++) {
            acc += t12[i12*4 + i13] * t36[i12];
        }
        t37[i13] = acc;
    }
    for (int i14 = 0; i14 < 4; i14++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 4; i13++) {
            acc += t10[i13*4 + i14] * t37[i13];
        }
        t38[i14] = acc;
    }
    for (int i2 = 0; i2 < 4; i2++) {
        double acc = 0.0;
        for (int i14 = 0; i14 < 4; i14++) {
            acc += tables[9][i2 + i14*4] * t38[i14];
        }
        t39[i2] = acc;
    }
    for (int i2 = 0; i2 < 4; i2++) {
        t40[i2] = t39[i2] * t8[i2];
    }
    for (int i3 = 0; i3 < 4; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 4; i2++) {
            acc += t7[i2*4 + i3] * t39[i2];
        }
        t41[i3] = acc;
    }
    for (int i3 = 0; i3 < 4; i3++) {
        t42[i3] = t41[i3] * t6[i3];
    }
    for (int i4 = 0; i4 < 4; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            acc += t5[i3*4 + i4] * t41[i3];
        }
        t43[i4] = acc;
    }
    for (int i4 = 0; i4 < 4; i4++) {
        t44[i4] = t43[i4] * t4[i4];
    }
    for (int i5 = 0; i5 < 4; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 4; i4++) {
            acc += t3[i4*4 + i5] * t43[i4];
        }
        t45[i5] = acc;
    }
    for (int i5 = 0; i5 < 4; i5++) {
        t46[i5] = t45[i5] * t2[i5];
    }
    for (int i6 = 0; i6 < 4; i6++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 4; i5++) {
            acc += t1[i5*4 + i6] * t45[i5];
        }
        t47[i6] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        t48[i7] = t23[i7] * t31[i7];
    }
    for (int i8 = 0; i8 < 4; i8++) {
        t49[i8] = t21[i8] * t32[i8];
    }
    for (int i9 = 0; i9 < 4; i9++) {
        t50[i9] = t19[i9] * t33[i9];
    }
    for (int i10 = 0; i10 < 4; i10++) {
        t51[i10] = t17[i10] * t34[i10];
    }
    for (int i11 = 0; i11 < 4; i11++) {
        t52[i11] = t15[i11] * t35[i11];
    }
    for (int i12 = 0; i12 < 4; i12++) {
        t53[i12] = t13[i12] * t36[i12];
    }
    for (int i13 = 0; i13 < 4; i13++) {
        t54[i13] = t11[i13] * t37[i13];
    }
    for (int i14 = 0; i14 < 4; i14++) {
        t55[i14] = t9[i14] * t38[i14];
    }
    double z = 0.0;
    for (int j = 0; j < 4; j++) z += t28[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 4; j++) out[0 + j] = t28[j] * iz;
    for (int j = 0; j < 4; j++) out[4 + j] = t30[j] * iz;
    for (int j = 0; j < 4; j++) out[8 + j] = t40[j] * iz;
    for (int j = 0; j < 4; j++) out[12 + j] = t42[j] * iz;
    for (int j = 0; j < 4; j++) out[16 + j] = t44[j] * iz;
    for (int j = 0; j < 4; j++) out[20 + j] = t46[j] * iz;
    for (int j = 0; j < 4; j++) out[24 + j] = t47[j] * iz;
    for (int j = 0; j < 4; j++) out[28 + j] = t48[j] * iz;
    for (int j = 0; j < 4; j++) out[32 + j] = t49[j] * iz;
    for (int j = 0; j < 4; j++) out[36 + j] = t50[j] * iz;
    for (int j = 0; j < 4; j++) out[40 + j] = t51[j] * iz;
    for (int j = 0; j < 4; j++) out[44 + j] = t52[j] * iz;
    for (int j = 0; j < 4; j++) out[48 + j] = t53[j] * iz;
    for (int j = 0; j < 4; j++) out[52 + j] = t54[j] * iz;
    for (int j = 0; j < 4; j++) out[56 + j] = t55[j] * iz;
}
