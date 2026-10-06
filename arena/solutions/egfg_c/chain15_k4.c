/* egfg program for chain15_k4 (cost model: 1052 operations, overhead 0) */
#include <stddef.h>
#include <math.h>

static double t0[16];
static double t1[4];
static double t2[4];
static double t3[4];
static double t4[4];
static double t5[4];
static double t6[4];
static double t7[4];
static double t8[4];
static double t9[4];
static double t10[4];
static double t11[4];
static double t12[4];
static double t13[4];
static double t14[4];
static double t15[4];
static double t16[4];
static double t17[4];
static double t18[4];
static double t19[4];
static double t20[4];
static double t21[4];
static double t22[4];
static double t23[4];
static double t24[4];
static double t25[4];
static double t26[4];
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
static double t56[4];
static double t57[4];
static double t58[4];
static double t59[4];
static double t60[4];
static double t61[4];
static double t62[4];
static double t63[4];
static double t64[4];
static double t65[4];
static double t66[4];
static double t67[4];
static double t68[4];

void infer(const double *const *tables, double *out) {
    for (int i5 = 0; i5 < 4; i5++) {
        for (int i6 = 0; i6 < 4; i6++) {
            t0[i5*4 + i6] = tables[13][i5*4 + i6] * tables[28][i6];
        }
    }
    for (int i5 = 0; i5 < 4; i5++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 4; i6++) {
            acc += t0[i5*4 + i6];
        }
        t1[i5] = acc;
    }
    for (int i5 = 0; i5 < 4; i5++) {
        t2[i5] = tables[27][i5] * t1[i5];
    }
    for (int i4 = 0; i4 < 4; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 4; i5++) {
            acc += tables[12][i4*4 + i5] * t2[i5];
        }
        t3[i4] = acc;
    }
    for (int i4 = 0; i4 < 4; i4++) {
        t4[i4] = tables[26][i4] * t3[i4];
    }
    for (int i3 = 0; i3 < 4; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 4; i4++) {
            acc += tables[11][i3*4 + i4] * t4[i4];
        }
        t5[i3] = acc;
    }
    for (int i3 = 0; i3 < 4; i3++) {
        t6[i3] = tables[25][i3] * t5[i3];
    }
    for (int i2 = 0; i2 < 4; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            acc += tables[10][i2*4 + i3] * t6[i3];
        }
        t7[i2] = acc;
    }
    for (int i2 = 0; i2 < 4; i2++) {
        t8[i2] = tables[24][i2] * t7[i2];
    }
    for (int i14 = 0; i14 < 4; i14++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 4; i2++) {
            acc += tables[9][i2 + i14*4] * t8[i2];
        }
        t9[i14] = acc;
    }
    for (int i14 = 0; i14 < 4; i14++) {
        t10[i14] = tables[23][i14] * t9[i14];
    }
    for (int i13 = 0; i13 < 4; i13++) {
        double acc = 0.0;
        for (int i14 = 0; i14 < 4; i14++) {
            acc += tables[8][i13*4 + i14] * t10[i14];
        }
        t11[i13] = acc;
    }
    for (int i13 = 0; i13 < 4; i13++) {
        t12[i13] = tables[22][i13] * t11[i13];
    }
    for (int i12 = 0; i12 < 4; i12++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 4; i13++) {
            acc += tables[7][i12*4 + i13] * t12[i13];
        }
        t13[i12] = acc;
    }
    for (int i12 = 0; i12 < 4; i12++) {
        t14[i12] = tables[21][i12] * t13[i12];
    }
    for (int i11 = 0; i11 < 4; i11++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 4; i12++) {
            acc += tables[6][i11*4 + i12] * t14[i12];
        }
        t15[i11] = acc;
    }
    for (int i11 = 0; i11 < 4; i11++) {
        t16[i11] = tables[20][i11] * t15[i11];
    }
    for (int i10 = 0; i10 < 4; i10++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 4; i11++) {
            acc += tables[5][i10*4 + i11] * t16[i11];
        }
        t17[i10] = acc;
    }
    for (int i10 = 0; i10 < 4; i10++) {
        t18[i10] = tables[19][i10] * t17[i10];
    }
    for (int i9 = 0; i9 < 4; i9++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 4; i10++) {
            acc += tables[4][i9*4 + i10] * t18[i10];
        }
        t19[i9] = acc;
    }
    for (int i9 = 0; i9 < 4; i9++) {
        t20[i9] = tables[18][i9] * t19[i9];
    }
    for (int i8 = 0; i8 < 4; i8++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 4; i9++) {
            acc += tables[3][i8*4 + i9] * t20[i9];
        }
        t21[i8] = acc;
    }
    for (int i8 = 0; i8 < 4; i8++) {
        t22[i8] = tables[17][i8] * t21[i8];
    }
    for (int i7 = 0; i7 < 4; i7++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 4; i8++) {
            acc += tables[2][i7*4 + i8] * t22[i8];
        }
        t23[i7] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        t24[i7] = tables[16][i7] * t23[i7];
    }
    for (int i1 = 0; i1 < 4; i1++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 4; i7++) {
            acc += tables[1][i1*4 + i7] * t24[i7];
        }
        t25[i1] = acc;
    }
    for (int i1 = 0; i1 < 4; i1++) {
        t26[i1] = tables[15][i1] * t25[i1];
    }
    for (int i0 = 0; i0 < 4; i0++) {
        for (int i1 = 0; i1 < 4; i1++) {
            t27[i0*4 + i1] = tables[0][i0*4 + i1] * tables[14][i0];
        }
    }
    for (int i0 = 0; i0 < 4; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            acc += t27[i0*4 + i1] * t26[i1];
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
        t30[i1] = tables[15][i1] * t29[i1];
    }
    for (int i1 = 0; i1 < 4; i1++) {
        t31[i1] = t25[i1] * t30[i1];
    }
    for (int i7 = 0; i7 < 4; i7++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 4; i1++) {
            acc += tables[1][i1*4 + i7] * t30[i1];
        }
        t32[i7] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        t33[i7] = tables[16][i7] * t32[i7];
    }
    for (int i8 = 0; i8 < 4; i8++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 4; i7++) {
            acc += tables[2][i7*4 + i8] * t33[i7];
        }
        t34[i8] = acc;
    }
    for (int i8 = 0; i8 < 4; i8++) {
        t35[i8] = tables[17][i8] * t34[i8];
    }
    for (int i9 = 0; i9 < 4; i9++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 4; i8++) {
            acc += tables[3][i8*4 + i9] * t35[i8];
        }
        t36[i9] = acc;
    }
    for (int i9 = 0; i9 < 4; i9++) {
        t37[i9] = tables[18][i9] * t36[i9];
    }
    for (int i10 = 0; i10 < 4; i10++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 4; i9++) {
            acc += tables[4][i9*4 + i10] * t37[i9];
        }
        t38[i10] = acc;
    }
    for (int i10 = 0; i10 < 4; i10++) {
        t39[i10] = tables[19][i10] * t38[i10];
    }
    for (int i11 = 0; i11 < 4; i11++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 4; i10++) {
            acc += tables[5][i10*4 + i11] * t39[i10];
        }
        t40[i11] = acc;
    }
    for (int i11 = 0; i11 < 4; i11++) {
        t41[i11] = tables[20][i11] * t40[i11];
    }
    for (int i12 = 0; i12 < 4; i12++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 4; i11++) {
            acc += tables[6][i11*4 + i12] * t41[i11];
        }
        t42[i12] = acc;
    }
    for (int i12 = 0; i12 < 4; i12++) {
        t43[i12] = tables[21][i12] * t42[i12];
    }
    for (int i13 = 0; i13 < 4; i13++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 4; i12++) {
            acc += tables[7][i12*4 + i13] * t43[i12];
        }
        t44[i13] = acc;
    }
    for (int i13 = 0; i13 < 4; i13++) {
        t45[i13] = tables[22][i13] * t44[i13];
    }
    for (int i14 = 0; i14 < 4; i14++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 4; i13++) {
            acc += tables[8][i13*4 + i14] * t45[i13];
        }
        t46[i14] = acc;
    }
    for (int i14 = 0; i14 < 4; i14++) {
        t47[i14] = tables[23][i14] * t46[i14];
    }
    for (int i2 = 0; i2 < 4; i2++) {
        double acc = 0.0;
        for (int i14 = 0; i14 < 4; i14++) {
            acc += tables[9][i2 + i14*4] * t47[i14];
        }
        t48[i2] = acc;
    }
    for (int i2 = 0; i2 < 4; i2++) {
        t49[i2] = t48[i2] * t8[i2];
    }
    for (int i2 = 0; i2 < 4; i2++) {
        t50[i2] = tables[24][i2] * t48[i2];
    }
    for (int i3 = 0; i3 < 4; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 4; i2++) {
            acc += tables[10][i2*4 + i3] * t50[i2];
        }
        t51[i3] = acc;
    }
    for (int i3 = 0; i3 < 4; i3++) {
        t52[i3] = t51[i3] * t6[i3];
    }
    for (int i3 = 0; i3 < 4; i3++) {
        t53[i3] = tables[25][i3] * t51[i3];
    }
    for (int i4 = 0; i4 < 4; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 4; i3++) {
            acc += tables[11][i3*4 + i4] * t53[i3];
        }
        t54[i4] = acc;
    }
    for (int i4 = 0; i4 < 4; i4++) {
        t55[i4] = t54[i4] * t4[i4];
    }
    for (int i4 = 0; i4 < 4; i4++) {
        t56[i4] = tables[26][i4] * t54[i4];
    }
    for (int i5 = 0; i5 < 4; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 4; i4++) {
            acc += tables[12][i4*4 + i5] * t56[i4];
        }
        t57[i5] = acc;
    }
    for (int i5 = 0; i5 < 4; i5++) {
        t58[i5] = t57[i5] * t2[i5];
    }
    for (int i5 = 0; i5 < 4; i5++) {
        t59[i5] = t57[i5] * tables[27][i5];
    }
    for (int i6 = 0; i6 < 4; i6++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 4; i5++) {
            acc += t59[i5] * t0[i5*4 + i6];
        }
        t60[i6] = acc;
    }
    for (int i7 = 0; i7 < 4; i7++) {
        t61[i7] = t23[i7] * t33[i7];
    }
    for (int i8 = 0; i8 < 4; i8++) {
        t62[i8] = t21[i8] * t35[i8];
    }
    for (int i9 = 0; i9 < 4; i9++) {
        t63[i9] = t19[i9] * t37[i9];
    }
    for (int i10 = 0; i10 < 4; i10++) {
        t64[i10] = t17[i10] * t39[i10];
    }
    for (int i11 = 0; i11 < 4; i11++) {
        t65[i11] = t15[i11] * t41[i11];
    }
    for (int i12 = 0; i12 < 4; i12++) {
        t66[i12] = t42[i12] * t14[i12];
    }
    for (int i13 = 0; i13 < 4; i13++) {
        t67[i13] = t44[i13] * t12[i13];
    }
    for (int i14 = 0; i14 < 4; i14++) {
        t68[i14] = t9[i14] * t47[i14];
    }
    double z = 0.0;
    for (int j = 0; j < 4; j++) z += t28[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 4; j++) out[0 + j] = t28[j] * iz;
    for (int j = 0; j < 4; j++) out[4 + j] = t31[j] * iz;
    for (int j = 0; j < 4; j++) out[8 + j] = t49[j] * iz;
    for (int j = 0; j < 4; j++) out[12 + j] = t52[j] * iz;
    for (int j = 0; j < 4; j++) out[16 + j] = t55[j] * iz;
    for (int j = 0; j < 4; j++) out[20 + j] = t58[j] * iz;
    for (int j = 0; j < 4; j++) out[24 + j] = t60[j] * iz;
    for (int j = 0; j < 4; j++) out[28 + j] = t61[j] * iz;
    for (int j = 0; j < 4; j++) out[32 + j] = t62[j] * iz;
    for (int j = 0; j < 4; j++) out[36 + j] = t63[j] * iz;
    for (int j = 0; j < 4; j++) out[40 + j] = t64[j] * iz;
    for (int j = 0; j < 4; j++) out[44 + j] = t65[j] * iz;
    for (int j = 0; j < 4; j++) out[48 + j] = t66[j] * iz;
    for (int j = 0; j < 4; j++) out[52 + j] = t67[j] * iz;
    for (int j = 0; j < 4; j++) out[56 + j] = t68[j] * iz;
}
