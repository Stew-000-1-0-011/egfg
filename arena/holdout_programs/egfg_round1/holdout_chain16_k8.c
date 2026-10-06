/* egfg program for holdout_chain16_k8 (cost model: 4176 operations, overhead 0) */
#include <stddef.h>

static double t0[64];
static double t1[8];
static double t2[8];
static double t3[8];
static double t4[8];
static double t5[8];
static double t6[8];
static double t7[8];
static double t8[8];
static double t9[8];
static double t10[8];
static double t11[8];
static double t12[8];
static double t13[8];
static double t14[8];
static double t15[8];
static double t16[8];
static double t17[8];
static double t18[8];
static double t19[8];
static double t20[8];
static double t21[8];
static double t22[8];
static double t23[8];
static double t24[8];
static double t25[8];
static double t26[8];
static double t27[8];
static double t28[8];
static double t29[64];
static double t30[8];
static double t31[8];
static double t32[8];
static double t33[8];
static double t34[8];
static double t35[8];
static double t36[8];
static double t37[8];
static double t38[8];
static double t39[8];
static double t40[8];
static double t41[8];
static double t42[8];
static double t43[8];
static double t44[8];
static double t45[8];
static double t46[8];
static double t47[8];
static double t48[8];
static double t49[8];
static double t50[8];
static double t51[8];
static double t52[8];
static double t53[8];
static double t54[8];
static double t55[8];
static double t56[8];
static double t57[8];
static double t58[8];
static double t59[8];
static double t60[8];
static double t61[8];
static double t62[8];
static double t63[8];
static double t64[8];
static double t65[8];
static double t66[8];
static double t67[8];
static double t68[8];
static double t69[8];
static double t70[8];
static double t71[8];
static double t72[8];
static double t73[8];

void infer(const double *const *tables, double *out) {
    for (int i6 = 0; i6 < 8; i6++) {
        for (int i7 = 0; i7 < 8; i7++) {
            t0[i6*8 + i7] = tables[14][i6*8 + i7] * tables[30][i7];
        }
    }
    for (int i6 = 0; i6 < 8; i6++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 8; i7++) {
            acc += t0[i6*8 + i7];
        }
        t1[i6] = acc;
    }
    for (int i6 = 0; i6 < 8; i6++) {
        t2[i6] = tables[29][i6] * t1[i6];
    }
    for (int i5 = 0; i5 < 8; i5++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 8; i6++) {
            acc += tables[13][i5*8 + i6] * t2[i6];
        }
        t3[i5] = acc;
    }
    for (int i5 = 0; i5 < 8; i5++) {
        t4[i5] = tables[28][i5] * t3[i5];
    }
    for (int i4 = 0; i4 < 8; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 8; i5++) {
            acc += tables[12][i4*8 + i5] * t4[i5];
        }
        t5[i4] = acc;
    }
    for (int i4 = 0; i4 < 8; i4++) {
        t6[i4] = tables[27][i4] * t5[i4];
    }
    for (int i3 = 0; i3 < 8; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 8; i4++) {
            acc += tables[11][i3*8 + i4] * t6[i4];
        }
        t7[i3] = acc;
    }
    for (int i3 = 0; i3 < 8; i3++) {
        t8[i3] = tables[26][i3] * t7[i3];
    }
    for (int i2 = 0; i2 < 8; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 8; i3++) {
            acc += tables[10][i2*8 + i3] * t8[i3];
        }
        t9[i2] = acc;
    }
    for (int i2 = 0; i2 < 8; i2++) {
        t10[i2] = tables[25][i2] * t9[i2];
    }
    for (int i15 = 0; i15 < 8; i15++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 8; i2++) {
            acc += tables[9][i2 + i15*8] * t10[i2];
        }
        t11[i15] = acc;
    }
    for (int i15 = 0; i15 < 8; i15++) {
        t12[i15] = tables[24][i15] * t11[i15];
    }
    for (int i14 = 0; i14 < 8; i14++) {
        double acc = 0.0;
        for (int i15 = 0; i15 < 8; i15++) {
            acc += tables[8][i14*8 + i15] * t12[i15];
        }
        t13[i14] = acc;
    }
    for (int i14 = 0; i14 < 8; i14++) {
        t14[i14] = tables[23][i14] * t13[i14];
    }
    for (int i13 = 0; i13 < 8; i13++) {
        double acc = 0.0;
        for (int i14 = 0; i14 < 8; i14++) {
            acc += tables[7][i13*8 + i14] * t14[i14];
        }
        t15[i13] = acc;
    }
    for (int i13 = 0; i13 < 8; i13++) {
        t16[i13] = tables[22][i13] * t15[i13];
    }
    for (int i12 = 0; i12 < 8; i12++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 8; i13++) {
            acc += tables[6][i12*8 + i13] * t16[i13];
        }
        t17[i12] = acc;
    }
    for (int i12 = 0; i12 < 8; i12++) {
        t18[i12] = tables[21][i12] * t17[i12];
    }
    for (int i11 = 0; i11 < 8; i11++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 8; i12++) {
            acc += tables[5][i11*8 + i12] * t18[i12];
        }
        t19[i11] = acc;
    }
    for (int i11 = 0; i11 < 8; i11++) {
        t20[i11] = tables[20][i11] * t19[i11];
    }
    for (int i10 = 0; i10 < 8; i10++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 8; i11++) {
            acc += tables[4][i10*8 + i11] * t20[i11];
        }
        t21[i10] = acc;
    }
    for (int i10 = 0; i10 < 8; i10++) {
        t22[i10] = tables[19][i10] * t21[i10];
    }
    for (int i9 = 0; i9 < 8; i9++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 8; i10++) {
            acc += tables[3][i9*8 + i10] * t22[i10];
        }
        t23[i9] = acc;
    }
    for (int i9 = 0; i9 < 8; i9++) {
        t24[i9] = tables[18][i9] * t23[i9];
    }
    for (int i8 = 0; i8 < 8; i8++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 8; i9++) {
            acc += tables[2][i8*8 + i9] * t24[i9];
        }
        t25[i8] = acc;
    }
    for (int i8 = 0; i8 < 8; i8++) {
        t26[i8] = tables[17][i8] * t25[i8];
    }
    for (int i1 = 0; i1 < 8; i1++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 8; i8++) {
            acc += tables[1][i1*8 + i8] * t26[i8];
        }
        t27[i1] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        t28[i1] = tables[16][i1] * t27[i1];
    }
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i1 = 0; i1 < 8; i1++) {
            t29[i0*8 + i1] = tables[0][i0*8 + i1] * tables[15][i0];
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 8; i1++) {
            acc += t29[i0*8 + i1] * t28[i1];
        }
        t30[i0] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 8; i0++) {
            acc += t29[i0*8 + i1];
        }
        t31[i1] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        t32[i1] = tables[16][i1] * t31[i1];
    }
    for (int i1 = 0; i1 < 8; i1++) {
        t33[i1] = t27[i1] * t32[i1];
    }
    for (int i8 = 0; i8 < 8; i8++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 8; i1++) {
            acc += tables[1][i1*8 + i8] * t32[i1];
        }
        t34[i8] = acc;
    }
    for (int i8 = 0; i8 < 8; i8++) {
        t35[i8] = tables[17][i8] * t34[i8];
    }
    for (int i9 = 0; i9 < 8; i9++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 8; i8++) {
            acc += tables[2][i8*8 + i9] * t35[i8];
        }
        t36[i9] = acc;
    }
    for (int i9 = 0; i9 < 8; i9++) {
        t37[i9] = tables[18][i9] * t36[i9];
    }
    for (int i10 = 0; i10 < 8; i10++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 8; i9++) {
            acc += tables[3][i9*8 + i10] * t37[i9];
        }
        t38[i10] = acc;
    }
    for (int i10 = 0; i10 < 8; i10++) {
        t39[i10] = tables[19][i10] * t38[i10];
    }
    for (int i11 = 0; i11 < 8; i11++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 8; i10++) {
            acc += tables[4][i10*8 + i11] * t39[i10];
        }
        t40[i11] = acc;
    }
    for (int i11 = 0; i11 < 8; i11++) {
        t41[i11] = tables[20][i11] * t40[i11];
    }
    for (int i12 = 0; i12 < 8; i12++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 8; i11++) {
            acc += tables[5][i11*8 + i12] * t41[i11];
        }
        t42[i12] = acc;
    }
    for (int i12 = 0; i12 < 8; i12++) {
        t43[i12] = tables[21][i12] * t42[i12];
    }
    for (int i13 = 0; i13 < 8; i13++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 8; i12++) {
            acc += tables[6][i12*8 + i13] * t43[i12];
        }
        t44[i13] = acc;
    }
    for (int i13 = 0; i13 < 8; i13++) {
        t45[i13] = tables[22][i13] * t44[i13];
    }
    for (int i14 = 0; i14 < 8; i14++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 8; i13++) {
            acc += tables[7][i13*8 + i14] * t45[i13];
        }
        t46[i14] = acc;
    }
    for (int i14 = 0; i14 < 8; i14++) {
        t47[i14] = tables[23][i14] * t46[i14];
    }
    for (int i15 = 0; i15 < 8; i15++) {
        double acc = 0.0;
        for (int i14 = 0; i14 < 8; i14++) {
            acc += tables[8][i14*8 + i15] * t47[i14];
        }
        t48[i15] = acc;
    }
    for (int i15 = 0; i15 < 8; i15++) {
        t49[i15] = tables[24][i15] * t48[i15];
    }
    for (int i2 = 0; i2 < 8; i2++) {
        double acc = 0.0;
        for (int i15 = 0; i15 < 8; i15++) {
            acc += tables[9][i2 + i15*8] * t49[i15];
        }
        t50[i2] = acc;
    }
    for (int i2 = 0; i2 < 8; i2++) {
        t51[i2] = t50[i2] * t10[i2];
    }
    for (int i2 = 0; i2 < 8; i2++) {
        t52[i2] = tables[25][i2] * t50[i2];
    }
    for (int i3 = 0; i3 < 8; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 8; i2++) {
            acc += tables[10][i2*8 + i3] * t52[i2];
        }
        t53[i3] = acc;
    }
    for (int i3 = 0; i3 < 8; i3++) {
        t54[i3] = t53[i3] * t8[i3];
    }
    for (int i3 = 0; i3 < 8; i3++) {
        t55[i3] = tables[26][i3] * t53[i3];
    }
    for (int i4 = 0; i4 < 8; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 8; i3++) {
            acc += tables[11][i3*8 + i4] * t55[i3];
        }
        t56[i4] = acc;
    }
    for (int i4 = 0; i4 < 8; i4++) {
        t57[i4] = t56[i4] * t6[i4];
    }
    for (int i4 = 0; i4 < 8; i4++) {
        t58[i4] = tables[27][i4] * t56[i4];
    }
    for (int i5 = 0; i5 < 8; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 8; i4++) {
            acc += tables[12][i4*8 + i5] * t58[i4];
        }
        t59[i5] = acc;
    }
    for (int i5 = 0; i5 < 8; i5++) {
        t60[i5] = t59[i5] * t4[i5];
    }
    for (int i5 = 0; i5 < 8; i5++) {
        t61[i5] = tables[28][i5] * t59[i5];
    }
    for (int i6 = 0; i6 < 8; i6++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 8; i5++) {
            acc += tables[13][i5*8 + i6] * t61[i5];
        }
        t62[i6] = acc;
    }
    for (int i6 = 0; i6 < 8; i6++) {
        t63[i6] = t62[i6] * t2[i6];
    }
    for (int i6 = 0; i6 < 8; i6++) {
        t64[i6] = t62[i6] * tables[29][i6];
    }
    for (int i7 = 0; i7 < 8; i7++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 8; i6++) {
            acc += t64[i6] * t0[i6*8 + i7];
        }
        t65[i7] = acc;
    }
    for (int i8 = 0; i8 < 8; i8++) {
        t66[i8] = t25[i8] * t35[i8];
    }
    for (int i9 = 0; i9 < 8; i9++) {
        t67[i9] = t23[i9] * t37[i9];
    }
    for (int i10 = 0; i10 < 8; i10++) {
        t68[i10] = t21[i10] * t39[i10];
    }
    for (int i11 = 0; i11 < 8; i11++) {
        t69[i11] = t19[i11] * t41[i11];
    }
    for (int i12 = 0; i12 < 8; i12++) {
        t70[i12] = t17[i12] * t43[i12];
    }
    for (int i13 = 0; i13 < 8; i13++) {
        t71[i13] = t44[i13] * t16[i13];
    }
    for (int i14 = 0; i14 < 8; i14++) {
        t72[i14] = t46[i14] * t14[i14];
    }
    for (int i15 = 0; i15 < 8; i15++) {
        t73[i15] = t48[i15] * t12[i15];
    }
    double z = 0.0;
    for (int j = 0; j < 8; j++) z += t30[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 8; j++) out[0 + j] = t30[j] * iz;
    for (int j = 0; j < 8; j++) out[8 + j] = t33[j] * iz;
    for (int j = 0; j < 8; j++) out[16 + j] = t51[j] * iz;
    for (int j = 0; j < 8; j++) out[24 + j] = t54[j] * iz;
    for (int j = 0; j < 8; j++) out[32 + j] = t57[j] * iz;
    for (int j = 0; j < 8; j++) out[40 + j] = t60[j] * iz;
    for (int j = 0; j < 8; j++) out[48 + j] = t63[j] * iz;
    for (int j = 0; j < 8; j++) out[56 + j] = t65[j] * iz;
    for (int j = 0; j < 8; j++) out[64 + j] = t66[j] * iz;
    for (int j = 0; j < 8; j++) out[72 + j] = t67[j] * iz;
    for (int j = 0; j < 8; j++) out[80 + j] = t68[j] * iz;
    for (int j = 0; j < 8; j++) out[88 + j] = t69[j] * iz;
    for (int j = 0; j < 8; j++) out[96 + j] = t70[j] * iz;
    for (int j = 0; j < 8; j++) out[104 + j] = t71[j] * iz;
    for (int j = 0; j < 8; j++) out[112 + j] = t72[j] * iz;
    for (int j = 0; j < 8; j++) out[120 + j] = t73[j] * iz;
}
