/* egfg program for holdout_chain19_k8 (cost model: 5016 operations, overhead 512) */
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
static double t29[8];
static double t30[8];
static double t31[8];
static double t32[8];
static double t33[8];
static double t34[8];
static double t35[64];
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
static double t74[8];
static double t75[8];
static double t76[8];
static double t77[8];
static double t78[8];
static double t79[8];
static double t80[8];
static double t81[8];
static double t82[8];
static double t83[8];
static double t84[8];
static double t85[8];
static double t86[8];
static double t87[8];
static double t88[8];

void infer(const double *const *tables, double *out) {
    for (int i9 = 0; i9 < 8; i9++) {
        for (int i10 = 0; i10 < 8; i10++) {
            t0[i9*8 + i10] = tables[17][i9*8 + i10] * tables[36][i10];
        }
    }
    for (int i9 = 0; i9 < 8; i9++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 8; i10++) {
            acc += t0[i9*8 + i10];
        }
        t1[i9] = acc;
    }
    for (int i9 = 0; i9 < 8; i9++) {
        t2[i9] = tables[35][i9] * t1[i9];
    }
    for (int i8 = 0; i8 < 8; i8++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 8; i9++) {
            acc += tables[16][i8*8 + i9] * t2[i9];
        }
        t3[i8] = acc;
    }
    for (int i8 = 0; i8 < 8; i8++) {
        t4[i8] = tables[34][i8] * t3[i8];
    }
    for (int i7 = 0; i7 < 8; i7++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 8; i8++) {
            acc += tables[15][i7*8 + i8] * t4[i8];
        }
        t5[i7] = acc;
    }
    for (int i7 = 0; i7 < 8; i7++) {
        t6[i7] = tables[33][i7] * t5[i7];
    }
    for (int i6 = 0; i6 < 8; i6++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 8; i7++) {
            acc += tables[14][i6*8 + i7] * t6[i7];
        }
        t7[i6] = acc;
    }
    for (int i6 = 0; i6 < 8; i6++) {
        t8[i6] = tables[32][i6] * t7[i6];
    }
    for (int i5 = 0; i5 < 8; i5++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 8; i6++) {
            acc += tables[13][i5*8 + i6] * t8[i6];
        }
        t9[i5] = acc;
    }
    for (int i5 = 0; i5 < 8; i5++) {
        t10[i5] = tables[31][i5] * t9[i5];
    }
    for (int i4 = 0; i4 < 8; i4++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 8; i5++) {
            acc += tables[12][i4*8 + i5] * t10[i5];
        }
        t11[i4] = acc;
    }
    for (int i4 = 0; i4 < 8; i4++) {
        t12[i4] = tables[30][i4] * t11[i4];
    }
    for (int i3 = 0; i3 < 8; i3++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 8; i4++) {
            acc += tables[11][i3*8 + i4] * t12[i4];
        }
        t13[i3] = acc;
    }
    for (int i3 = 0; i3 < 8; i3++) {
        t14[i3] = tables[29][i3] * t13[i3];
    }
    for (int i2 = 0; i2 < 8; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 8; i3++) {
            acc += tables[10][i2*8 + i3] * t14[i3];
        }
        t15[i2] = acc;
    }
    for (int i2 = 0; i2 < 8; i2++) {
        t16[i2] = tables[28][i2] * t15[i2];
    }
    for (int i18 = 0; i18 < 8; i18++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 8; i2++) {
            acc += tables[9][i2 + i18*8] * t16[i2];
        }
        t17[i18] = acc;
    }
    for (int i18 = 0; i18 < 8; i18++) {
        t18[i18] = tables[27][i18] * t17[i18];
    }
    for (int i17 = 0; i17 < 8; i17++) {
        double acc = 0.0;
        for (int i18 = 0; i18 < 8; i18++) {
            acc += tables[8][i17*8 + i18] * t18[i18];
        }
        t19[i17] = acc;
    }
    for (int i17 = 0; i17 < 8; i17++) {
        t20[i17] = tables[26][i17] * t19[i17];
    }
    for (int i16 = 0; i16 < 8; i16++) {
        double acc = 0.0;
        for (int i17 = 0; i17 < 8; i17++) {
            acc += tables[7][i16*8 + i17] * t20[i17];
        }
        t21[i16] = acc;
    }
    for (int i16 = 0; i16 < 8; i16++) {
        t22[i16] = tables[25][i16] * t21[i16];
    }
    for (int i15 = 0; i15 < 8; i15++) {
        double acc = 0.0;
        for (int i16 = 0; i16 < 8; i16++) {
            acc += tables[6][i15*8 + i16] * t22[i16];
        }
        t23[i15] = acc;
    }
    for (int i15 = 0; i15 < 8; i15++) {
        t24[i15] = tables[24][i15] * t23[i15];
    }
    for (int i14 = 0; i14 < 8; i14++) {
        double acc = 0.0;
        for (int i15 = 0; i15 < 8; i15++) {
            acc += tables[5][i14*8 + i15] * t24[i15];
        }
        t25[i14] = acc;
    }
    for (int i14 = 0; i14 < 8; i14++) {
        t26[i14] = tables[23][i14] * t25[i14];
    }
    for (int i13 = 0; i13 < 8; i13++) {
        double acc = 0.0;
        for (int i14 = 0; i14 < 8; i14++) {
            acc += tables[4][i13*8 + i14] * t26[i14];
        }
        t27[i13] = acc;
    }
    for (int i13 = 0; i13 < 8; i13++) {
        t28[i13] = tables[22][i13] * t27[i13];
    }
    for (int i12 = 0; i12 < 8; i12++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 8; i13++) {
            acc += tables[3][i12*8 + i13] * t28[i13];
        }
        t29[i12] = acc;
    }
    for (int i12 = 0; i12 < 8; i12++) {
        t30[i12] = tables[21][i12] * t29[i12];
    }
    for (int i11 = 0; i11 < 8; i11++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 8; i12++) {
            acc += tables[2][i11*8 + i12] * t30[i12];
        }
        t31[i11] = acc;
    }
    for (int i11 = 0; i11 < 8; i11++) {
        t32[i11] = tables[20][i11] * t31[i11];
    }
    for (int i1 = 0; i1 < 8; i1++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 8; i11++) {
            acc += tables[1][i1*8 + i11] * t32[i11];
        }
        t33[i1] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        t34[i1] = tables[19][i1] * t33[i1];
    }
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i1 = 0; i1 < 8; i1++) {
            t35[i0*8 + i1] = tables[0][i0*8 + i1] * tables[18][i0];
        }
    }
    for (int i0 = 0; i0 < 8; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 8; i1++) {
            acc += t35[i0*8 + i1] * t34[i1];
        }
        t36[i0] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 8; i0++) {
            acc += t35[i0*8 + i1];
        }
        t37[i1] = acc;
    }
    for (int i1 = 0; i1 < 8; i1++) {
        t38[i1] = tables[19][i1] * t37[i1];
    }
    for (int i1 = 0; i1 < 8; i1++) {
        t39[i1] = t33[i1] * t38[i1];
    }
    for (int i11 = 0; i11 < 8; i11++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 8; i1++) {
            acc += tables[1][i1*8 + i11] * t38[i1];
        }
        t40[i11] = acc;
    }
    for (int i11 = 0; i11 < 8; i11++) {
        t41[i11] = tables[20][i11] * t40[i11];
    }
    for (int i12 = 0; i12 < 8; i12++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 8; i11++) {
            acc += tables[2][i11*8 + i12] * t41[i11];
        }
        t42[i12] = acc;
    }
    for (int i12 = 0; i12 < 8; i12++) {
        t43[i12] = tables[21][i12] * t42[i12];
    }
    for (int i13 = 0; i13 < 8; i13++) {
        double acc = 0.0;
        for (int i12 = 0; i12 < 8; i12++) {
            acc += tables[3][i12*8 + i13] * t43[i12];
        }
        t44[i13] = acc;
    }
    for (int i13 = 0; i13 < 8; i13++) {
        t45[i13] = tables[22][i13] * t44[i13];
    }
    for (int i14 = 0; i14 < 8; i14++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 8; i13++) {
            acc += tables[4][i13*8 + i14] * t45[i13];
        }
        t46[i14] = acc;
    }
    for (int i14 = 0; i14 < 8; i14++) {
        t47[i14] = tables[23][i14] * t46[i14];
    }
    for (int i15 = 0; i15 < 8; i15++) {
        double acc = 0.0;
        for (int i14 = 0; i14 < 8; i14++) {
            acc += tables[5][i14*8 + i15] * t47[i14];
        }
        t48[i15] = acc;
    }
    for (int i15 = 0; i15 < 8; i15++) {
        t49[i15] = tables[24][i15] * t48[i15];
    }
    for (int i16 = 0; i16 < 8; i16++) {
        double acc = 0.0;
        for (int i15 = 0; i15 < 8; i15++) {
            acc += tables[6][i15*8 + i16] * t49[i15];
        }
        t50[i16] = acc;
    }
    for (int i16 = 0; i16 < 8; i16++) {
        t51[i16] = tables[25][i16] * t50[i16];
    }
    for (int i17 = 0; i17 < 8; i17++) {
        double acc = 0.0;
        for (int i16 = 0; i16 < 8; i16++) {
            acc += tables[7][i16*8 + i17] * t51[i16];
        }
        t52[i17] = acc;
    }
    for (int i17 = 0; i17 < 8; i17++) {
        t53[i17] = tables[26][i17] * t52[i17];
    }
    for (int i18 = 0; i18 < 8; i18++) {
        double acc = 0.0;
        for (int i17 = 0; i17 < 8; i17++) {
            acc += tables[8][i17*8 + i18] * t53[i17];
        }
        t54[i18] = acc;
    }
    for (int i18 = 0; i18 < 8; i18++) {
        t55[i18] = tables[27][i18] * t54[i18];
    }
    for (int i2 = 0; i2 < 8; i2++) {
        double acc = 0.0;
        for (int i18 = 0; i18 < 8; i18++) {
            acc += tables[9][i2 + i18*8] * t55[i18];
        }
        t56[i2] = acc;
    }
    for (int i2 = 0; i2 < 8; i2++) {
        t57[i2] = tables[28][i2] * t56[i2];
    }
    for (int i2 = 0; i2 < 8; i2++) {
        t58[i2] = t15[i2] * t57[i2];
    }
    for (int i3 = 0; i3 < 8; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 8; i2++) {
            acc += tables[10][i2*8 + i3] * t57[i2];
        }
        t59[i3] = acc;
    }
    for (int i3 = 0; i3 < 8; i3++) {
        t60[i3] = t59[i3] * t14[i3];
    }
    for (int i3 = 0; i3 < 8; i3++) {
        t61[i3] = tables[29][i3] * t59[i3];
    }
    for (int i4 = 0; i4 < 8; i4++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 8; i3++) {
            acc += tables[11][i3*8 + i4] * t61[i3];
        }
        t62[i4] = acc;
    }
    for (int i4 = 0; i4 < 8; i4++) {
        t63[i4] = t62[i4] * t12[i4];
    }
    for (int i4 = 0; i4 < 8; i4++) {
        t64[i4] = tables[30][i4] * t62[i4];
    }
    for (int i5 = 0; i5 < 8; i5++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 8; i4++) {
            acc += tables[12][i4*8 + i5] * t64[i4];
        }
        t65[i5] = acc;
    }
    for (int i5 = 0; i5 < 8; i5++) {
        t66[i5] = t65[i5] * t10[i5];
    }
    for (int i5 = 0; i5 < 8; i5++) {
        t67[i5] = tables[31][i5] * t65[i5];
    }
    for (int i6 = 0; i6 < 8; i6++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 8; i5++) {
            acc += tables[13][i5*8 + i6] * t67[i5];
        }
        t68[i6] = acc;
    }
    for (int i6 = 0; i6 < 8; i6++) {
        t69[i6] = t68[i6] * t8[i6];
    }
    for (int i6 = 0; i6 < 8; i6++) {
        t70[i6] = tables[32][i6] * t68[i6];
    }
    for (int i7 = 0; i7 < 8; i7++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 8; i6++) {
            acc += tables[14][i6*8 + i7] * t70[i6];
        }
        t71[i7] = acc;
    }
    for (int i7 = 0; i7 < 8; i7++) {
        t72[i7] = t71[i7] * t6[i7];
    }
    for (int i7 = 0; i7 < 8; i7++) {
        t73[i7] = tables[33][i7] * t71[i7];
    }
    for (int i8 = 0; i8 < 8; i8++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 8; i7++) {
            acc += tables[15][i7*8 + i8] * t73[i7];
        }
        t74[i8] = acc;
    }
    for (int i8 = 0; i8 < 8; i8++) {
        t75[i8] = t74[i8] * t4[i8];
    }
    for (int i8 = 0; i8 < 8; i8++) {
        t76[i8] = tables[34][i8] * t74[i8];
    }
    for (int i9 = 0; i9 < 8; i9++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 8; i8++) {
            acc += tables[16][i8*8 + i9] * t76[i8];
        }
        t77[i9] = acc;
    }
    for (int i9 = 0; i9 < 8; i9++) {
        t78[i9] = t77[i9] * t2[i9];
    }
    for (int i9 = 0; i9 < 8; i9++) {
        t79[i9] = t77[i9] * tables[35][i9];
    }
    for (int i10 = 0; i10 < 8; i10++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 8; i9++) {
            acc += t79[i9] * t0[i9*8 + i10];
        }
        t80[i10] = acc;
    }
    for (int i11 = 0; i11 < 8; i11++) {
        t81[i11] = t31[i11] * t41[i11];
    }
    for (int i12 = 0; i12 < 8; i12++) {
        t82[i12] = t29[i12] * t43[i12];
    }
    for (int i13 = 0; i13 < 8; i13++) {
        t83[i13] = t27[i13] * t45[i13];
    }
    for (int i14 = 0; i14 < 8; i14++) {
        t84[i14] = t25[i14] * t47[i14];
    }
    for (int i15 = 0; i15 < 8; i15++) {
        t85[i15] = t23[i15] * t49[i15];
    }
    for (int i16 = 0; i16 < 8; i16++) {
        t86[i16] = t21[i16] * t51[i16];
    }
    for (int i17 = 0; i17 < 8; i17++) {
        t87[i17] = t19[i17] * t53[i17];
    }
    for (int i18 = 0; i18 < 8; i18++) {
        t88[i18] = t17[i18] * t55[i18];
    }
    double z = 0.0;
    for (int j = 0; j < 8; j++) z += t36[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 8; j++) out[0 + j] = t36[j] * iz;
    for (int j = 0; j < 8; j++) out[8 + j] = t39[j] * iz;
    for (int j = 0; j < 8; j++) out[16 + j] = t58[j] * iz;
    for (int j = 0; j < 8; j++) out[24 + j] = t60[j] * iz;
    for (int j = 0; j < 8; j++) out[32 + j] = t63[j] * iz;
    for (int j = 0; j < 8; j++) out[40 + j] = t66[j] * iz;
    for (int j = 0; j < 8; j++) out[48 + j] = t69[j] * iz;
    for (int j = 0; j < 8; j++) out[56 + j] = t72[j] * iz;
    for (int j = 0; j < 8; j++) out[64 + j] = t75[j] * iz;
    for (int j = 0; j < 8; j++) out[72 + j] = t78[j] * iz;
    for (int j = 0; j < 8; j++) out[80 + j] = t80[j] * iz;
    for (int j = 0; j < 8; j++) out[88 + j] = t81[j] * iz;
    for (int j = 0; j < 8; j++) out[96 + j] = t82[j] * iz;
    for (int j = 0; j < 8; j++) out[104 + j] = t83[j] * iz;
    for (int j = 0; j < 8; j++) out[112 + j] = t84[j] * iz;
    for (int j = 0; j < 8; j++) out[120 + j] = t85[j] * iz;
    for (int j = 0; j < 8; j++) out[128 + j] = t86[j] * iz;
    for (int j = 0; j < 8; j++) out[136 + j] = t87[j] * iz;
    for (int j = 0; j < 8; j++) out[144 + j] = t88[j] * iz;
}
