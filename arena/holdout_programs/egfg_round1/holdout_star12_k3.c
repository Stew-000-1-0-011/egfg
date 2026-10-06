/* egfg program for holdout_star12_k3 (cost model: 429 operations, overhead 64) */
#include <stddef.h>

static double t0[3];
static double t1[3];
static double t2[3];
static double t3[3];
static double t4[3];
static double t5[3];
static double t6[3];
static double t7[3];
static double t8[3];
static double t9[3];
static double t10[3];
static double t11[3];
static double t12[3];
static double t13[3];
static double t14[3];
static double t15[3];
static double t16[3];
static double t17[3];
static double t18[3];
static double t19[3];
static double t20[3];
static double t21[3];
static double t22[3];
static double t23[3];
static double t24[3];
static double t25[3];
static double t26[3];
static double t27[3];
static double t28[3];
static double t29[3];
static double t30[3];
static double t31[3];
static double t32[3];
static double t33[3];
static double t34[3];
static double t35[3];
static double t36[3];
static double t37[3];
static double t38[3];
static double t39[3];
static double t40[3];
static double t41[3];
static double t42[3];
static double t43[3];
static double t44[3];
static double t45[3];
static double t46[3];
static double t47[3];
static double t48[3];
static double t49[3];
static double t50[3];
static double t51[3];
static double t52[3];
static double t53[3];
static double t54[3];
static double t55[3];
static double t56[3];
static double t57[3];
static double t58[3];
static double t59[3];
static double t60[3];
static double t61[9];
static double t62[3];
static double t63[3];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            acc += tables[9][i0*3 + i2];
        }
        t0[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            acc += tables[0][i0*3 + i1];
        }
        t1[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t2[i0] = t1[i0] * t0[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            acc += tables[2][i0*3 + i5];
        }
        t3[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            acc += tables[10][i0*3 + i3];
        }
        t4[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t5[i0] = t4[i0] * t3[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t6[i0] = t5[i0] * t2[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += tables[3][i0*3 + i6];
        }
        t7[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t8[i0] = t7[i0] * t6[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            acc += tables[1][i0*3 + i4];
        }
        t9[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t10[i0] = t9[i0] * t8[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 3; i9++) {
            acc += tables[6][i0*3 + i9];
        }
        t11[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 3; i8++) {
            acc += tables[5][i0*3 + i8];
        }
        t12[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t13[i0] = t12[i0] * t11[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t14[i0] = t13[i0] * t10[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i10 = 0; i10 < 3; i10++) {
            acc += tables[7][i0*3 + i10];
        }
        t15[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 3; i7++) {
            acc += tables[4][i0*3 + i7];
        }
        t16[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t17[i0] = t16[i0] * t15[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t18[i0] = t17[i0] * t14[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 3; i11++) {
            acc += tables[8][i0*3 + i11];
        }
        t19[i0] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t20[i0] = t19[i0] * t18[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t21[i0] = t4[i0] * t9[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t22[i0] = t21[i0] * t3[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t23[i0] = t22[i0] * t17[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t24[i0] = t0[i0] * t23[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t25[i0] = t13[i0] * t19[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t26[i0] = t7[i0] * t25[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t27[i0] = t26[i0] * t24[i0];
    }
    for (int i1 = 0; i1 < 3; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[0][i0*3 + i1] * t27[i0];
        }
        t28[i1] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t29[i0] = t1[i0] * t23[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t30[i0] = t26[i0] * t29[i0];
    }
    for (int i2 = 0; i2 < 3; i2++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[9][i0*3 + i2] * t30[i0];
        }
        t31[i2] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t32[i0] = t9[i0] * t3[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t33[i0] = t32[i0] * t2[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t34[i0] = t7[i0] * t16[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t35[i0] = t34[i0] * t33[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t36[i0] = t15[i0] * t19[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t37[i0] = t13[i0] * t36[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t38[i0] = t37[i0] * t35[i0];
    }
    for (int i3 = 0; i3 < 3; i3++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[10][i0*3 + i3] * t38[i0];
        }
        t39[i3] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t40[i0] = t16[i0] * t8[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t41[i0] = t37[i0] * t40[i0];
    }
    for (int i4 = 0; i4 < 3; i4++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[1][i0*3 + i4] * t41[i0];
        }
        t42[i4] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t43[i0] = t21[i0] * t2[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t44[i0] = t34[i0] * t43[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t45[i0] = t37[i0] * t44[i0];
    }
    for (int i5 = 0; i5 < 3; i5++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[2][i0*3 + i5] * t45[i0];
        }
        t46[i5] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t47[i0] = t3[i0] * t16[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t48[i0] = t47[i0] * t43[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t49[i0] = t37[i0] * t48[i0];
    }
    for (int i6 = 0; i6 < 3; i6++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[3][i0*3 + i6] * t49[i0];
        }
        t50[i6] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t51[i0] = t37[i0] * t10[i0];
    }
    for (int i7 = 0; i7 < 3; i7++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[4][i0*3 + i7] * t51[i0];
        }
        t52[i7] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t53[i0] = t16[i0] * t11[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t54[i0] = t53[i0] * t36[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t55[i0] = t54[i0] * t10[i0];
    }
    for (int i8 = 0; i8 < 3; i8++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[5][i0*3 + i8] * t55[i0];
        }
        t56[i8] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t57[i0] = t16[i0] * t12[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t58[i0] = t57[i0] * t36[i0];
    }
    for (int i0 = 0; i0 < 3; i0++) {
        t59[i0] = t58[i0] * t10[i0];
    }
    for (int i9 = 0; i9 < 3; i9++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[6][i0*3 + i9] * t59[i0];
        }
        t60[i9] = acc;
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i10 = 0; i10 < 3; i10++) {
            t61[i0*3 + i10] = tables[7][i0*3 + i10] * t48[i0];
        }
    }
    for (int i10 = 0; i10 < 3; i10++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += t61[i0*3 + i10] * t26[i0];
        }
        t62[i10] = acc;
    }
    for (int i11 = 0; i11 < 3; i11++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            acc += tables[8][i0*3 + i11] * t18[i0];
        }
        t63[i11] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 3; j++) z += t20[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 3; j++) out[0 + j] = t20[j] * iz;
    for (int j = 0; j < 3; j++) out[3 + j] = t28[j] * iz;
    for (int j = 0; j < 3; j++) out[6 + j] = t31[j] * iz;
    for (int j = 0; j < 3; j++) out[9 + j] = t39[j] * iz;
    for (int j = 0; j < 3; j++) out[12 + j] = t42[j] * iz;
    for (int j = 0; j < 3; j++) out[15 + j] = t46[j] * iz;
    for (int j = 0; j < 3; j++) out[18 + j] = t50[j] * iz;
    for (int j = 0; j < 3; j++) out[21 + j] = t52[j] * iz;
    for (int j = 0; j < 3; j++) out[24 + j] = t56[j] * iz;
    for (int j = 0; j < 3; j++) out[27 + j] = t60[j] * iz;
    for (int j = 0; j < 3; j++) out[30 + j] = t62[j] * iz;
    for (int j = 0; j < 3; j++) out[33 + j] = t63[j] * iz;
}
