/* egfg program for holdout_star10_k6 (cost model: 1206 operations, overhead 64) */
#include <stddef.h>

static double t0[6];
static double t1[6];
static double t2[6];
static double t3[6];
static double t4[6];
static double t5[6];
static double t6[6];
static double t7[6];
static double t8[6];
static double t9[6];
static double t10[6];
static double t11[6];
static double t12[6];
static double t13[6];
static double t14[6];
static double t15[6];
static double t16[6];
static double t17[6];
static double t18[6];
static double t19[6];
static double t20[6];
static double t21[6];
static double t22[6];
static double t23[6];
static double t24[6];
static double t25[6];
static double t26[6];
static double t27[6];
static double t28[6];
static double t29[6];
static double t30[6];
static double t31[6];
static double t32[6];
static double t33[6];
static double t34[6];
static double t35[6];
static double t36[6];
static double t37[6];
static double t38[6];
static double t39[6];
static double t40[6];
static double t41[6];
static double t42[6];
static double t43[6];
static double t44[6];
static double t45[6];
static double t46[6];
static double t47[6];
static double t48[6];
static double t49[6];
static double t50[36];
static double t51[6];

void infer(const double *const *tables, double *out) {
    for (int i0 = 0; i0 < 6; i0++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 6; i2++) {
            acc += tables[1][i0*6 + i2];
        }
        t0[i0] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 6; i6++) {
            acc += tables[5][i0*6 + i6];
        }
        t1[i0] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t2[i0] = t1[i0] * t0[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 6; i9++) {
            acc += tables[8][i0*6 + i9];
        }
        t3[i0] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 6; i7++) {
            acc += tables[6][i0*6 + i7];
        }
        t4[i0] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t5[i0] = t4[i0] * t3[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t6[i0] = t5[i0] * t2[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 6; i3++) {
            acc += tables[2][i0*6 + i3];
        }
        t7[i0] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 6; i8++) {
            acc += tables[7][i0*6 + i8];
        }
        t8[i0] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t9[i0] = t8[i0] * t7[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 6; i5++) {
            acc += tables[4][i0*6 + i5];
        }
        t10[i0] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 6; i4++) {
            acc += tables[3][i0*6 + i4];
        }
        t11[i0] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t12[i0] = t11[i0] * t10[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t13[i0] = t12[i0] * t9[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t14[i0] = t13[i0] * t6[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 6; i1++) {
            acc += tables[0][i0*6 + i1];
        }
        t15[i0] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t16[i0] = t15[i0] * t14[i0];
    }
    for (int i1 = 0; i1 < 6; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 6; i0++) {
            acc += tables[0][i0*6 + i1] * t14[i0];
        }
        t17[i1] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t18[i0] = t10[i0] * t1[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t19[i0] = t18[i0] * t5[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t20[i0] = t15[i0] * t11[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t21[i0] = t20[i0] * t9[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t22[i0] = t21[i0] * t19[i0];
    }
    for (int i2 = 0; i2 < 6; i2++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 6; i0++) {
            acc += tables[1][i0*6 + i2] * t22[i0];
        }
        t23[i2] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t24[i0] = t15[i0] * t10[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t25[i0] = t0[i0] * t11[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t26[i0] = t25[i0] * t24[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t27[i0] = t8[i0] * t3[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t28[i0] = t1[i0] * t4[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t29[i0] = t28[i0] * t27[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t30[i0] = t29[i0] * t26[i0];
    }
    for (int i3 = 0; i3 < 6; i3++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 6; i0++) {
            acc += tables[2][i0*6 + i3] * t30[i0];
        }
        t31[i3] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t32[i0] = t0[i0] * t7[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t33[i0] = t32[i0] * t24[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t34[i0] = t29[i0] * t33[i0];
    }
    for (int i4 = 0; i4 < 6; i4++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 6; i0++) {
            acc += tables[3][i0*6 + i4] * t34[i0];
        }
        t35[i4] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t36[i0] = t32[i0] * t20[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t37[i0] = t28[i0] * t8[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t38[i0] = t37[i0] * t36[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t39[i0] = t3[i0] * t38[i0];
    }
    for (int i5 = 0; i5 < 6; i5++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 6; i0++) {
            acc += tables[4][i0*6 + i5] * t39[i0];
        }
        t40[i5] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t41[i0] = t10[i0] * t4[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t42[i0] = t41[i0] * t27[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t43[i0] = t42[i0] * t36[i0];
    }
    for (int i6 = 0; i6 < 6; i6++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 6; i0++) {
            acc += tables[5][i0*6 + i6] * t43[i0];
        }
        t44[i6] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t45[i0] = t18[i0] * t27[i0];
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t46[i0] = t45[i0] * t36[i0];
    }
    for (int i7 = 0; i7 < 6; i7++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 6; i0++) {
            acc += tables[6][i0*6 + i7] * t46[i0];
        }
        t47[i7] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        t48[i0] = t19[i0] * t36[i0];
    }
    for (int i8 = 0; i8 < 6; i8++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 6; i0++) {
            acc += tables[7][i0*6 + i8] * t48[i0];
        }
        t49[i8] = acc;
    }
    for (int i0 = 0; i0 < 6; i0++) {
        for (int i9 = 0; i9 < 6; i9++) {
            t50[i0*6 + i9] = tables[8][i0*6 + i9] * t38[i0];
        }
    }
    for (int i9 = 0; i9 < 6; i9++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 6; i0++) {
            acc += t10[i0] * t50[i0*6 + i9];
        }
        t51[i9] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 6; j++) z += t16[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 6; j++) out[0 + j] = t16[j] * iz;
    for (int j = 0; j < 6; j++) out[6 + j] = t17[j] * iz;
    for (int j = 0; j < 6; j++) out[12 + j] = t23[j] * iz;
    for (int j = 0; j < 6; j++) out[18 + j] = t31[j] * iz;
    for (int j = 0; j < 6; j++) out[24 + j] = t35[j] * iz;
    for (int j = 0; j < 6; j++) out[30 + j] = t40[j] * iz;
    for (int j = 0; j < 6; j++) out[36 + j] = t44[j] * iz;
    for (int j = 0; j < 6; j++) out[42 + j] = t47[j] * iz;
    for (int j = 0; j < 6; j++) out[48 + j] = t49[j] * iz;
    for (int j = 0; j < 6; j++) out[54 + j] = t51[j] * iz;
}
