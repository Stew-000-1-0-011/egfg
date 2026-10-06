/* egfg program for holdout_lowrank_cycle10_k8_r1 (cost model: 448 operations, overhead 64) */
#include <stddef.h>
#include <math.h>


/* Solve W X = B in place (W: r x r, B: r x n, row-major) by Gauss-Jordan with partial pivoting. */
static void egfg_solve(int r, double *W, double *B, int n) {
    for (int c = 0; c < r; c++) {
        int p = c;
        for (int i = c + 1; i < r; i++)
            if (fabs(W[i * r + c]) > fabs(W[p * r + c])) p = i;
        if (p != c) {
            for (int j = 0; j < r; j++) { double t = W[c * r + j]; W[c * r + j] = W[p * r + j]; W[p * r + j] = t; }
            for (int j = 0; j < n; j++) { double t = B[c * n + j]; B[c * n + j] = B[p * n + j]; B[p * n + j] = t; }
        }
        double inv = 1.0 / W[c * r + c];
        for (int j = 0; j < r; j++) W[c * r + j] *= inv;
        for (int j = 0; j < n; j++) B[c * n + j] *= inv;
        for (int i = 0; i < r; i++) {
            if (i == c) continue;
            double f = W[i * r + c];
            if (f == 0.0) continue;
            for (int j = 0; j < r; j++) W[i * r + j] -= f * W[c * r + j];
            for (int j = 0; j < n; j++) B[i * n + j] -= f * B[c * n + j];
        }
    }
}

static double lrU16[8];
static double lrV17[8];
static double lrW16[1];
static double lrU14[8];
static double lrV15[8];
static double lrW14[1];
static double t0[8];
static double t1[1];
static double lrU12[8];
static double lrV13[8];
static double lrW12[1];
static double t2[8];
static double t3[1];
static double t4[1];
static double lrU26[8];
static double lrV27[8];
static double lrW26[1];
static double lrU24[8];
static double lrV25[8];
static double lrW24[1];
static double t5[8];
static double t6[1];
static double lrU22[8];
static double lrV23[8];
static double lrW22[1];
static double t7[8];
static double t8[1];
static double lrU20[8];
static double lrV21[8];
static double lrW20[1];
static double lrU18[8];
static double lrV19[8];
static double lrW18[1];
static double t9[8];
static double t10[1];
static double t11[8];
static double t12[1];
static double t13[1];
static double t14[8];
static double t15[1];
static double t16[1];
static double t17[1];
static double t18[1];
static double t19[1];
static double lrU28[8];
static double lrV29[8];
static double lrW28[1];
static double t20[8];
static double t21[1];
static double t22[1];
static double lrU10[8];
static double lrV11[8];
static double lrW10[1];
static double t23[8];
static double t24[1];
static double t25[1];
static double t26[8];
static double t27[8];
static double t28[1];
static double t29[1];
static double t30[8];
static double t31[1];
static double t32[1];
static double t33[1];
static double t34[1];
static double t35[8];
static double t36[1];
static double t37[8];
static double t38[1];
static double t39[1];
static double t40[1];
static double t41[1];
static double t42[1];
static double t43[8];
static double t44[1];
static double t45[8];
static double t46[1];
static double t47[8];
static double t48[1];
static double t49[8];
static double t50[1];
static double t51[8];
static double t52[1];
static double t53[8];

void infer(const double *const *tables, double *out) {
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU16[i * 1 + a] = tables[3][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV17[a * 8 + j] = tables[3][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW16[a * 1 + b] = tables[3][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW16, lrV17, 8);
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU14[i * 1 + a] = tables[2][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV15[a * 8 + j] = tables[2][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW14[a * 1 + b] = tables[2][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW14, lrV15, 8);
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 1; i3++) {
            for (int i13 = 0; i13 < 8; i13++) {
                t0[i2*8 + i3*8 + i13] = lrV15[i2*8 + i13] * lrU16[i3 + i13];
            }
        }
    }
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 1; i3++) {
            double acc = 0.0;
            for (int i13 = 0; i13 < 8; i13++) {
                acc += t0[i2*8 + i3*8 + i13];
            }
            t1[i2 + i3] = acc;
        }
    }
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU12[i * 1 + a] = tables[1][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV13[a * 8 + j] = tables[1][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW12[a * 1 + b] = tables[1][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW12, lrV13, 8);
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i2 = 0; i2 < 1; i2++) {
            for (int i12 = 0; i12 < 8; i12++) {
                t2[i1*8 + i2*8 + i12] = lrV13[i1*8 + i12] * lrU14[i2 + i12];
            }
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i2 = 0; i2 < 1; i2++) {
            double acc = 0.0;
            for (int i12 = 0; i12 < 8; i12++) {
                acc += t2[i1*8 + i2*8 + i12];
            }
            t3[i1 + i2] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i3 = 0; i3 < 1; i3++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 1; i2++) {
                acc += t3[i1 + i2] * t1[i2 + i3];
            }
            t4[i1 + i3] = acc;
        }
    }
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU26[i * 1 + a] = tables[8][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV27[a * 8 + j] = tables[8][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW26[a * 1 + b] = tables[8][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW26, lrV27, 8);
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU24[i * 1 + a] = tables[7][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV25[a * 8 + j] = tables[7][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW24[a * 1 + b] = tables[7][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW24, lrV25, 8);
    for (int i7 = 0; i7 < 1; i7++) {
        for (int i8 = 0; i8 < 1; i8++) {
            for (int i18 = 0; i18 < 8; i18++) {
                t5[i7*8 + i8*8 + i18] = lrV25[i7*8 + i18] * lrU26[i8 + i18];
            }
        }
    }
    for (int i7 = 0; i7 < 1; i7++) {
        for (int i8 = 0; i8 < 1; i8++) {
            double acc = 0.0;
            for (int i18 = 0; i18 < 8; i18++) {
                acc += t5[i7*8 + i8*8 + i18];
            }
            t6[i7 + i8] = acc;
        }
    }
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU22[i * 1 + a] = tables[6][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV23[a * 8 + j] = tables[6][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW22[a * 1 + b] = tables[6][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW22, lrV23, 8);
    for (int i6 = 0; i6 < 1; i6++) {
        for (int i7 = 0; i7 < 1; i7++) {
            for (int i17 = 0; i17 < 8; i17++) {
                t7[i6*8 + i7*8 + i17] = lrV23[i6*8 + i17] * lrU24[i7 + i17];
            }
        }
    }
    for (int i6 = 0; i6 < 1; i6++) {
        for (int i7 = 0; i7 < 1; i7++) {
            double acc = 0.0;
            for (int i17 = 0; i17 < 8; i17++) {
                acc += t7[i6*8 + i7*8 + i17];
            }
            t8[i6 + i7] = acc;
        }
    }
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU20[i * 1 + a] = tables[5][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV21[a * 8 + j] = tables[5][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW20[a * 1 + b] = tables[5][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW20, lrV21, 8);
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU18[i * 1 + a] = tables[4][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV19[a * 8 + j] = tables[4][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW18[a * 1 + b] = tables[4][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW18, lrV19, 8);
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            for (int i15 = 0; i15 < 8; i15++) {
                t9[i4*8 + i5*8 + i15] = lrV19[i4*8 + i15] * lrU20[i5 + i15];
            }
        }
    }
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            double acc = 0.0;
            for (int i15 = 0; i15 < 8; i15++) {
                acc += t9[i4*8 + i5*8 + i15];
            }
            t10[i4 + i5] = acc;
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i4 = 0; i4 < 1; i4++) {
            for (int i14 = 0; i14 < 8; i14++) {
                t11[i3*8 + i4*8 + i14] = lrV17[i3*8 + i14] * lrU18[i4 + i14];
            }
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i14 = 0; i14 < 8; i14++) {
                acc += t11[i3*8 + i4*8 + i14];
            }
            t12[i3 + i4] = acc;
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i5 = 0; i5 < 1; i5++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 1; i4++) {
                acc += t12[i3 + i4] * t10[i4 + i5];
            }
            t13[i3 + i5] = acc;
        }
    }
    for (int i5 = 0; i5 < 1; i5++) {
        for (int i6 = 0; i6 < 1; i6++) {
            for (int i16 = 0; i16 < 8; i16++) {
                t14[i5*8 + i6*8 + i16] = lrV21[i5*8 + i16] * lrU22[i6 + i16];
            }
        }
    }
    for (int i5 = 0; i5 < 1; i5++) {
        for (int i6 = 0; i6 < 1; i6++) {
            double acc = 0.0;
            for (int i16 = 0; i16 < 8; i16++) {
                acc += t14[i5*8 + i6*8 + i16];
            }
            t15[i5 + i6] = acc;
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i6 = 0; i6 < 1; i6++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 1; i5++) {
                acc += t15[i5 + i6] * t13[i3 + i5];
            }
            t16[i3 + i6] = acc;
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i7 = 0; i7 < 1; i7++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 1; i6++) {
                acc += t16[i3 + i6] * t8[i6 + i7];
            }
            t17[i3 + i7] = acc;
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i8 = 0; i8 < 1; i8++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 1; i7++) {
                acc += t17[i3 + i7] * t6[i7 + i8];
            }
            t18[i3 + i8] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i8 = 0; i8 < 1; i8++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t18[i3 + i8] * t4[i1 + i3];
            }
            t19[i1 + i8] = acc;
        }
    }
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU28[i * 1 + a] = tables[9][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV29[a * 8 + j] = tables[9][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW28[a * 1 + b] = tables[9][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW28, lrV29, 8);
    for (int i8 = 0; i8 < 1; i8++) {
        for (int i9 = 0; i9 < 1; i9++) {
            for (int i19 = 0; i19 < 8; i19++) {
                t20[i8*8 + i9*8 + i19] = lrV27[i8*8 + i19] * lrU28[i9 + i19];
            }
        }
    }
    for (int i8 = 0; i8 < 1; i8++) {
        for (int i9 = 0; i9 < 1; i9++) {
            double acc = 0.0;
            for (int i19 = 0; i19 < 8; i19++) {
                acc += t20[i8*8 + i9*8 + i19];
            }
            t21[i8 + i9] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i9 = 0; i9 < 1; i9++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 1; i8++) {
                acc += t21[i8 + i9] * t19[i1 + i8];
            }
            t22[i1 + i9] = acc;
        }
    }
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU10[i * 1 + a] = tables[0][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV11[a * 8 + j] = tables[0][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW10[a * 1 + b] = tables[0][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW10, lrV11, 8);
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 1; i1++) {
            for (int i11 = 0; i11 < 8; i11++) {
                t23[i0*8 + i1*8 + i11] = lrV11[i0*8 + i11] * lrU12[i1 + i11];
            }
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 1; i1++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 8; i11++) {
                acc += t23[i0*8 + i1*8 + i11];
            }
            t24[i0 + i1] = acc;
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i9 = 0; i9 < 1; i9++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t24[i0 + i1] * t22[i1 + i9];
            }
            t25[i0 + i9] = acc;
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i9 = 0; i9 < 1; i9++) {
            for (int i10 = 0; i10 < 8; i10++) {
                t26[i0*8 + i9*8 + i10] = lrU10[i0 + i10] * lrV29[i9*8 + i10];
            }
        }
    }
    for (int i10 = 0; i10 < 8; i10++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i9 = 0; i9 < 1; i9++) {
                acc += t26[i0*8 + i9*8 + i10] * t25[i0 + i9];
            }
        }
        t27[i10] = acc;
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i9 = 0; i9 < 1; i9++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 8; i10++) {
                acc += t26[i0*8 + i9*8 + i10];
            }
            t28[i0 + i9] = acc;
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 1; i1++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 1; i9++) {
                acc += t28[i0 + i9] * t22[i1 + i9];
            }
            t29[i0 + i1] = acc;
        }
    }
    for (int i11 = 0; i11 < 8; i11++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t23[i0*8 + i1*8 + i11] * t29[i0 + i1];
            }
        }
        t30[i11] = acc;
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i9 = 0; i9 < 1; i9++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 1; i0++) {
                acc += t28[i0 + i9] * t24[i0 + i1];
            }
            t31[i1 + i9] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i8 = 0; i8 < 1; i8++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 1; i9++) {
                acc += t31[i1 + i9] * t21[i8 + i9];
            }
            t32[i1 + i8] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i3 = 0; i3 < 1; i3++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 1; i8++) {
                acc += t32[i1 + i8] * t18[i3 + i8];
            }
            t33[i1 + i3] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i2 = 0; i2 < 1; i2++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t33[i1 + i3] * t1[i2 + i3];
            }
            t34[i1 + i2] = acc;
        }
    }
    for (int i12 = 0; i12 < 8; i12++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 1; i1++) {
            for (int i2 = 0; i2 < 1; i2++) {
                acc += t2[i1*8 + i2*8 + i12] * t34[i1 + i2];
            }
        }
        t35[i12] = acc;
    }
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 1; i3++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t33[i1 + i3] * t3[i1 + i2];
            }
            t36[i2 + i3] = acc;
        }
    }
    for (int i13 = 0; i13 < 8; i13++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 1; i2++) {
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t0[i2*8 + i3*8 + i13] * t36[i2 + i3];
            }
        }
        t37[i13] = acc;
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i8 = 0; i8 < 1; i8++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t32[i1 + i8] * t4[i1 + i3];
            }
            t38[i3 + i8] = acc;
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i7 = 0; i7 < 1; i7++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 1; i8++) {
                acc += t6[i7 + i8] * t38[i3 + i8];
            }
            t39[i3 + i7] = acc;
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i6 = 0; i6 < 1; i6++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 1; i7++) {
                acc += t8[i6 + i7] * t39[i3 + i7];
            }
            t40[i3 + i6] = acc;
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i5 = 0; i5 < 1; i5++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 1; i6++) {
                acc += t15[i5 + i6] * t40[i3 + i6];
            }
            t41[i3 + i5] = acc;
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 1; i5++) {
                acc += t10[i4 + i5] * t41[i3 + i5];
            }
            t42[i3 + i4] = acc;
        }
    }
    for (int i14 = 0; i14 < 8; i14++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 1; i3++) {
            for (int i4 = 0; i4 < 1; i4++) {
                acc += t11[i3*8 + i4*8 + i14] * t42[i3 + i4];
            }
        }
        t43[i14] = acc;
    }
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t12[i3 + i4] * t41[i3 + i5];
            }
            t44[i4 + i5] = acc;
        }
    }
    for (int i15 = 0; i15 < 8; i15++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 1; i4++) {
            for (int i5 = 0; i5 < 1; i5++) {
                acc += t9[i4*8 + i5*8 + i15] * t44[i4 + i5];
            }
        }
        t45[i15] = acc;
    }
    for (int i5 = 0; i5 < 1; i5++) {
        for (int i6 = 0; i6 < 1; i6++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t13[i3 + i5] * t40[i3 + i6];
            }
            t46[i5 + i6] = acc;
        }
    }
    for (int i16 = 0; i16 < 8; i16++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 1; i5++) {
            for (int i6 = 0; i6 < 1; i6++) {
                acc += t14[i5*8 + i6*8 + i16] * t46[i5 + i6];
            }
        }
        t47[i16] = acc;
    }
    for (int i6 = 0; i6 < 1; i6++) {
        for (int i7 = 0; i7 < 1; i7++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t16[i3 + i6] * t39[i3 + i7];
            }
            t48[i6 + i7] = acc;
        }
    }
    for (int i17 = 0; i17 < 8; i17++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 1; i6++) {
            for (int i7 = 0; i7 < 1; i7++) {
                acc += t7[i6*8 + i7*8 + i17] * t48[i6 + i7];
            }
        }
        t49[i17] = acc;
    }
    for (int i7 = 0; i7 < 1; i7++) {
        for (int i8 = 0; i8 < 1; i8++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t17[i3 + i7] * t38[i3 + i8];
            }
            t50[i7 + i8] = acc;
        }
    }
    for (int i18 = 0; i18 < 8; i18++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 1; i7++) {
            for (int i8 = 0; i8 < 1; i8++) {
                acc += t5[i7*8 + i8*8 + i18] * t50[i7 + i8];
            }
        }
        t51[i18] = acc;
    }
    for (int i8 = 0; i8 < 1; i8++) {
        for (int i9 = 0; i9 < 1; i9++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t31[i1 + i9] * t19[i1 + i8];
            }
            t52[i8 + i9] = acc;
        }
    }
    for (int i19 = 0; i19 < 8; i19++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 1; i8++) {
            for (int i9 = 0; i9 < 1; i9++) {
                acc += t20[i8*8 + i9*8 + i19] * t52[i8 + i9];
            }
        }
        t53[i19] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 8; j++) z += t27[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 8; j++) out[0 + j] = t27[j] * iz;
    for (int j = 0; j < 8; j++) out[8 + j] = t30[j] * iz;
    for (int j = 0; j < 8; j++) out[16 + j] = t35[j] * iz;
    for (int j = 0; j < 8; j++) out[24 + j] = t37[j] * iz;
    for (int j = 0; j < 8; j++) out[32 + j] = t43[j] * iz;
    for (int j = 0; j < 8; j++) out[40 + j] = t45[j] * iz;
    for (int j = 0; j < 8; j++) out[48 + j] = t47[j] * iz;
    for (int j = 0; j < 8; j++) out[56 + j] = t49[j] * iz;
    for (int j = 0; j < 8; j++) out[64 + j] = t51[j] * iz;
    for (int j = 0; j < 8; j++) out[72 + j] = t53[j] * iz;
}
