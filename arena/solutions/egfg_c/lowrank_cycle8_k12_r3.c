/* egfg program for lowrank_cycle8_k12_r3 (cost model: 5508 operations, overhead 512) */
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

static double lrU22[36];
static double lrV23[36];
static double lrW22[9];
static double lrU20[36];
static double lrV21[36];
static double lrW20[9];
static double t0[108];
static double t1[9];
static double t2[36];
static double lrU18[36];
static double lrV19[36];
static double lrW18[9];
static double lrU16[36];
static double lrV17[36];
static double lrW16[9];
static double t3[108];
static double t4[9];
static double lrU12[36];
static double lrV13[36];
static double lrW12[9];
static double lrU10[36];
static double lrV11[36];
static double lrW10[9];
static double t5[108];
static double t6[9];
static double lrU14[36];
static double lrV15[36];
static double lrW14[9];
static double t7[108];
static double t8[9];
static double t9[108];
static double t10[9];
static double t11[9];
static double t12[9];
static double t13[9];
static double t14[108];
static double t15[9];
static double t16[9];
static double lrU8[36];
static double lrV9[36];
static double lrW8[9];
static double t17[108];
static double t18[9];
static double t19[36];
static double t20[36];
static double t21[12];
static double t22[36];
static double t23[9];
static double t24[12];
static double t25[9];
static double t26[9];
static double t27[9];
static double t28[9];
static double t29[12];
static double t30[9];
static double t31[9];
static double t32[12];
static double t33[9];
static double t34[12];
static double t35[9];
static double t36[12];
static double t37[9];
static double t38[12];
static double t39[9];
static double t40[12];

void infer(const double *const *tables, double *out) {
    for (int i = 0; i < 12; i++) for (int a = 0; a < 3; a++) lrU22[i * 3 + a] = tables[7][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int j = 0; j < 12; j++) lrV23[a * 12 + j] = tables[7][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) lrW22[a * 3 + b] = tables[7][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(3, lrW22, lrV23, 12);
    for (int i = 0; i < 12; i++) for (int a = 0; a < 3; a++) lrU20[i * 3 + a] = tables[6][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int j = 0; j < 12; j++) lrV21[a * 12 + j] = tables[6][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) lrW20[a * 3 + b] = tables[6][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(3, lrW20, lrV21, 12);
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i7 = 0; i7 < 3; i7++) {
            for (int i15 = 0; i15 < 12; i15++) {
                t0[i6*36 + i7*12 + i15] = lrV21[i6*12 + i15] * lrU22[i7 + i15*3];
            }
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i7 = 0; i7 < 3; i7++) {
            double acc = 0.0;
            for (int i15 = 0; i15 < 12; i15++) {
                acc += t0[i6*36 + i7*12 + i15];
            }
            t1[i6*3 + i7] = acc;
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i8 = 0; i8 < 12; i8++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 3; i7++) {
                acc += lrV23[i7*12 + i8] * t1[i6*3 + i7];
            }
            t2[i6*12 + i8] = acc;
        }
    }
    for (int i = 0; i < 12; i++) for (int a = 0; a < 3; a++) lrU18[i * 3 + a] = tables[5][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int j = 0; j < 12; j++) lrV19[a * 12 + j] = tables[5][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) lrW18[a * 3 + b] = tables[5][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(3, lrW18, lrV19, 12);
    for (int i = 0; i < 12; i++) for (int a = 0; a < 3; a++) lrU16[i * 3 + a] = tables[4][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int j = 0; j < 12; j++) lrV17[a * 12 + j] = tables[4][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) lrW16[a * 3 + b] = tables[4][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(3, lrW16, lrV17, 12);
    for (int i4 = 0; i4 < 3; i4++) {
        for (int i5 = 0; i5 < 3; i5++) {
            for (int i13 = 0; i13 < 12; i13++) {
                t3[i4*36 + i5*12 + i13] = lrV17[i4*12 + i13] * lrU18[i5 + i13*3];
            }
        }
    }
    for (int i4 = 0; i4 < 3; i4++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i13 = 0; i13 < 12; i13++) {
                acc += t3[i4*36 + i5*12 + i13];
            }
            t4[i4*3 + i5] = acc;
        }
    }
    for (int i = 0; i < 12; i++) for (int a = 0; a < 3; a++) lrU12[i * 3 + a] = tables[2][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int j = 0; j < 12; j++) lrV13[a * 12 + j] = tables[2][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) lrW12[a * 3 + b] = tables[2][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(3, lrW12, lrV13, 12);
    for (int i = 0; i < 12; i++) for (int a = 0; a < 3; a++) lrU10[i * 3 + a] = tables[1][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int j = 0; j < 12; j++) lrV11[a * 12 + j] = tables[1][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) lrW10[a * 3 + b] = tables[1][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(3, lrW10, lrV11, 12);
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i2 = 0; i2 < 3; i2++) {
            for (int i10 = 0; i10 < 12; i10++) {
                t5[i1*36 + i2*12 + i10] = lrV11[i1*12 + i10] * lrU12[i2 + i10*3];
            }
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i2 = 0; i2 < 3; i2++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 12; i10++) {
                acc += t5[i1*36 + i2*12 + i10];
            }
            t6[i1*3 + i2] = acc;
        }
    }
    for (int i = 0; i < 12; i++) for (int a = 0; a < 3; a++) lrU14[i * 3 + a] = tables[3][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int j = 0; j < 12; j++) lrV15[a * 12 + j] = tables[3][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) lrW14[a * 3 + b] = tables[3][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(3, lrW14, lrV15, 12);
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            for (int i12 = 0; i12 < 12; i12++) {
                t7[i3*36 + i4*12 + i12] = lrV15[i3*12 + i12] * lrU16[i4 + i12*3];
            }
        }
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i12 = 0; i12 < 12; i12++) {
                acc += t7[i3*36 + i4*12 + i12];
            }
            t8[i3*3 + i4] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i3 = 0; i3 < 3; i3++) {
            for (int i11 = 0; i11 < 12; i11++) {
                t9[i2*36 + i3*12 + i11] = lrV13[i2*12 + i11] * lrU14[i3 + i11*3];
            }
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i3 = 0; i3 < 3; i3++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 12; i11++) {
                acc += t9[i2*36 + i3*12 + i11];
            }
            t10[i2*3 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 3; i3++) {
                acc += t10[i2*3 + i3] * t8[i3*3 + i4];
            }
            t11[i2*3 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 3; i2++) {
                acc += t11[i2*3 + i4] * t6[i1*3 + i2];
            }
            t12[i1*3 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 3; i4++) {
                acc += t12[i1*3 + i4] * t4[i4*3 + i5];
            }
            t13[i1*3 + i5] = acc;
        }
    }
    for (int i5 = 0; i5 < 3; i5++) {
        for (int i6 = 0; i6 < 3; i6++) {
            for (int i14 = 0; i14 < 12; i14++) {
                t14[i5*36 + i6*12 + i14] = lrV19[i5*12 + i14] * lrU20[i6 + i14*3];
            }
        }
    }
    for (int i5 = 0; i5 < 3; i5++) {
        for (int i6 = 0; i6 < 3; i6++) {
            double acc = 0.0;
            for (int i14 = 0; i14 < 12; i14++) {
                acc += t14[i5*36 + i6*12 + i14];
            }
            t15[i5*3 + i6] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i6 = 0; i6 < 3; i6++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 3; i5++) {
                acc += t15[i5*3 + i6] * t13[i1*3 + i5];
            }
            t16[i1*3 + i6] = acc;
        }
    }
    for (int i = 0; i < 12; i++) for (int a = 0; a < 3; a++) lrU8[i * 3 + a] = tables[0][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int j = 0; j < 12; j++) lrV9[a * 12 + j] = tables[0][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) lrW8[a * 3 + b] = tables[0][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(3, lrW8, lrV9, 12);
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i9 = 0; i9 < 12; i9++) {
                t17[i0*36 + i1*12 + i9] = lrV9[i0*12 + i9] * lrU10[i1 + i9*3];
            }
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 12; i9++) {
                acc += t17[i0*36 + i1*12 + i9];
            }
            t18[i0*3 + i1] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i8 = 0; i8 < 12; i8++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 3; i0++) {
                acc += lrU8[i0 + i8*3] * t18[i0*3 + i1];
            }
            t19[i1*12 + i8] = acc;
        }
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i8 = 0; i8 < 12; i8++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 3; i1++) {
                acc += t19[i1*12 + i8] * t16[i1*3 + i6];
            }
            t20[i6*12 + i8] = acc;
        }
    }
    for (int i8 = 0; i8 < 12; i8++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            acc += t20[i6*12 + i8] * t2[i6*12 + i8];
        }
        t21[i8] = acc;
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i8 = 0; i8 < 12; i8++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 3; i6++) {
                acc += t16[i1*3 + i6] * t2[i6*12 + i8];
            }
            t22[i1*12 + i8] = acc;
        }
    }
    for (int i0 = 0; i0 < 3; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 12; i8++) {
                acc += lrU8[i0 + i8*3] * t22[i1*12 + i8];
            }
            t23[i0*3 + i1] = acc;
        }
    }
    for (int i9 = 0; i9 < 12; i9++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 3; i0++) {
            for (int i1 = 0; i1 < 3; i1++) {
                acc += t17[i0*36 + i1*12 + i9] * t23[i0*3 + i1];
            }
        }
        t24[i9] = acc;
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i6 = 0; i6 < 3; i6++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 12; i8++) {
                acc += t19[i1*12 + i8] * t2[i6*12 + i8];
            }
            t25[i1*3 + i6] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 3; i6++) {
                acc += t15[i5*3 + i6] * t25[i1*3 + i6];
            }
            t26[i1*3 + i5] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 3; i5++) {
                acc += t4[i4*3 + i5] * t26[i1*3 + i5];
            }
            t27[i1*3 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 3; i1++) {
        for (int i2 = 0; i2 < 3; i2++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 3; i4++) {
                acc += t11[i2*3 + i4] * t27[i1*3 + i4];
            }
            t28[i1*3 + i2] = acc;
        }
    }
    for (int i10 = 0; i10 < 12; i10++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 3; i2++) {
                acc += t5[i1*36 + i2*12 + i10] * t28[i1*3 + i2];
            }
        }
        t29[i10] = acc;
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 3; i1++) {
                acc += t6[i1*3 + i2] * t27[i1*3 + i4];
            }
            t30[i2*3 + i4] = acc;
        }
    }
    for (int i2 = 0; i2 < 3; i2++) {
        for (int i3 = 0; i3 < 3; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 3; i4++) {
                acc += t8[i3*3 + i4] * t30[i2*3 + i4];
            }
            t31[i2*3 + i3] = acc;
        }
    }
    for (int i11 = 0; i11 < 12; i11++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 3; i2++) {
            for (int i3 = 0; i3 < 3; i3++) {
                acc += t9[i2*36 + i3*12 + i11] * t31[i2*3 + i3];
            }
        }
        t32[i11] = acc;
    }
    for (int i3 = 0; i3 < 3; i3++) {
        for (int i4 = 0; i4 < 3; i4++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 3; i2++) {
                acc += t10[i2*3 + i3] * t30[i2*3 + i4];
            }
            t33[i3*3 + i4] = acc;
        }
    }
    for (int i12 = 0; i12 < 12; i12++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 3; i3++) {
            for (int i4 = 0; i4 < 3; i4++) {
                acc += t7[i3*36 + i4*12 + i12] * t33[i3*3 + i4];
            }
        }
        t34[i12] = acc;
    }
    for (int i4 = 0; i4 < 3; i4++) {
        for (int i5 = 0; i5 < 3; i5++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 3; i1++) {
                acc += t12[i1*3 + i4] * t26[i1*3 + i5];
            }
            t35[i4*3 + i5] = acc;
        }
    }
    for (int i13 = 0; i13 < 12; i13++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 3; i4++) {
            for (int i5 = 0; i5 < 3; i5++) {
                acc += t3[i4*36 + i5*12 + i13] * t35[i4*3 + i5];
            }
        }
        t36[i13] = acc;
    }
    for (int i5 = 0; i5 < 3; i5++) {
        for (int i6 = 0; i6 < 3; i6++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 3; i1++) {
                acc += t13[i1*3 + i5] * t25[i1*3 + i6];
            }
            t37[i5*3 + i6] = acc;
        }
    }
    for (int i14 = 0; i14 < 12; i14++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 3; i5++) {
            for (int i6 = 0; i6 < 3; i6++) {
                acc += t14[i5*36 + i6*12 + i14] * t37[i5*3 + i6];
            }
        }
        t38[i14] = acc;
    }
    for (int i6 = 0; i6 < 3; i6++) {
        for (int i7 = 0; i7 < 3; i7++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 12; i8++) {
                acc += lrV23[i7*12 + i8] * t20[i6*12 + i8];
            }
            t39[i6*3 + i7] = acc;
        }
    }
    for (int i15 = 0; i15 < 12; i15++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 3; i6++) {
            for (int i7 = 0; i7 < 3; i7++) {
                acc += t0[i6*36 + i7*12 + i15] * t39[i6*3 + i7];
            }
        }
        t40[i15] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 12; j++) z += t21[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 12; j++) out[0 + j] = t21[j] * iz;
    for (int j = 0; j < 12; j++) out[12 + j] = t24[j] * iz;
    for (int j = 0; j < 12; j++) out[24 + j] = t29[j] * iz;
    for (int j = 0; j < 12; j++) out[36 + j] = t32[j] * iz;
    for (int j = 0; j < 12; j++) out[48 + j] = t34[j] * iz;
    for (int j = 0; j < 12; j++) out[60 + j] = t36[j] * iz;
    for (int j = 0; j < 12; j++) out[72 + j] = t38[j] * iz;
    for (int j = 0; j < 12; j++) out[84 + j] = t40[j] * iz;
}
