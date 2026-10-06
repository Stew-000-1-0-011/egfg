/* egfg program for holdout_lowrank_chain8_k12_r1 (cost model: 474 operations, overhead 0) */
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

static double lrU11[12];
static double lrV12[12];
static double lrW11[1];
static double lrU9[12];
static double lrV10[12];
static double lrW9[1];
static double t0[12];
static double t1[1];
static double lrU17[12];
static double lrV18[12];
static double lrW17[1];
static double lrU15[12];
static double lrV16[12];
static double lrW15[1];
static double t2[12];
static double t3[1];
static double lrU19[12];
static double lrV20[12];
static double lrW19[1];
static double t4[12];
static double t5[1];
static double t6[1];
static double t7[1];
static double t8[1];
static double lrU13[12];
static double lrV14[12];
static double lrW13[1];
static double t9[12];
static double t10[1];
static double t11[1];
static double t12[12];
static double t13[1];
static double t14[1];
static double t15[1];
static double lrU7[12];
static double lrV8[12];
static double lrW7[1];
static double t16[12];
static double t17[1];
static double t18[1];
static double t19[12];
static double t20[1];
static double t21[1];
static double t22[12];
static double t23[1];
static double t24[1];
static double t25[1];
static double t26[12];
static double t27[1];
static double t28[1];
static double t29[12];
static double t30[1];
static double t31[1];
static double t32[1];
static double t33[1];
static double t34[12];
static double t35[1];
static double t36[1];
static double t37[1];
static double t38[12];
static double t39[1];
static double t40[1];
static double t41[12];
static double t42[1];
static double t43[1];
static double t44[12];

void infer(const double *const *tables, double *out) {
    for (int i = 0; i < 12; i++) for (int a = 0; a < 1; a++) lrU11[i * 1 + a] = tables[2][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 12; j++) lrV12[a * 12 + j] = tables[2][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW11[a * 1 + b] = tables[2][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(1, lrW11, lrV12, 12);
    for (int i = 0; i < 12; i++) for (int a = 0; a < 1; a++) lrU9[i * 1 + a] = tables[1][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 12; j++) lrV10[a * 12 + j] = tables[1][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW9[a * 1 + b] = tables[1][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(1, lrW9, lrV10, 12);
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i2 = 0; i2 < 1; i2++) {
            for (int i9 = 0; i9 < 12; i9++) {
                t0[i1*12 + i2*12 + i9] = lrV10[i1*12 + i9] * lrU11[i2 + i9];
            }
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i2 = 0; i2 < 1; i2++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 12; i9++) {
                acc += t0[i1*12 + i2*12 + i9];
            }
            t1[i1 + i2] = acc;
        }
    }
    for (int i = 0; i < 12; i++) for (int a = 0; a < 1; a++) lrU17[i * 1 + a] = tables[5][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 12; j++) lrV18[a * 12 + j] = tables[5][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW17[a * 1 + b] = tables[5][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(1, lrW17, lrV18, 12);
    for (int i = 0; i < 12; i++) for (int a = 0; a < 1; a++) lrU15[i * 1 + a] = tables[4][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 12; j++) lrV16[a * 12 + j] = tables[4][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW15[a * 1 + b] = tables[4][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(1, lrW15, lrV16, 12);
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            for (int i12 = 0; i12 < 12; i12++) {
                t2[i4*12 + i5*12 + i12] = lrV16[i4*12 + i12] * lrU17[i5 + i12];
            }
        }
    }
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            double acc = 0.0;
            for (int i12 = 0; i12 < 12; i12++) {
                acc += t2[i4*12 + i5*12 + i12];
            }
            t3[i4 + i5] = acc;
        }
    }
    for (int i = 0; i < 12; i++) for (int a = 0; a < 1; a++) lrU19[i * 1 + a] = tables[6][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 12; j++) lrV20[a * 12 + j] = tables[6][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW19[a * 1 + b] = tables[6][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(1, lrW19, lrV20, 12);
    for (int i5 = 0; i5 < 1; i5++) {
        for (int i6 = 0; i6 < 1; i6++) {
            for (int i13 = 0; i13 < 12; i13++) {
                t4[i5*12 + i6*12 + i13] = lrV18[i5*12 + i13] * lrU19[i6 + i13];
            }
        }
    }
    for (int i5 = 0; i5 < 1; i5++) {
        for (int i6 = 0; i6 < 1; i6++) {
            double acc = 0.0;
            for (int i13 = 0; i13 < 12; i13++) {
                acc += t4[i5*12 + i6*12 + i13];
            }
            t5[i5 + i6] = acc;
        }
    }
    for (int i6 = 0; i6 < 1; i6++) {
        double acc = 0.0;
        for (int i14 = 0; i14 < 12; i14++) {
            acc += lrV20[i6*12 + i14];
        }
        t6[i6] = acc;
    }
    for (int i5 = 0; i5 < 1; i5++) {
        for (int i6 = 0; i6 < 1; i6++) {
            t7[i5 + i6] = t6[i6] * t5[i5 + i6];
        }
    }
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            for (int i6 = 0; i6 < 1; i6++) {
                t8[i4 + i5 + i6] = t7[i5 + i6] * t3[i4 + i5];
            }
        }
    }
    for (int i = 0; i < 12; i++) for (int a = 0; a < 1; a++) lrU13[i * 1 + a] = tables[3][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 12; j++) lrV14[a * 12 + j] = tables[3][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW13[a * 1 + b] = tables[3][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(1, lrW13, lrV14, 12);
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i4 = 0; i4 < 1; i4++) {
            for (int i11 = 0; i11 < 12; i11++) {
                t9[i3*12 + i4*12 + i11] = lrV14[i3*12 + i11] * lrU15[i4 + i11];
            }
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 12; i11++) {
                acc += t9[i3*12 + i4*12 + i11];
            }
            t10[i3 + i4] = acc;
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i4 = 0; i4 < 1; i4++) {
            for (int i5 = 0; i5 < 1; i5++) {
                double acc = 0.0;
                for (int i6 = 0; i6 < 1; i6++) {
                    acc += t10[i3 + i4] * t8[i4 + i5 + i6];
                }
                t11[i3 + i4 + i5] = acc;
            }
        }
    }
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 1; i3++) {
            for (int i10 = 0; i10 < 12; i10++) {
                t12[i2*12 + i3*12 + i10] = lrV12[i2*12 + i10] * lrU13[i3 + i10];
            }
        }
    }
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 1; i3++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 12; i10++) {
                acc += t12[i2*12 + i3*12 + i10];
            }
            t13[i2 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 1; i3++) {
                for (int i5 = 0; i5 < 1; i5++) {
                    acc += t13[i2 + i3] * t11[i3 + i4 + i5];
                }
            }
            t14[i2 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i2 = 0; i2 < 1; i2++) {
            for (int i4 = 0; i4 < 1; i4++) {
                t15[i1 + i2 + i4] = t14[i2 + i4] * t1[i1 + i2];
            }
        }
    }
    for (int i = 0; i < 12; i++) for (int a = 0; a < 1; a++) lrU7[i * 1 + a] = tables[0][((i / 1) % 12) * 12 + ((a / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 12; j++) lrV8[a * 12 + j] = tables[0][((a / 1) % 12) * 12 + ((j / 1) % 12) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW7[a * 1 + b] = tables[0][((a / 1) % 12) * 12 + ((b / 1) % 12) * 1];
    egfg_solve(1, lrW7, lrV8, 12);
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 1; i1++) {
            for (int i8 = 0; i8 < 12; i8++) {
                t16[i0*12 + i1*12 + i8] = lrV8[i0*12 + i8] * lrU9[i1 + i8];
            }
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 1; i1++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 12; i8++) {
                acc += t16[i0*12 + i1*12 + i8];
            }
            t17[i0 + i1] = acc;
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 1; i1++) {
            for (int i2 = 0; i2 < 1; i2++) {
                for (int i4 = 0; i4 < 1; i4++) {
                    acc += t17[i0 + i1] * t15[i1 + i2 + i4];
                }
            }
        }
        t18[i0] = acc;
    }
    for (int i7 = 0; i7 < 12; i7++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 1; i0++) {
            acc += lrU7[i0 + i7] * t18[i0];
        }
        t19[i7] = acc;
    }
    for (int i0 = 0; i0 < 1; i0++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 12; i7++) {
            acc += lrU7[i0 + i7];
        }
        t20[i0] = acc;
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 1; i1++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 1; i2++) {
                for (int i4 = 0; i4 < 1; i4++) {
                    acc += t20[i0] * t15[i1 + i2 + i4];
                }
            }
            t21[i0 + i1] = acc;
        }
    }
    for (int i8 = 0; i8 < 12; i8++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t16[i0*12 + i1*12 + i8] * t21[i0 + i1];
            }
        }
        t22[i8] = acc;
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 1; i1++) {
            t23[i0 + i1] = t20[i0] * t17[i0 + i1];
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 1; i0++) {
            acc += t23[i0 + i1];
        }
        t24[i1] = acc;
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i2 = 0; i2 < 1; i2++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 1; i4++) {
                acc += t14[i2 + i4] * t24[i1];
            }
            t25[i1 + i2] = acc;
        }
    }
    for (int i9 = 0; i9 < 12; i9++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 1; i1++) {
            for (int i2 = 0; i2 < 1; i2++) {
                acc += t0[i1*12 + i2*12 + i9] * t25[i1 + i2];
            }
        }
        t26[i9] = acc;
    }
    for (int i2 = 0; i2 < 1; i2++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t23[i0 + i1] * t1[i1 + i2];
            }
        }
        t27[i2] = acc;
    }
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 1; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 1; i4++) {
                for (int i5 = 0; i5 < 1; i5++) {
                    acc += t27[i2] * t11[i3 + i4 + i5];
                }
            }
            t28[i2 + i3] = acc;
        }
    }
    for (int i10 = 0; i10 < 12; i10++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 1; i2++) {
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t12[i2*12 + i3*12 + i10] * t28[i2 + i3];
            }
        }
        t29[i10] = acc;
    }
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i6 = 0; i6 < 1; i6++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 1; i5++) {
                acc += t8[i4 + i5 + i6];
            }
            t30[i4 + i6] = acc;
        }
    }
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 1; i3++) {
            t31[i2 + i3] = t27[i2] * t13[i2 + i3];
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 1; i2++) {
            acc += t31[i2 + i3];
        }
        t32[i3] = acc;
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 1; i6++) {
                acc += t32[i3] * t30[i4 + i6];
            }
            t33[i3 + i4] = acc;
        }
    }
    for (int i11 = 0; i11 < 12; i11++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 1; i3++) {
            for (int i4 = 0; i4 < 1; i4++) {
                acc += t9[i3*12 + i4*12 + i11] * t33[i3 + i4];
            }
        }
        t34[i11] = acc;
    }
    for (int i5 = 0; i5 < 1; i5++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 1; i6++) {
            acc += t7[i5 + i6];
        }
        t35[i5] = acc;
    }
    for (int i4 = 0; i4 < 1; i4++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 1; i2++) {
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t10[i3 + i4] * t31[i2 + i3];
            }
        }
        t36[i4] = acc;
    }
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            t37[i4 + i5] = t36[i4] * t35[i5];
        }
    }
    for (int i12 = 0; i12 < 12; i12++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 1; i4++) {
            for (int i5 = 0; i5 < 1; i5++) {
                acc += t2[i4*12 + i5*12 + i12] * t37[i4 + i5];
            }
        }
        t38[i12] = acc;
    }
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            for (int i6 = 0; i6 < 1; i6++) {
                t39[i4 + i5 + i6] = t6[i6] * t3[i4 + i5];
            }
        }
    }
    for (int i5 = 0; i5 < 1; i5++) {
        for (int i6 = 0; i6 < 1; i6++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 1; i4++) {
                acc += t39[i4 + i5 + i6] * t36[i4];
            }
            t40[i5 + i6] = acc;
        }
    }
    for (int i13 = 0; i13 < 12; i13++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 1; i5++) {
            for (int i6 = 0; i6 < 1; i6++) {
                acc += t4[i5*12 + i6*12 + i13] * t40[i5 + i6];
            }
        }
        t41[i13] = acc;
    }
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            for (int i6 = 0; i6 < 1; i6++) {
                t42[i4 + i5 + i6] = t5[i5 + i6] * t3[i4 + i5];
            }
        }
    }
    for (int i6 = 0; i6 < 1; i6++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 1; i4++) {
            for (int i5 = 0; i5 < 1; i5++) {
                acc += t42[i4 + i5 + i6] * t36[i4];
            }
        }
        t43[i6] = acc;
    }
    for (int i14 = 0; i14 < 12; i14++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 1; i6++) {
            acc += lrV20[i6*12 + i14] * t43[i6];
        }
        t44[i14] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 12; j++) z += t19[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 12; j++) out[0 + j] = t19[j] * iz;
    for (int j = 0; j < 12; j++) out[12 + j] = t22[j] * iz;
    for (int j = 0; j < 12; j++) out[24 + j] = t26[j] * iz;
    for (int j = 0; j < 12; j++) out[36 + j] = t29[j] * iz;
    for (int j = 0; j < 12; j++) out[48 + j] = t34[j] * iz;
    for (int j = 0; j < 12; j++) out[60 + j] = t38[j] * iz;
    for (int j = 0; j < 12; j++) out[72 + j] = t41[j] * iz;
    for (int j = 0; j < 12; j++) out[84 + j] = t44[j] * iz;
}
