/* egfg program for lowrank_cycle6_k8_r1 (cost model: 264 operations, overhead 64) */
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

static double lrU10[8];
static double lrV11[8];
static double lrW10[1];
static double lrU8[8];
static double lrV9[8];
static double lrW8[1];
static double t0[8];
static double t1[1];
static double lrU12[8];
static double lrV13[8];
static double lrW12[1];
static double t2[8];
static double t3[1];
static double t4[1];
static double lrU14[8];
static double lrV15[8];
static double lrW14[1];
static double t5[8];
static double t6[1];
static double t7[1];
static double lrU6[8];
static double lrV7[8];
static double lrW6[1];
static double t8[8];
static double t9[1];
static double t10[1];
static double lrU16[8];
static double lrV17[8];
static double lrW16[1];
static double t11[8];
static double t12[1];
static double t13[1];
static double t14[8];
static double t15[8];
static double t16[1];
static double t17[1];
static double t18[1];
static double t19[8];
static double t20[1];
static double t21[1];
static double t22[1];
static double t23[8];
static double t24[1];
static double t25[8];
static double t26[1];
static double t27[8];
static double t28[1];
static double t29[8];

void infer(const double *const *tables, double *out) {
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU10[i * 1 + a] = tables[2][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV11[a * 8 + j] = tables[2][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW10[a * 1 + b] = tables[2][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW10, lrV11, 8);
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU8[i * 1 + a] = tables[1][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV9[a * 8 + j] = tables[1][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW8[a * 1 + b] = tables[1][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW8, lrV9, 8);
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i2 = 0; i2 < 1; i2++) {
            for (int i8 = 0; i8 < 8; i8++) {
                t0[i1*8 + i2*8 + i8] = lrV9[i1*8 + i8] * lrU10[i2 + i8];
            }
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i2 = 0; i2 < 1; i2++) {
            double acc = 0.0;
            for (int i8 = 0; i8 < 8; i8++) {
                acc += t0[i1*8 + i2*8 + i8];
            }
            t1[i1 + i2] = acc;
        }
    }
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU12[i * 1 + a] = tables[3][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV13[a * 8 + j] = tables[3][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW12[a * 1 + b] = tables[3][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW12, lrV13, 8);
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 1; i3++) {
            for (int i9 = 0; i9 < 8; i9++) {
                t2[i2*8 + i3*8 + i9] = lrV11[i2*8 + i9] * lrU12[i3 + i9];
            }
        }
    }
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 1; i3++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 8; i9++) {
                acc += t2[i2*8 + i3*8 + i9];
            }
            t3[i2 + i3] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i3 = 0; i3 < 1; i3++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 1; i2++) {
                acc += t3[i2 + i3] * t1[i1 + i2];
            }
            t4[i1 + i3] = acc;
        }
    }
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU14[i * 1 + a] = tables[4][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV15[a * 8 + j] = tables[4][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW14[a * 1 + b] = tables[4][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW14, lrV15, 8);
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i4 = 0; i4 < 1; i4++) {
            for (int i10 = 0; i10 < 8; i10++) {
                t5[i3*8 + i4*8 + i10] = lrV13[i3*8 + i10] * lrU14[i4 + i10];
            }
        }
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 8; i10++) {
                acc += t5[i3*8 + i4*8 + i10];
            }
            t6[i3 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t6[i3 + i4] * t4[i1 + i3];
            }
            t7[i1 + i4] = acc;
        }
    }
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU6[i * 1 + a] = tables[0][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV7[a * 8 + j] = tables[0][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW6[a * 1 + b] = tables[0][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW6, lrV7, 8);
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 1; i1++) {
            for (int i7 = 0; i7 < 8; i7++) {
                t8[i0*8 + i1*8 + i7] = lrV7[i0*8 + i7] * lrU8[i1 + i7];
            }
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 1; i1++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 8; i7++) {
                acc += t8[i0*8 + i1*8 + i7];
            }
            t9[i0 + i1] = acc;
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t9[i0 + i1] * t7[i1 + i4];
            }
            t10[i0 + i4] = acc;
        }
    }
    for (int i = 0; i < 8; i++) for (int a = 0; a < 1; a++) lrU16[i * 1 + a] = tables[5][((i / 1) % 8) * 8 + ((a / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int j = 0; j < 8; j++) lrV17[a * 8 + j] = tables[5][((a / 1) % 8) * 8 + ((j / 1) % 8) * 1];
    for (int a = 0; a < 1; a++) for (int b = 0; b < 1; b++) lrW16[a * 1 + b] = tables[5][((a / 1) % 8) * 8 + ((b / 1) % 8) * 1];
    egfg_solve(1, lrW16, lrV17, 8);
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            for (int i11 = 0; i11 < 8; i11++) {
                t11[i4*8 + i5*8 + i11] = lrV15[i4*8 + i11] * lrU16[i5 + i11];
            }
        }
    }
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            double acc = 0.0;
            for (int i11 = 0; i11 < 8; i11++) {
                acc += t11[i4*8 + i5*8 + i11];
            }
            t12[i4 + i5] = acc;
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i5 = 0; i5 < 1; i5++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 1; i4++) {
                acc += t12[i4 + i5] * t10[i0 + i4];
            }
            t13[i0 + i5] = acc;
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i5 = 0; i5 < 1; i5++) {
            for (int i6 = 0; i6 < 8; i6++) {
                t14[i0*8 + i5*8 + i6] = lrU6[i0 + i6] * lrV17[i5*8 + i6];
            }
        }
    }
    for (int i6 = 0; i6 < 8; i6++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i5 = 0; i5 < 1; i5++) {
                acc += t14[i0*8 + i5*8 + i6] * t13[i0 + i5];
            }
        }
        t15[i6] = acc;
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i5 = 0; i5 < 1; i5++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 8; i6++) {
                acc += t14[i0*8 + i5*8 + i6];
            }
            t16[i0 + i5] = acc;
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i5 = 0; i5 < 1; i5++) {
                acc += t12[i4 + i5] * t16[i0 + i5];
            }
            t17[i0 + i4] = acc;
        }
    }
    for (int i0 = 0; i0 < 1; i0++) {
        for (int i1 = 0; i1 < 1; i1++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 1; i4++) {
                acc += t17[i0 + i4] * t7[i1 + i4];
            }
            t18[i0 + i1] = acc;
        }
    }
    for (int i7 = 0; i7 < 8; i7++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 1; i0++) {
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t8[i0*8 + i1*8 + i7] * t18[i0 + i1];
            }
        }
        t19[i7] = acc;
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 1; i0++) {
                acc += t17[i0 + i4] * t9[i0 + i1];
            }
            t20[i1 + i4] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i3 = 0; i3 < 1; i3++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 1; i4++) {
                acc += t20[i1 + i4] * t6[i3 + i4];
            }
            t21[i1 + i3] = acc;
        }
    }
    for (int i1 = 0; i1 < 1; i1++) {
        for (int i2 = 0; i2 < 1; i2++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t3[i2 + i3] * t21[i1 + i3];
            }
            t22[i1 + i2] = acc;
        }
    }
    for (int i8 = 0; i8 < 8; i8++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 1; i1++) {
            for (int i2 = 0; i2 < 1; i2++) {
                acc += t0[i1*8 + i2*8 + i8] * t22[i1 + i2];
            }
        }
        t23[i8] = acc;
    }
    for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 1; i3++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t21[i1 + i3] * t1[i1 + i2];
            }
            t24[i2 + i3] = acc;
        }
    }
    for (int i9 = 0; i9 < 8; i9++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 1; i2++) {
            for (int i3 = 0; i3 < 1; i3++) {
                acc += t2[i2*8 + i3*8 + i9] * t24[i2 + i3];
            }
        }
        t25[i9] = acc;
    }
    for (int i3 = 0; i3 < 1; i3++) {
        for (int i4 = 0; i4 < 1; i4++) {
            double acc = 0.0;
            for (int i1 = 0; i1 < 1; i1++) {
                acc += t20[i1 + i4] * t4[i1 + i3];
            }
            t26[i3 + i4] = acc;
        }
    }
    for (int i10 = 0; i10 < 8; i10++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 1; i3++) {
            for (int i4 = 0; i4 < 1; i4++) {
                acc += t5[i3*8 + i4*8 + i10] * t26[i3 + i4];
            }
        }
        t27[i10] = acc;
    }
    for (int i4 = 0; i4 < 1; i4++) {
        for (int i5 = 0; i5 < 1; i5++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 1; i0++) {
                acc += t16[i0 + i5] * t10[i0 + i4];
            }
            t28[i4 + i5] = acc;
        }
    }
    for (int i11 = 0; i11 < 8; i11++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 1; i4++) {
            for (int i5 = 0; i5 < 1; i5++) {
                acc += t11[i4*8 + i5*8 + i11] * t28[i4 + i5];
            }
        }
        t29[i11] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 8; j++) z += t15[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 8; j++) out[0 + j] = t15[j] * iz;
    for (int j = 0; j < 8; j++) out[8 + j] = t19[j] * iz;
    for (int j = 0; j < 8; j++) out[16 + j] = t23[j] * iz;
    for (int j = 0; j < 8; j++) out[24 + j] = t25[j] * iz;
    for (int j = 0; j < 8; j++) out[32 + j] = t27[j] * iz;
    for (int j = 0; j < 8; j++) out[40 + j] = t29[j] * iz;
}
