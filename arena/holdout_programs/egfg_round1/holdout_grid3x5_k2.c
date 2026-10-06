/* egfg program for holdout_grid3x5_k2 (cost model: 972 operations, overhead 512) */
#include <stddef.h>

static double t0[8];
static double t1[4];
static double t2[8];
static double t3[16];
static double t4[8];
static double t5[8];
static double t6[4];
static double t7[8];
static double t8[8];
static double t9[8];
static double t10[8];
static double t11[8];
static double t12[8];
static double t13[8];
static double t14[8];
static double t15[4];
static double t16[8];
static double t17[16];
static double t18[16];
static double t19[8];
static double t20[8];
static double t21[8];
static double t22[4];
static double t23[8];
static double t24[4];
static double t25[2];
static double t26[2];
static double t27[4];
static double t28[8];
static double t29[8];
static double t30[4];
static double t31[4];
static double t32[2];
static double t33[2];
static double t34[4];
static double t35[2];
static double t36[8];
static double t37[8];
static double t38[8];
static double t39[8];
static double t40[16];
static double t41[4];
static double t42[4];
static double t43[2];
static double t44[2];
static double t45[8];
static double t46[4];
static double t47[2];
static double t48[4];
static double t49[4];
static double t50[2];
static double t51[2];
static double t52[2];
static double t53[2];
static double t54[2];
static double t55[2];
static double t56[2];

void infer(const double *const *tables, double *out) {
    for (int i8 = 0; i8 < 2; i8++) {
        for (int i9 = 0; i9 < 2; i9++) {
            for (int i14 = 0; i14 < 2; i14++) {
                t0[i8*4 + i9*2 + i14] = tables[6][i8*2 + i9] * tables[8][i9*2 + i14];
            }
        }
    }
    for (int i8 = 0; i8 < 2; i8++) {
        for (int i14 = 0; i14 < 2; i14++) {
            double acc = 0.0;
            for (int i9 = 0; i9 < 2; i9++) {
                acc += t0[i8*4 + i9*2 + i14];
            }
            t1[i8*2 + i14] = acc;
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i8 = 0; i8 < 2; i8++) {
            for (int i13 = 0; i13 < 2; i13++) {
                t2[i7*4 + i8*2 + i13] = tables[4][i7*2 + i8] * tables[7][i8*2 + i13];
            }
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i8 = 0; i8 < 2; i8++) {
            for (int i13 = 0; i13 < 2; i13++) {
                for (int i14 = 0; i14 < 2; i14++) {
                    t3[i7*8 + i8*4 + i13*2 + i14] = t2[i7*4 + i8*2 + i13] * tables[15][i13*2 + i14];
                }
            }
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i13 = 0; i13 < 2; i13++) {
            for (int i14 = 0; i14 < 2; i14++) {
                double acc = 0.0;
                for (int i8 = 0; i8 < 2; i8++) {
                    acc += t3[i7*8 + i8*4 + i13*2 + i14] * t1[i8*2 + i14];
                }
                t4[i7*4 + i13*2 + i14] = acc;
            }
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i6 = 0; i6 < 2; i6++) {
            for (int i14 = 0; i14 < 2; i14++) {
                t5[i5*4 + i6*2 + i14] = tables[17][i6 + i14*2] * tables[21][i5*2 + i6];
            }
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i14 = 0; i14 < 2; i14++) {
            double acc = 0.0;
            for (int i6 = 0; i6 < 2; i6++) {
                acc += t5[i5*4 + i6*2 + i14];
            }
            t6[i5*2 + i14] = acc;
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i13 = 0; i13 < 2; i13++) {
            for (int i14 = 0; i14 < 2; i14++) {
                t7[i5*4 + i13*2 + i14] = tables[16][i5 + i13*2] * t6[i5*2 + i14];
            }
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i13 = 0; i13 < 2; i13++) {
                double acc = 0.0;
                for (int i14 = 0; i14 < 2; i14++) {
                    acc += t7[i5*4 + i13*2 + i14] * t4[i7*4 + i13*2 + i14];
                }
                t8[i5*4 + i7*2 + i13] = acc;
            }
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i12 = 0; i12 < 2; i12++) {
            for (int i13 = 0; i13 < 2; i13++) {
                t9[i7*4 + i12*2 + i13] = tables[5][i7*2 + i12] * tables[13][i12*2 + i13];
            }
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i12 = 0; i12 < 2; i12++) {
                double acc = 0.0;
                for (int i13 = 0; i13 < 2; i13++) {
                    acc += t9[i7*4 + i12*2 + i13] * t8[i5*4 + i7*2 + i13];
                }
                t10[i5*4 + i7*2 + i12] = acc;
            }
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i5 = 0; i5 < 2; i5++) {
            for (int i12 = 0; i12 < 2; i12++) {
                t11[i4*4 + i5*2 + i12] = tables[14][i4 + i12*2] * tables[20][i4*2 + i5];
            }
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i12 = 0; i12 < 2; i12++) {
                double acc = 0.0;
                for (int i5 = 0; i5 < 2; i5++) {
                    acc += t11[i4*4 + i5*2 + i12] * t10[i5*4 + i7*2 + i12];
                }
                t12[i4*4 + i7*2 + i12] = acc;
            }
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i11 = 0; i11 < 2; i11++) {
                double acc = 0.0;
                for (int i12 = 0; i12 < 2; i12++) {
                    acc += tables[11][i11*2 + i12] * t12[i4*4 + i7*2 + i12];
                }
                t13[i4*4 + i7*2 + i11] = acc;
            }
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            for (int i10 = 0; i10 < 2; i10++) {
                t14[i2*4 + i3*2 + i10] = tables[10][i2 + i10*2] * tables[18][i2*2 + i3];
            }
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i10 = 0; i10 < 2; i10++) {
            double acc = 0.0;
            for (int i2 = 0; i2 < 2; i2++) {
                acc += t14[i2*4 + i3*2 + i10];
            }
            t15[i3*2 + i10] = acc;
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i10 = 0; i10 < 2; i10++) {
            for (int i11 = 0; i11 < 2; i11++) {
                t16[i3*4 + i10*2 + i11] = tables[9][i10*2 + i11] * tables[12][i3 + i11*2];
            }
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i4 = 0; i4 < 2; i4++) {
            for (int i10 = 0; i10 < 2; i10++) {
                for (int i11 = 0; i11 < 2; i11++) {
                    t17[i3*8 + i4*4 + i10*2 + i11] = t16[i3*4 + i10*2 + i11] * tables[19][i3*2 + i4];
                }
            }
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i4 = 0; i4 < 2; i4++) {
            for (int i10 = 0; i10 < 2; i10++) {
                for (int i11 = 0; i11 < 2; i11++) {
                    t18[i3*8 + i4*4 + i10*2 + i11] = t17[i3*8 + i4*4 + i10*2 + i11] * t15[i3*2 + i10];
                }
            }
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i10 = 0; i10 < 2; i10++) {
            for (int i11 = 0; i11 < 2; i11++) {
                double acc = 0.0;
                for (int i3 = 0; i3 < 2; i3++) {
                    acc += t18[i3*8 + i4*4 + i10*2 + i11];
                }
                t19[i4*4 + i10*2 + i11] = acc;
            }
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i10 = 0; i10 < 2; i10++) {
            for (int i11 = 0; i11 < 2; i11++) {
                double acc = 0.0;
                for (int i4 = 0; i4 < 2; i4++) {
                    acc += t19[i4*4 + i10*2 + i11] * t13[i4*4 + i7*2 + i11];
                }
                t20[i7*4 + i10*2 + i11] = acc;
            }
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i11 = 0; i11 < 2; i11++) {
                t21[i1*4 + i7*2 + i11] = tables[2][i1*2 + i7] * tables[3][i1*2 + i11];
            }
        }
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i10 = 0; i10 < 2; i10++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 2; i7++) {
                for (int i11 = 0; i11 < 2; i11++) {
                    acc += t21[i1*4 + i7*2 + i11] * t20[i7*4 + i10*2 + i11];
                }
            }
            t22[i1*2 + i10] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i1 = 0; i1 < 2; i1++) {
            for (int i10 = 0; i10 < 2; i10++) {
                t23[i0*4 + i1*2 + i10] = tables[0][i0*2 + i1] * tables[1][i0*2 + i10];
            }
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i1 = 0; i1 < 2; i1++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 2; i10++) {
                acc += t23[i0*4 + i1*2 + i10] * t22[i1*2 + i10];
            }
            t24[i0*2 + i1] = acc;
        }
    }
    for (int i0 = 0; i0 < 2; i0++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 2; i1++) {
            acc += t24[i0*2 + i1];
        }
        t25[i0] = acc;
    }
    for (int i1 = 0; i1 < 2; i1++) {
        double acc = 0.0;
        for (int i0 = 0; i0 < 2; i0++) {
            acc += t24[i0*2 + i1];
        }
        t26[i1] = acc;
    }
    for (int i1 = 0; i1 < 2; i1++) {
        for (int i10 = 0; i10 < 2; i10++) {
            double acc = 0.0;
            for (int i0 = 0; i0 < 2; i0++) {
                acc += t23[i0*4 + i1*2 + i10];
            }
            t27[i1*2 + i10] = acc;
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i10 = 0; i10 < 2; i10++) {
            for (int i11 = 0; i11 < 2; i11++) {
                double acc = 0.0;
                for (int i1 = 0; i1 < 2; i1++) {
                    acc += t21[i1*4 + i7*2 + i11] * t27[i1*2 + i10];
                }
                t28[i7*4 + i10*2 + i11] = acc;
            }
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i10 = 0; i10 < 2; i10++) {
            for (int i11 = 0; i11 < 2; i11++) {
                double acc = 0.0;
                for (int i7 = 0; i7 < 2; i7++) {
                    acc += t28[i7*4 + i10*2 + i11] * t13[i4*4 + i7*2 + i11];
                }
                t29[i4*4 + i10*2 + i11] = acc;
            }
        }
    }
    for (int i3 = 0; i3 < 2; i3++) {
        for (int i10 = 0; i10 < 2; i10++) {
            double acc = 0.0;
            for (int i4 = 0; i4 < 2; i4++) {
                for (int i11 = 0; i11 < 2; i11++) {
                    acc += t17[i3*8 + i4*4 + i10*2 + i11] * t29[i4*4 + i10*2 + i11];
                }
            }
            t30[i3*2 + i10] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
            double acc = 0.0;
            for (int i10 = 0; i10 < 2; i10++) {
                acc += t14[i2*4 + i3*2 + i10] * t30[i3*2 + i10];
            }
            t31[i2*2 + i3] = acc;
        }
    }
    for (int i2 = 0; i2 < 2; i2++) {
        double acc = 0.0;
        for (int i3 = 0; i3 < 2; i3++) {
            acc += t31[i2*2 + i3];
        }
        t32[i2] = acc;
    }
    for (int i3 = 0; i3 < 2; i3++) {
        double acc = 0.0;
        for (int i2 = 0; i2 < 2; i2++) {
            acc += t31[i2*2 + i3];
        }
        t33[i3] = acc;
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i11 = 0; i11 < 2; i11++) {
            double acc = 0.0;
            for (int i3 = 0; i3 < 2; i3++) {
                for (int i10 = 0; i10 < 2; i10++) {
                    acc += t18[i3*8 + i4*4 + i10*2 + i11] * t29[i4*4 + i10*2 + i11];
                }
            }
            t34[i4*2 + i11] = acc;
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        double acc = 0.0;
        for (int i11 = 0; i11 < 2; i11++) {
            acc += t34[i4*2 + i11];
        }
        t35[i4] = acc;
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i11 = 0; i11 < 2; i11++) {
                double acc = 0.0;
                for (int i10 = 0; i10 < 2; i10++) {
                    acc += t19[i4*4 + i10*2 + i11] * t28[i7*4 + i10*2 + i11];
                }
                t36[i4*4 + i7*2 + i11] = acc;
            }
        }
    }
    for (int i4 = 0; i4 < 2; i4++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i12 = 0; i12 < 2; i12++) {
                double acc = 0.0;
                for (int i11 = 0; i11 < 2; i11++) {
                    acc += tables[11][i11*2 + i12] * t36[i4*4 + i7*2 + i11];
                }
                t37[i4*4 + i7*2 + i12] = acc;
            }
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i12 = 0; i12 < 2; i12++) {
                double acc = 0.0;
                for (int i4 = 0; i4 < 2; i4++) {
                    acc += t11[i4*4 + i5*2 + i12] * t37[i4*4 + i7*2 + i12];
                }
                t38[i5*4 + i7*2 + i12] = acc;
            }
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i13 = 0; i13 < 2; i13++) {
                double acc = 0.0;
                for (int i12 = 0; i12 < 2; i12++) {
                    acc += t9[i7*4 + i12*2 + i13] * t38[i5*4 + i7*2 + i12];
                }
                t39[i5*4 + i7*2 + i13] = acc;
            }
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i7 = 0; i7 < 2; i7++) {
            for (int i13 = 0; i13 < 2; i13++) {
                for (int i14 = 0; i14 < 2; i14++) {
                    t40[i5*8 + i7*4 + i13*2 + i14] = tables[16][i5 + i13*2] * t4[i7*4 + i13*2 + i14];
                }
            }
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i14 = 0; i14 < 2; i14++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 2; i7++) {
                for (int i13 = 0; i13 < 2; i13++) {
                    acc += t40[i5*8 + i7*4 + i13*2 + i14] * t39[i5*4 + i7*2 + i13];
                }
            }
            t41[i5*2 + i14] = acc;
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        for (int i6 = 0; i6 < 2; i6++) {
            double acc = 0.0;
            for (int i14 = 0; i14 < 2; i14++) {
                acc += t5[i5*4 + i6*2 + i14] * t41[i5*2 + i14];
            }
            t42[i5*2 + i6] = acc;
        }
    }
    for (int i5 = 0; i5 < 2; i5++) {
        double acc = 0.0;
        for (int i6 = 0; i6 < 2; i6++) {
            acc += t42[i5*2 + i6];
        }
        t43[i5] = acc;
    }
    for (int i6 = 0; i6 < 2; i6++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 2; i5++) {
            acc += t42[i5*2 + i6];
        }
        t44[i6] = acc;
    }
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i13 = 0; i13 < 2; i13++) {
            for (int i14 = 0; i14 < 2; i14++) {
                double acc = 0.0;
                for (int i5 = 0; i5 < 2; i5++) {
                    acc += t7[i5*4 + i13*2 + i14] * t39[i5*4 + i7*2 + i13];
                }
                t45[i7*4 + i13*2 + i14] = acc;
            }
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        for (int i13 = 0; i13 < 2; i13++) {
            double acc = 0.0;
            for (int i14 = 0; i14 < 2; i14++) {
                acc += t45[i7*4 + i13*2 + i14] * t4[i7*4 + i13*2 + i14];
            }
            t46[i7*2 + i13] = acc;
        }
    }
    for (int i7 = 0; i7 < 2; i7++) {
        double acc = 0.0;
        for (int i13 = 0; i13 < 2; i13++) {
            acc += t46[i7*2 + i13];
        }
        t47[i7] = acc;
    }
    for (int i8 = 0; i8 < 2; i8++) {
        for (int i14 = 0; i14 < 2; i14++) {
            double acc = 0.0;
            for (int i7 = 0; i7 < 2; i7++) {
                for (int i13 = 0; i13 < 2; i13++) {
                    acc += t3[i7*8 + i8*4 + i13*2 + i14] * t45[i7*4 + i13*2 + i14];
                }
            }
            t48[i8*2 + i14] = acc;
        }
    }
    for (int i8 = 0; i8 < 2; i8++) {
        for (int i9 = 0; i9 < 2; i9++) {
            double acc = 0.0;
            for (int i14 = 0; i14 < 2; i14++) {
                acc += t0[i8*4 + i9*2 + i14] * t48[i8*2 + i14];
            }
            t49[i8*2 + i9] = acc;
        }
    }
    for (int i8 = 0; i8 < 2; i8++) {
        double acc = 0.0;
        for (int i9 = 0; i9 < 2; i9++) {
            acc += t49[i8*2 + i9];
        }
        t50[i8] = acc;
    }
    for (int i9 = 0; i9 < 2; i9++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 2; i8++) {
            acc += t49[i8*2 + i9];
        }
        t51[i9] = acc;
    }
    for (int i10 = 0; i10 < 2; i10++) {
        double acc = 0.0;
        for (int i1 = 0; i1 < 2; i1++) {
            acc += t22[i1*2 + i10] * t27[i1*2 + i10];
        }
        t52[i10] = acc;
    }
    for (int i11 = 0; i11 < 2; i11++) {
        double acc = 0.0;
        for (int i4 = 0; i4 < 2; i4++) {
            acc += t34[i4*2 + i11];
        }
        t53[i11] = acc;
    }
    for (int i12 = 0; i12 < 2; i12++) {
        double acc = 0.0;
        for (int i5 = 0; i5 < 2; i5++) {
            for (int i7 = 0; i7 < 2; i7++) {
                acc += t38[i5*4 + i7*2 + i12] * t10[i5*4 + i7*2 + i12];
            }
        }
        t54[i12] = acc;
    }
    for (int i13 = 0; i13 < 2; i13++) {
        double acc = 0.0;
        for (int i7 = 0; i7 < 2; i7++) {
            acc += t46[i7*2 + i13];
        }
        t55[i13] = acc;
    }
    for (int i14 = 0; i14 < 2; i14++) {
        double acc = 0.0;
        for (int i8 = 0; i8 < 2; i8++) {
            acc += t48[i8*2 + i14] * t1[i8*2 + i14];
        }
        t56[i14] = acc;
    }
    double z = 0.0;
    for (int j = 0; j < 2; j++) z += t25[j];
    double iz = 1.0 / z;
    for (int j = 0; j < 2; j++) out[0 + j] = t25[j] * iz;
    for (int j = 0; j < 2; j++) out[2 + j] = t26[j] * iz;
    for (int j = 0; j < 2; j++) out[4 + j] = t32[j] * iz;
    for (int j = 0; j < 2; j++) out[6 + j] = t33[j] * iz;
    for (int j = 0; j < 2; j++) out[8 + j] = t35[j] * iz;
    for (int j = 0; j < 2; j++) out[10 + j] = t43[j] * iz;
    for (int j = 0; j < 2; j++) out[12 + j] = t44[j] * iz;
    for (int j = 0; j < 2; j++) out[14 + j] = t47[j] * iz;
    for (int j = 0; j < 2; j++) out[16 + j] = t50[j] * iz;
    for (int j = 0; j < 2; j++) out[18 + j] = t51[j] * iz;
    for (int j = 0; j < 2; j++) out[20 + j] = t52[j] * iz;
    for (int j = 0; j < 2; j++) out[22 + j] = t53[j] * iz;
    for (int j = 0; j < 2; j++) out[24 + j] = t54[j] * iz;
    for (int j = 0; j < 2; j++) out[26 + j] = t55[j] * iz;
    for (int j = 0; j < 2; j++) out[28 + j] = t56[j] * iz;
}
