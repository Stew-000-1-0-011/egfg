import numpy as np

# Tree: two-pass belief propagation with .dot messages, rooted at x7.
# The root belief is normalised before the downward pass, so every downward message
# (and hence every belief) comes out already normalised.
_ones = np.ones


def infer(tables):
    t0 = tables[0]
    t1 = tables[1]
    t2 = tables[2]
    t3 = tables[3]
    t4 = tables[4]
    t5 = tables[5]
    t6 = tables[6]
    t7 = tables[7]
    t8 = tables[8]
    t9 = tables[9]
    t10 = tables[10]
    one = _ones(3)
    u8 = t7.dot(one)
    u9 = t8.dot(one)
    u6 = t5.dot(u8)
    u5 = t4.dot(u9)
    u11 = t10.dot(one)
    p4 = u5 * u6
    u4 = t3.dot(p4)
    u3 = t2.dot(u11)
    u10 = t9.dot(one)
    p2 = u3 * u4
    u2 = t1.dot(p2)
    u0 = u10.dot(t0)
    p1 = u0 * u2
    u1 = p1.dot(t6)
    b7 = u1 / u1.sum()
    e1 = b7 / u1
    d1 = t6.dot(e1)
    e0 = d1 * u2
    e2 = d1 * u0
    b1 = e0 * u0
    d0 = t0.dot(e0)
    d2 = e2.dot(t1)
    b0 = d0 * u10
    d10 = d0.dot(t9)
    e3 = d2 * u4
    e4 = d2 * u3
    b2 = e3 * u3
    d3 = e3.dot(t2)
    d4 = e4.dot(t3)
    b3 = d3 * u11
    d11 = d3.dot(t10)
    e5 = d4 * u6
    e6 = d4 * u5
    b4 = e5 * u5
    d5 = e5.dot(t4)
    d6 = e6.dot(t5)
    b5 = d5 * u9
    d9 = d5.dot(t8)
    b6 = d6 * u8
    d8 = d6.dot(t7)
    return {'x0': b0, 'x1': b1, 'x2': b2, 'x3': b3, 'x4': b4, 'x5': b5, 'x6': b6, 'x7': b7, 'x8': d8, 'x9': d9, 'x10': d10, 'x11': d11}
