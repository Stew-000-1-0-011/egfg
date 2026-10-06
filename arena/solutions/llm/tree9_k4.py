import numpy as np

# Tree: two-pass belief propagation with .dot messages, rooted at x4.
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
    one = _ones(4)
    u7 = t6.dot(one)
    u6 = t5.dot(one)
    u5 = t4.dot(u7)
    p3 = u5 * u6
    u3 = t2.dot(p3)
    u8 = t7.dot(one)
    u2 = t1.dot(u3)
    u0 = u8.dot(t0)
    p1 = u0 * u2
    u1 = p1.dot(t3)
    b4 = u1 / u1.sum()
    e1 = b4 / u1
    d1 = t3.dot(e1)
    e0 = d1 * u2
    e2 = d1 * u0
    b1 = e0 * u0
    d0 = t0.dot(e0)
    d2 = e2.dot(t1)
    b0 = d0 * u8
    d8 = d0.dot(t7)
    b2 = d2 * u3
    d3 = d2.dot(t2)
    e5 = d3 * u6
    e6 = d3 * u5
    b3 = e5 * u5
    d5 = e5.dot(t4)
    d6 = e6.dot(t5)
    b5 = d5 * u7
    d7 = d5.dot(t6)
    return {'x0': b0, 'x1': b1, 'x2': b2, 'x3': b3, 'x4': b4, 'x5': b5, 'x6': d6, 'x7': d7, 'x8': d8}
