from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import (
    run_module_suite, assert_equal
)

float_types = [np.float32, np.float64, np.longdouble]
complex_types = [np.cfloat, np.cdouble, np.clongdouble]
int_types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, 
          np.uint64, np.longlong, np.ulonglong]

# helper functions
def check(f, x1, x2, expected):
    assert_equal(f(x1, x2), expected)


def check_all(x1, x2):
    yield check, np.all_equal,          x1, x2, (x1==x2).all()
    yield check, np.any_equal,          x1, x2, (x1==x2).any()
    yield check, np.all_not_equal,      x1, x2, (x1!=x2).all()
    yield check, np.any_not_equal,      x1, x2, (x1!=x2).any()
    yield check, np.all_greater,        x1, x2, (x1>x2).all()
    yield check, np.any_greater,        x1, x2, (x1>x2).any()
    yield check, np.all_greater_equal,  x1, x2, (x1>=x2).all()
    yield check, np.any_greater_equal,  x1, x2, (x1>=x2).any()
    yield check, np.all_less,           x1, x2, (x1<x2).all()
    yield check, np.any_less,           x1, x2, (x1<x2).any()
    yield check, np.all_less_equal,     x1, x2, (x1<=x2).all()
    yield check, np.any_less_equal,     x1, x2, (x1<=x2).any()


def test_real():
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[0]):
            for dtype in int_types + float_types + complex_types:
                x1 = inputs[i, :].astype(dtype)
                x2 = inputs[j, :].astype(dtype)
                for x in check_all(x1, x2):
                    yield x


def test_complex():
    j = 1j
    for m in range(-1, 2):
        for n in range(-1, 2):
            for dtype in complex_types:
                x1 = np.zeros(2, dtype=dtype)
                x2 = x1 + m + n * j
                for x in check_all(x1, x2):
                    yield x


if __name__ == "__main__":
    run_module_suite()
