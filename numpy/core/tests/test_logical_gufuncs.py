from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import (
    run_module_suite, assert_equal
)

dtypes = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, 
          np.uint64, np.float32, np.float64]

# helper function
def check(f, x1, x2, dtype, expected):
    result = f(x1.astype(dtype), x2.astype(dtype))
    assert_equal(result, expected)


def test_generator():
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[0]):
            x1 = inputs[i, :]
            x2 = inputs[j, :]

            for dtype in dtypes:

                yield check, np.all_equal, x1, x2, dtype, (x1==x2).all()
                yield check, np.any_equal, x1, x2, dtype, (x1==x2).any()
                yield check, np.all_not_equal, x1, x2, dtype, (x1!=x2).all()
                yield check, np.any_not_equal, x1, x2, dtype, (x1!=x2).any()
                yield check, np.all_greater, x1, x2, dtype, (x1>x2).all()
                yield check, np.any_greater, x1, x2, dtype, (x1>x2).any()
                yield check, np.all_greater_equal, x1, x2, dtype, (x1>=x2).all()
                yield check, np.any_greater_equal, x1, x2, dtype, (x1>=x2).any()
                yield check, np.all_less, x1, x2, dtype, (x1<x2).all()
                yield check, np.any_less, x1, x2, dtype, (x1<x2).any()
                yield check, np.all_less_equal, x1, x2, dtype, (x1<=x2).all()
                yield check, np.any_less_equal, x1, x2, dtype, (x1<=x2).any()

if __name__ == "__main__":
    run_module_suite()
