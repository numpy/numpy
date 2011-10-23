import numpy as np
from numpy.compat import asbytes
from numpy.testing import *
import sys, warnings

# The C implementation of fancy indexing is relatively complicated,
# and has many seeming inconsistencies. It also appears to lack any
# kind of test suite, making any changes to the underlying code difficult
# because of its fragility.

def test_boolean_indexing():
    
    # Indexing a 1-dimensional array with a boolean array
    a = np.arange(12)
    b = np.array([True, False, True])
    assert_raises(ValueError, a.__getitem__, b)
    b.resize(12)
    assert_equal(a[b], [0, 2])
    
    # Indexing a 2-dimensional array with a boolean array
    a = np.arange(12).reshape((4,3))
    b = np.array([True, False, True])
    assert_raises(ValueError, a.__getitem__, b)
    b.resize((4,3))
    assert_equal(a[b], [0, 2])
    
    # Indexing and assigning a 2-dimensional array with a boolean array
    a = np.array([[ 0.,  0.,  0.]])
    b = np.array([[True, True, True]], dtype=bool)
    assert_equal(a[b], a.flatten())

    a[b] = 1.
    assert_equal(a, [[1., 1., 1.]])

if __name__ == "__main__":
    run_module_suite()
