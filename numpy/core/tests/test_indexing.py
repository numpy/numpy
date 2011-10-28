from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.compat import asbytes
from numpy.testing import *
import sys, warnings

# The C implementation of fancy indexing is relatively complicated,
# and has many seeming inconsistencies. It also appears to lack any
# kind of test suite, making any changes to the underlying code difficult
# because of its fragility.

# This file is to remedy the test suite part a little bit,
# but hopefully NumPy indexing can be changed to be more systematic
# at some point in the future.

class TestIndexing(TestCase):

    def test_none_index(self):
        # `None` index adds newaxis
        a = np.array([1, 2, 3])
        assert_equal(a[None], a[np.newaxis])
        assert_equal(a[None].ndim, a.ndim + 1)

    def test_empty_tuple_index(self):
        # Empty tuple index creates a view
        a = np.array([1, 2, 3])
        assert_equal(a[()], a)
        assert_(a[()].base is a)

    def _test_empty_list_index(self):
        # Empty list index (is buggy!)
        a = np.array([1, 2, 3])
        assert_equal(a[[]], a)

    def test_empty_array_index(self):
        # Empty array index is illegal
        a = np.array([1, 2, 3])
        b = np.array([])
        assert_raises(IndexError, a.__getitem__, b)

    def test_ellipsis_index(self):
        # Ellipsis index does not create a view
        a = np.array([[1, 2, 3],
                      [4 ,5, 6],
                      [7, 8, 9]])
        assert_equal(a[...], a)
        assert_(a[...] is a)

        # Slicing with ellipsis can skip an
        # arbitrary number of dimensions
        assert_equal(a[0, ...], a[0])
        assert_equal(a[0, ...], a[0, :])
        assert_equal(a[..., 0], a[:, 0])

        # Slicing with ellipsis always results
        # in an array, not a scalar
        assert_equal(a[0, ..., 1], np.array(2))

    def test_single_int_index(self):
        # Single integer index selects one row
        a = np.array([[1, 2, 3],
                      [4 ,5, 6],
                      [7, 8, 9]])

        assert_equal(a[0], [1, 2, 3])
        assert_equal(a[-1], [7, 8, 9])

        # Index out of bounds produces IndexError
        assert_raises(IndexError, a.__getitem__, 1<<30)
        # Index overflow produces ValueError
        assert_raises(ValueError, a.__getitem__, 1<<64)

    def _test_single_bool_index(self):
        # Single boolean index (is buggy?)
        a = np.array([[1, 2, 3],
                      [4 ,5, 6],
                      [7, 8, 9]])

        # Python boolean converts to integer (invalid?)
        assert_equal(a[True], a[1])

        # NumPy zero-dimensional boolean array (*crashes*)
        assert_equal(a[np.array(True)], a) # what should be the behaviour?
        assert_equal(a[np.array(False)], []) # what should be the behaviour?

    def test_boolean_indexing(self):
        # Indexing a 2-dimensional array with a length-1 array of 'True'
        a = np.array([[ 0.,  0.,  0.]])
        b = np.array([ True], dtype=bool)
        assert_equal(a[b], a)

        a[b] = 1.
        assert_equal(a, [[1., 1., 1.]])

if __name__ == "__main__":
    run_module_suite()
