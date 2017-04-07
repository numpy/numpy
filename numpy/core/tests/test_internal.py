from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import (
    TestCase, assert_, run_module_suite, assert_array_equal)


class CtypesAvailabilityTest(TestCase):

    def setUp(self):
        self.test_arr = np.array([[1, 2, 3], [4, 5, 6]])

    def test_ctypes_is_available(self):
        import ctypes
        from numpy.core import _internal

        _ctypes = _internal._ctypes(self.test_arr)
        shape = _ctypes.get_shape()

        self.assertEqual(ctypes, _ctypes._ctypes)
        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0], 2)
        self.assertEqual(shape[1], 3)
        assert_array_equal(_ctypes._arr, self.test_arr)

    def test_ctypes_is_not_available(self):
        # Remove ctypes from sys.modules.
        from numpy.core import _internal
        _internal.ctypes = None

        _ctypes = _internal._ctypes(self.test_arr)
        shape = _ctypes.get_shape()

        self.assertIsInstance(_ctypes._ctypes, _internal._missing_ctypes)
        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0], 2)
        self.assertEqual(shape[1], 3)
        assert_array_equal(_ctypes._arr, self.test_arr)

if __name__ == "__main__":
    run_module_suite()
