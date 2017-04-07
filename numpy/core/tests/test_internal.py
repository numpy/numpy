from __future__ import division, absolute_import, print_function

import ctypes
import numpy as np
from numpy.core import _internal
from numpy.testing import (
    TestCase, run_module_suite, assert_array_equal, assert_equal)


class InternalBaseTest(TestCase):
    """Base test class for the _internal module."""

    def tearDown(self):
        # Ensure that ctypes is set on _internal after tests run.
        _internal.ctypes = ctypes  


class CtypesTest(InternalBaseTest):
    """Tests for the _ctypes class."""

    def setUp(self):
        self.test_arr = np.array([[1, 2, 3], [4, 5, 6]])

    def test_ctypes_is_available(self):
        _ctypes = _internal._ctypes(self.test_arr)

        self.assertEqual(ctypes, _ctypes._ctypes)
        assert_array_equal(_ctypes._arr, self.test_arr)

    def test_ctypes_is_not_available(self):
        _internal.ctypes = None
        _ctypes = _internal._ctypes(self.test_arr)

        self.assertIsInstance(_ctypes._ctypes, _internal._missing_ctypes)
        assert_array_equal(_ctypes._arr, self.test_arr)


class GetIntPCtypeTest(InternalBaseTest):
    """Tests for the _getintp_ctype function."""

    def test_ctypes_is_available(self):
        assert_equal(_internal._getintp_ctype(), ctypes.c_long)

    def test_ctypes_is_not_available(self):
        _internal.ctypes = None
        assert_equal(
            _internal._getintp_ctype(), _internal.dummy_ctype(np.intp))


if __name__ == "__main__":
    run_module_suite()
