import numpy as N
from numpy.testing.utils import *

class TestEqual:
    def _test_equal(self, a, b):
        assert_array_equal(a, b)

    def _test_not_equal(self, a, b):
        passed = False
        try:
            assert_array_equal(a, b)
            passed = True
        except AssertionError:
            pass

        if passed:
            raise AssertionError("a and b are found equal but are not")

    def test_array_rank1_eq(self):
        """Test two equal array are found equal."""
        a = N.array([1, 2])
        b = N.array([1, 2])

        self._test_equal(a, b)

    def test_array_rank1_noteq(self):
        a = N.array([1, 2])
        b = N.array([2, 2])

        self._test_not_equal(a, b)
