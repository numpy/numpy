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
        """Test two equal array of rank 1 are found equal."""
        a = N.array([1, 2])
        b = N.array([1, 2])

        self._test_equal(a, b)

    def test_array_rank1_noteq(self):
        """Test two different array of rank 1 are found not equal."""
        a = N.array([1, 2])
        b = N.array([2, 2])

        self._test_not_equal(a, b)

    def test_array_rank2_eq(self):
        """Test two equal array of rank 2 are found equal."""
        a = N.array([[1, 2], [3, 4]])
        b = N.array([[1, 2], [3, 4]])

        self._test_equal(a, b)

    def test_array_diffshape(self):
        """Test two arrays with different shapes are found not equal."""
        a = N.array([1, 2])
        b = N.array([[1, 2], [1, 2]])

        self._test_not_equal(a, b)

    def test_nan_array(self):
        """Test two arrays with different shapes are found not equal."""
        a = N.array([1, 2])
        b = N.array([[1, 2], [1, 2]])

        self._test_not_equal(a, b)

    def test_string_arrays(self):
        """Test two arrays with different shapes are found not equal."""
        a = N.array(['floupi', 'floupa'])
        b = N.array(['floupi', 'floupa'])

        self._test_equal(a, b)

        c = N.array(['floupipi', 'floupa'])

        self._test_not_equal(c, b)

    def test_recarrays(self):
        """Test record arrays."""
        a = N.empty(2, [('floupi', N.float), ('floupa', N.float)])
        a['floupi'] = [1, 2]
        a['floupa'] = [1, 2]
        b = a.copy()

        self._test_equal(a, b)

        c = N.empty(2, [('floupipi', N.float), ('floupa', N.float)])
        c['floupipi'] = a['floupi'].copy()
        c['floupa'] = a['floupa'].copy()

        self._test_not_equal(c, b)

    def test_generic_rank1(self):
        """Test rank 1 array for all dtypes."""
        def foo(t):
            a = N.empty(2, t)
            a.fill(1)
            b = a.copy()
            c = a.copy()
            c.fill(0)
            self._test_equal(a, b)
            self._test_not_equal(c, b)

        # Test numeric types and object
        for t in '?bhilqpBHILQPfdgFDG':
            foo(t)

        # Test strings
        for t in ['S1', 'U1']:
            foo(t)
