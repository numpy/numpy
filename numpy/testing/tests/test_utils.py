import numpy as np
from numpy.testing import *
import unittest

class _GenericTest(object):
    def _test_equal(self, a, b):
        self._assert_func(a, b)

    def _test_not_equal(self, a, b):
        try:
            self._assert_func(a, b)
            passed = True
        except AssertionError:
            pass
        else:
            raise AssertionError("a and b are found equal but are not")

    def test_array_rank1_eq(self):
        """Test two equal array of rank 1 are found equal."""
        a = np.array([1, 2])
        b = np.array([1, 2])

        self._test_equal(a, b)

    def test_array_rank1_noteq(self):
        """Test two different array of rank 1 are found not equal."""
        a = np.array([1, 2])
        b = np.array([2, 2])

        self._test_not_equal(a, b)

    def test_array_rank2_eq(self):
        """Test two equal array of rank 2 are found equal."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 4]])

        self._test_equal(a, b)

    def test_array_diffshape(self):
        """Test two arrays with different shapes are found not equal."""
        a = np.array([1, 2])
        b = np.array([[1, 2], [1, 2]])

        self._test_not_equal(a, b)

    def test_objarray(self):
        """Test object arrays."""
        a = np.array([1, 1], dtype=np.object)
        self._test_equal(a, 1)

class TestEqual(_GenericTest, unittest.TestCase):
    def setUp(self):
        self._assert_func = assert_array_equal

    def test_generic_rank1(self):
        """Test rank 1 array for all dtypes."""
        def foo(t):
            a = np.empty(2, t)
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

    def test_generic_rank3(self):
        """Test rank 3 array for all dtypes."""
        def foo(t):
            a = np.empty((4, 2, 3), t)
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

    def test_nan_array(self):
        """Test arrays with nan values in them."""
        a = np.array([1, 2, np.nan])
        b = np.array([1, 2, np.nan])

        self._test_equal(a, b)

        c = np.array([1, 2, 3])
        self._test_not_equal(c, b)

    def test_string_arrays(self):
        """Test two arrays with different shapes are found not equal."""
        a = np.array(['floupi', 'floupa'])
        b = np.array(['floupi', 'floupa'])

        self._test_equal(a, b)

        c = np.array(['floupipi', 'floupa'])

        self._test_not_equal(c, b)

    def test_recarrays(self):
        """Test record arrays."""
        a = np.empty(2, [('floupi', np.float), ('floupa', np.float)])
        a['floupi'] = [1, 2]
        a['floupa'] = [1, 2]
        b = a.copy()

        self._test_equal(a, b)

        c = np.empty(2, [('floupipi', np.float), ('floupa', np.float)])
        c['floupipi'] = a['floupi'].copy()
        c['floupa'] = a['floupa'].copy()

        self._test_not_equal(c, b)


class TestArrayAlmostEqual(_GenericTest, unittest.TestCase):
    def setUp(self):
        self._assert_func = assert_array_almost_equal

    def test_simple(self):
        x = np.array([1234.2222])
        y = np.array([1234.2223])

        self._assert_func(x, y, decimal=3)
        self._assert_func(x, y, decimal=4)
        self.failUnlessRaises(AssertionError,
                lambda: self._assert_func(x, y, decimal=5))

    def test_nan(self):
        anan = np.array([np.nan])
        aone = np.array([1])
        ainf = np.array([np.inf])
        self._assert_func(anan, anan)
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(anan, aone))
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(anan, ainf))
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(ainf, anan))

class TestAlmostEqual(_GenericTest, unittest.TestCase):
    def setUp(self):
        self._assert_func = assert_almost_equal

    def test_nan_item(self):
        self._assert_func(np.nan, np.nan)
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(np.nan, 1))
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(np.nan, np.inf))
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(np.inf, np.nan))

    def test_inf_item(self):
        self._assert_func(np.inf, np.inf)
        self._assert_func(-np.inf, -np.inf)

    def test_simple_item(self):
        self._test_not_equal(1, 2)

class TestApproxEqual(unittest.TestCase):
    def setUp(self):
        self._assert_func = assert_approx_equal

    def test_simple_arrays(self):
        x = np.array([1234.22])
        y = np.array([1234.23])

        self._assert_func(x, y, significant=5)
        self._assert_func(x, y, significant=6)
        self.failUnlessRaises(AssertionError,
                lambda: self._assert_func(x, y, significant=7))

    def test_simple_items(self):
        x = 1234.22
        y = 1234.23

        self._assert_func(x, y, significant=4)
        self._assert_func(x, y, significant=5)
        self._assert_func(x, y, significant=6)
        self.failUnlessRaises(AssertionError,
                lambda: self._assert_func(x, y, significant=7))

    def test_nan_array(self):
        anan = np.array(np.nan)
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(anan, anan)
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(anan, aone))
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(anan, ainf))
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(ainf, anan))

    def test_nan_items(self):
        anan = np.array(np.nan)
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(anan, anan)
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(anan, aone))
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(anan, ainf))
        self.failUnlessRaises(AssertionError,
                lambda : self._assert_func(ainf, anan))

class TestRaises(unittest.TestCase):
    def setUp(self):
        class MyException(Exception):
            pass

        self.e = MyException

    def raises_exception(self, e):
        raise e

    def does_not_raise_exception(self):
        pass

    def test_correct_catch(self):
        f = raises(self.e)(self.raises_exception)(self.e)

    def test_wrong_exception(self):
        try:
            f = raises(self.e)(self.raises_exception)(RuntimeError)
        except RuntimeError:
            return
        else:
            raise AssertionError("should have caught RuntimeError")

    def test_catch_no_raise(self):
        try:
            f = raises(self.e)(self.does_not_raise_exception)()
        except AssertionError:
            return
        else:
            raise AssertionError("should have raised an AssertionError")


if __name__ == '__main__':
    run_module_suite()
