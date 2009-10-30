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

class TestArrayEqual(_GenericTest, unittest.TestCase):
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

class TestEqual(TestArrayEqual):
    def setUp(self):
        self._assert_func = assert_equal

    def test_nan_items(self):
        self._assert_func(np.nan, np.nan)
        self._assert_func([np.nan], [np.nan])
        self._test_not_equal(np.nan, [np.nan])
        self._test_not_equal(np.nan, 1)

    def test_inf_items(self):
        self._assert_func(np.inf, np.inf)
        self._assert_func([np.inf], [np.inf])
        self._test_not_equal(np.inf, [np.inf])

    def test_non_numeric(self):
        self._assert_func('ab', 'ab')
        self._test_not_equal('ab', 'abb')

    def test_complex_item(self):
        self._assert_func(complex(1, 2), complex(1, 2))
        self._assert_func(complex(1, np.nan), complex(1, np.nan))
        self._test_not_equal(complex(1, np.nan), complex(1, 2))
        self._test_not_equal(complex(np.nan, 1), complex(1, np.nan))
        self._test_not_equal(complex(np.nan, np.inf), complex(np.nan, 2))

    def test_negative_zero(self):
        self._test_not_equal(np.PZERO, np.NZERO)

    def test_complex(self):
        x = np.array([complex(1, 2), complex(1, np.nan)])
        y = np.array([complex(1, 2), complex(1, 2)])
        self._assert_func(x, x)
        self._test_not_equal(x, y)

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

    def test_complex_item(self):
        self._assert_func(complex(1, 2), complex(1, 2))
        self._assert_func(complex(1, np.nan), complex(1, np.nan))
        self._assert_func(complex(np.inf, np.nan), complex(np.inf, np.nan))
        self._test_not_equal(complex(1, np.nan), complex(1, 2))
        self._test_not_equal(complex(np.nan, 1), complex(1, np.nan))
        self._test_not_equal(complex(np.nan, np.inf), complex(np.nan, 2))

    def test_complex(self):
        x = np.array([complex(1, 2), complex(1, np.nan)])
        z = np.array([complex(1, 2), complex(np.nan, 1)])
        y = np.array([complex(1, 2), complex(1, 2)])
        self._assert_func(x, x)
        self._test_not_equal(x, y)
        self._test_not_equal(x, z)

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

class TestSpacing(unittest.TestCase):
    def test_one(self):
        for dt, dec in zip([np.float32, np.float64], (10, 20)):
            x = np.array(1, dtype=dt)
            # In theory, eps and spacing(1) should be exactly equal
            assert_array_almost_equal(spacing(x), np.finfo(dt).eps, decimal=dec)

    def test_simple(self):
        # Reference from this fortran file, built with gfortran 4.3.3 on linux
        # 32bits:
        #       PROGRAM test_spacing
        #        INTEGER, PARAMETER :: SGL = SELECTED_REAL_KIND(p=6, r=37)
        #        INTEGER, PARAMETER :: DBL = SELECTED_REAL_KIND(p=13, r=200)
        #
        #        WRITE(*,*) spacing(0.00001_DBL)
        #        WRITE(*,*) spacing(1.0_DBL)
        #        WRITE(*,*) spacing(1000._DBL)
        #        WRITE(*,*) spacing(10500._DBL)
        #
        #        WRITE(*,*) spacing(0.00001_SGL)
        #        WRITE(*,*) spacing(1.0_SGL)
        #        WRITE(*,*) spacing(1000._SGL)
        #        WRITE(*,*) spacing(10500._SGL)
        #       END PROGRAM
        ref = {}
        ref[np.float64] = [1.69406589450860068E-021,
               2.22044604925031308E-016,
               1.13686837721616030E-013,
               1.81898940354585648E-012]
        ref[np.float32] = [
                9.09494702E-13,
                1.19209290E-07,
                6.10351563E-05,
                9.76562500E-04]

        for dt, dec in zip([np.float32, np.float64], (10, 20)):
            x = np.array([1e-5, 1, 1000, 10500], dtype=dt)
            assert_array_almost_equal(spacing(x), ref[dt], decimal=dec)

if __name__ == '__main__':
    run_module_suite()
