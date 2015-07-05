from __future__ import division, absolute_import, print_function

import warnings
import sys

import numpy as np
from numpy.testing import (
    assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, build_err_msg, raises, assert_raises,
    assert_warns, assert_no_warnings, assert_allclose, assert_approx_equal,
    assert_array_almost_equal_nulp, assert_array_max_ulp,
    clear_and_catch_warnings, run_module_suite
    )
import unittest


class _GenericTest(object):

    def _test_equal(self, a, b):
        self._assert_func(a, b)

    def _test_not_equal(self, a, b):
        try:
            self._assert_func(a, b)
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

    def test_array_likes(self):
        self._test_equal([1, 2, 3], (1, 2, 3))


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


class TestBuildErrorMessage(unittest.TestCase):

    def test_build_err_msg_defaults(self):
        x = np.array([1.00001, 2.00002, 3.00003])
        y = np.array([1.00002, 2.00003, 3.00004])
        err_msg = 'There is a mismatch'

        a = build_err_msg([x, y], err_msg)
        b = ('\nItems are not equal: There is a mismatch\n ACTUAL: array([ '
             '1.00001,  2.00002,  3.00003])\n DESIRED: array([ 1.00002,  '
             '2.00003,  3.00004])')
        self.assertEqual(a, b)

    def test_build_err_msg_no_verbose(self):
        x = np.array([1.00001, 2.00002, 3.00003])
        y = np.array([1.00002, 2.00003, 3.00004])
        err_msg = 'There is a mismatch'

        a = build_err_msg([x, y], err_msg, verbose=False)
        b = '\nItems are not equal: There is a mismatch'
        self.assertEqual(a, b)

    def test_build_err_msg_custom_names(self):
        x = np.array([1.00001, 2.00002, 3.00003])
        y = np.array([1.00002, 2.00003, 3.00004])
        err_msg = 'There is a mismatch'

        a = build_err_msg([x, y], err_msg, names=('FOO', 'BAR'))
        b = ('\nItems are not equal: There is a mismatch\n FOO: array([ '
             '1.00001,  2.00002,  3.00003])\n BAR: array([ 1.00002,  2.00003,  '
             '3.00004])')
        self.assertEqual(a, b)

    def test_build_err_msg_custom_precision(self):
        x = np.array([1.000000001, 2.00002, 3.00003])
        y = np.array([1.000000002, 2.00003, 3.00004])
        err_msg = 'There is a mismatch'

        a = build_err_msg([x, y], err_msg, precision=10)
        b = ('\nItems are not equal: There is a mismatch\n ACTUAL: array([ '
             '1.000000001,  2.00002    ,  3.00003    ])\n DESIRED: array([ '
             '1.000000002,  2.00003    ,  3.00004    ])')
        self.assertEqual(a, b)


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
        self.assertRaises(AssertionError,
                lambda: self._assert_func(x, y, decimal=5))

    def test_nan(self):
        anan = np.array([np.nan])
        aone = np.array([1])
        ainf = np.array([np.inf])
        self._assert_func(anan, anan)
        self.assertRaises(AssertionError,
                lambda: self._assert_func(anan, aone))
        self.assertRaises(AssertionError,
                lambda: self._assert_func(anan, ainf))
        self.assertRaises(AssertionError,
                lambda: self._assert_func(ainf, anan))

    def test_inf(self):
        a = np.array([[1., 2.], [3., 4.]])
        b = a.copy()
        a[0, 0] = np.inf
        self.assertRaises(AssertionError,
                lambda: self._assert_func(a, b))

    def test_subclass(self):
        a = np.array([[1., 2.], [3., 4.]])
        b = np.ma.masked_array([[1., 2.], [0., 4.]],
                               [[False, False], [True, False]])
        assert_array_almost_equal(a, b)
        assert_array_almost_equal(b, a)
        assert_array_almost_equal(b, b)


class TestAlmostEqual(_GenericTest, unittest.TestCase):

    def setUp(self):
        self._assert_func = assert_almost_equal

    def test_nan_item(self):
        self._assert_func(np.nan, np.nan)
        self.assertRaises(AssertionError,
                lambda: self._assert_func(np.nan, 1))
        self.assertRaises(AssertionError,
                lambda: self._assert_func(np.nan, np.inf))
        self.assertRaises(AssertionError,
                lambda: self._assert_func(np.inf, np.nan))

    def test_inf_item(self):
        self._assert_func(np.inf, np.inf)
        self._assert_func(-np.inf, -np.inf)
        self.assertRaises(AssertionError,
                lambda: self._assert_func(np.inf, 1))

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

    def test_error_message(self):
        """Check the message is formatted correctly for the decimal value"""
        x = np.array([1.00000000001, 2.00000000002, 3.00003])
        y = np.array([1.00000000002, 2.00000000003, 3.00004])

        # test with a different amount of decimal digits
        # note that we only check for the formatting of the arrays themselves
        b = ('x: array([ 1.00000000001,  2.00000000002,  3.00003     '
             ' ])\n y: array([ 1.00000000002,  2.00000000003,  3.00004      ])')
        try:
            self._assert_func(x, y, decimal=12)
        except AssertionError as e:
            # remove anything that's not the array string
            self.assertEqual(str(e).split('%)\n ')[1], b)

        # with the default value of decimal digits, only the 3rd element differs
        # note that we only check for the formatting of the arrays themselves
        b = ('x: array([ 1.     ,  2.     ,  3.00003])\n y: array([ 1.     ,  '
             '2.     ,  3.00004])')
        try:
            self._assert_func(x, y)
        except AssertionError as e:
            # remove anything that's not the array string
            self.assertEqual(str(e).split('%)\n ')[1], b)


class TestApproxEqual(unittest.TestCase):

    def setUp(self):
        self._assert_func = assert_approx_equal

    def test_simple_arrays(self):
        x = np.array([1234.22])
        y = np.array([1234.23])

        self._assert_func(x, y, significant=5)
        self._assert_func(x, y, significant=6)
        self.assertRaises(AssertionError,
                lambda: self._assert_func(x, y, significant=7))

    def test_simple_items(self):
        x = 1234.22
        y = 1234.23

        self._assert_func(x, y, significant=4)
        self._assert_func(x, y, significant=5)
        self._assert_func(x, y, significant=6)
        self.assertRaises(AssertionError,
                lambda: self._assert_func(x, y, significant=7))

    def test_nan_array(self):
        anan = np.array(np.nan)
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(anan, anan)
        self.assertRaises(AssertionError,
                lambda: self._assert_func(anan, aone))
        self.assertRaises(AssertionError,
                lambda: self._assert_func(anan, ainf))
        self.assertRaises(AssertionError,
                lambda: self._assert_func(ainf, anan))

    def test_nan_items(self):
        anan = np.array(np.nan)
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(anan, anan)
        self.assertRaises(AssertionError,
                lambda: self._assert_func(anan, aone))
        self.assertRaises(AssertionError,
                lambda: self._assert_func(anan, ainf))
        self.assertRaises(AssertionError,
                lambda: self._assert_func(ainf, anan))


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
        raises(self.e)(self.raises_exception)(self.e)  # raises?

    def test_wrong_exception(self):
        try:
            raises(self.e)(self.raises_exception)(RuntimeError)  # raises?
        except RuntimeError:
            return
        else:
            raise AssertionError("should have caught RuntimeError")

    def test_catch_no_raise(self):
        try:
            raises(self.e)(self.does_not_raise_exception)()  # raises?
        except AssertionError:
            return
        else:
            raise AssertionError("should have raised an AssertionError")


class TestWarns(unittest.TestCase):

    def test_warn(self):
        def f():
            warnings.warn("yo")
            return 3

        before_filters = sys.modules['warnings'].filters[:]
        assert_equal(assert_warns(UserWarning, f), 3)
        after_filters = sys.modules['warnings'].filters

        assert_raises(AssertionError, assert_no_warnings, f)
        assert_equal(assert_no_warnings(lambda x: x, 1), 1)

        # Check that the warnings state is unchanged
        assert_equal(before_filters, after_filters,
                     "assert_warns does not preserver warnings state")

    def test_warn_wrong_warning(self):
        def f():
            warnings.warn("yo", DeprecationWarning)

        failed = False
        filters = sys.modules['warnings'].filters[:]
        try:
            try:
                # Should raise an AssertionError
                assert_warns(UserWarning, f)
                failed = True
            except AssertionError:
                pass
        finally:
            sys.modules['warnings'].filters = filters

        if failed:
            raise AssertionError("wrong warning caught by assert_warn")


class TestAssertAllclose(unittest.TestCase):
    
    def test_simple(self):
        x = 1e-3
        y = 1e-9

        assert_allclose(x, y, atol=1)
        self.assertRaises(AssertionError, assert_allclose, x, y)

        a = np.array([x, y, x, y])
        b = np.array([x, y, x, x])

        assert_allclose(a, b, atol=1)
        self.assertRaises(AssertionError, assert_allclose, a, b)

        b[-1] = y * (1 + 1e-8)
        assert_allclose(a, b)
        self.assertRaises(AssertionError, assert_allclose, a, b,
                          rtol=1e-9)

        assert_allclose(6, 10, rtol=0.5)
        self.assertRaises(AssertionError, assert_allclose, 10, 6, rtol=0.5)

    def test_min_int(self):
        a = np.array([np.iinfo(np.int_).min], dtype=np.int_)
        # Should not raise:
        assert_allclose(a, a)

    def test_report_fail_percentage(self):
        a = np.array([1, 1, 1, 1])
        b = np.array([1, 1, 1, 2])
        try:
            assert_allclose(a, b)
            msg = ''
        except AssertionError as exc:
            msg = exc.args[0]
        self.assertTrue("mismatch 25.0%" in msg)


class TestArrayAlmostEqualNulp(unittest.TestCase):

    def test_float64_pass(self):
        # The number of units of least precision
        # In this case, use a few places above the lowest level (ie nulp=1)
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10**x
        x = np.r_[-x, x]

        # Addition
        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp/2.
        assert_array_almost_equal_nulp(x, y, nulp)

        # Subtraction
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float64_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10**x
        x = np.r_[-x, x]

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp*2.
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          x, y, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          x, y, nulp)

    def test_float32_pass(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10**x
        x = np.r_[-x, x]

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp/2.
        assert_array_almost_equal_nulp(x, y, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float32_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10**x
        x = np.r_[-x, x]

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp*2.
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          x, y, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          x, y, nulp)

    def test_complex128_pass(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10**x
        x = np.r_[-x, x]
        xi = x + x*1j

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp/2.
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        # The test condition needs to be at least a factor of sqrt(2) smaller
        # because the real and imaginary parts both change
        y = x + x*eps*nulp/4.
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        y = x - x*epsneg*nulp/4.
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

    def test_complex128_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10**x
        x = np.r_[-x, x]
        xi = x + x*1j

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp*2.
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, x + y*1j, nulp)
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, y + x*1j, nulp)
        # The test condition needs to be at least a factor of sqrt(2) smaller
        # because the real and imaginary parts both change
        y = x + x*eps*nulp
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, y + y*1j, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, x + y*1j, nulp)
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, y + x*1j, nulp)
        y = x - x*epsneg*nulp
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, y + y*1j, nulp)

    def test_complex64_pass(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10**x
        x = np.r_[-x, x]
        xi = x + x*1j

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp/2.
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        y = x + x*eps*nulp/4.
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        y = x - x*epsneg*nulp/4.
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

    def test_complex64_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10**x
        x = np.r_[-x, x]
        xi = x + x*1j

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp*2.
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, x + y*1j, nulp)
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, y + x*1j, nulp)
        y = x + x*eps*nulp
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, y + y*1j, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, x + y*1j, nulp)
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, y + x*1j, nulp)
        y = x - x*epsneg*nulp
        self.assertRaises(AssertionError, assert_array_almost_equal_nulp,
                          xi, y + y*1j, nulp)


class TestULP(unittest.TestCase):

    def test_equal(self):
        x = np.random.randn(10)
        assert_array_max_ulp(x, x, maxulp=0)

    def test_single(self):
        # Generate 1 + small deviation, check that adding eps gives a few UNL
        x = np.ones(10).astype(np.float32)
        x += 0.01 * np.random.randn(10).astype(np.float32)
        eps = np.finfo(np.float32).eps
        assert_array_max_ulp(x, x+eps, maxulp=20)

    def test_double(self):
        # Generate 1 + small deviation, check that adding eps gives a few UNL
        x = np.ones(10).astype(np.float64)
        x += 0.01 * np.random.randn(10).astype(np.float64)
        eps = np.finfo(np.float64).eps
        assert_array_max_ulp(x, x+eps, maxulp=200)

    def test_inf(self):
        for dt in [np.float32, np.float64]:
            inf = np.array([np.inf]).astype(dt)
            big = np.array([np.finfo(dt).max])
            assert_array_max_ulp(inf, big, maxulp=200)

    def test_nan(self):
        # Test that nan is 'far' from small, tiny, inf, max and min
        for dt in [np.float32, np.float64]:
            if dt == np.float32:
                maxulp = 1e6
            else:
                maxulp = 1e12
            inf = np.array([np.inf]).astype(dt)
            nan = np.array([np.nan]).astype(dt)
            big = np.array([np.finfo(dt).max])
            tiny = np.array([np.finfo(dt).tiny])
            zero = np.array([np.PZERO]).astype(dt)
            nzero = np.array([np.NZERO]).astype(dt)
            self.assertRaises(AssertionError,
                                  lambda: assert_array_max_ulp(nan, inf,
                                                               maxulp=maxulp))
            self.assertRaises(AssertionError,
                                  lambda: assert_array_max_ulp(nan, big,
                                                               maxulp=maxulp))
            self.assertRaises(AssertionError,
                                  lambda: assert_array_max_ulp(nan, tiny,
                                                               maxulp=maxulp))
            self.assertRaises(AssertionError,
                                  lambda: assert_array_max_ulp(nan, zero,
                                                               maxulp=maxulp))
            self.assertRaises(AssertionError,
                                  lambda: assert_array_max_ulp(nan, nzero,
                                                               maxulp=maxulp))

def assert_warn_len_equal(mod, n_in_context):
    mod_warns = mod.__warningregistry__
    # Python 3.4 appears to clear any pre-existing warnings of the same type,
    # when raising warnings inside a catch_warnings block. So, there is a
    # warning generated by the tests within the context manager, but no
    # previous warnings.
    if 'version' in mod_warns:
        assert_equal(len(mod_warns), 2)  # including 'version'
    else:
        assert_equal(len(mod_warns), n_in_context)


def _get_fresh_mod():
    # Get this module, with warning registry empty
    my_mod = sys.modules[__name__]
    try:
        my_mod.__warningregistry__.clear()
    except AttributeError:
        pass
    return my_mod


def test_clear_and_catch_warnings():
    # Initial state of module, no warnings
    my_mod = _get_fresh_mod()
    assert_equal(getattr(my_mod, '__warningregistry__', {}), {})
    with clear_and_catch_warnings(modules=[my_mod]):
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_equal(my_mod.__warningregistry__, {})
    # Without specified modules, don't clear warnings during context
    with clear_and_catch_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 1)
    # Confirm that specifying module keeps old warning, does not add new
    with clear_and_catch_warnings(modules=[my_mod]):
        warnings.simplefilter('ignore')
        warnings.warn('Another warning')
    assert_warn_len_equal(my_mod, 1)
    # Another warning, no module spec does add to warnings dict, except on
    # Python 3.4 (see comments in `assert_warn_len_equal`)
    with clear_and_catch_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Another warning')
    assert_warn_len_equal(my_mod, 2)


class my_cacw(clear_and_catch_warnings):

    class_modules = (sys.modules[__name__],)


def test_clear_and_catch_warnings_inherit():
    # Test can subclass and add default modules
    my_mod = _get_fresh_mod()
    with my_cacw():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_equal(my_mod.__warningregistry__, {})


if __name__ == '__main__':
    run_module_suite()
