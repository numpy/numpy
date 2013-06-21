from __future__ import division, absolute_import, print_function

import warnings

import numpy as np
from numpy.testing import (
    run_module_suite, TestCase, assert_, assert_equal, assert_almost_equal
    )
from numpy.lib import (
    nansum, nanmax, nanargmax, nanargmin, nanmin, nanmean, nanvar, nanstd
    )

class TestNaNFuncts(TestCase):
    def setUp(self):
        self.A = np.array([[[ np.nan, 0.01319214, 0.01620964],
                         [ 0.11704017, np.nan, 0.75157887],
                         [ 0.28333658, 0.1630199 , np.nan       ]],
                        [[ 0.59541557, np.nan, 0.37910852],
                         [ np.nan, 0.87964135, np.nan       ],
                         [ 0.70543747, np.nan, 0.34306596]],
                        [[ 0.72687499, 0.91084584, np.nan       ],
                         [ 0.84386844, 0.38944762, 0.23913896],
                         [ np.nan, 0.37068164, 0.33850425]]])

    def test_nansum(self):
        assert_almost_equal(nansum(self.A), 8.0664079100000006)
        assert_almost_equal(nansum(self.A, 0),
                            np.array([[ 1.32229056, 0.92403798, 0.39531816],
                                   [ 0.96090861, 1.26908897, 0.99071783],
                                   [ 0.98877405, 0.53370154, 0.68157021]]))
        assert_almost_equal(nansum(self.A, 1),
                            np.array([[ 0.40037675, 0.17621204, 0.76778851],
                                   [ 1.30085304, 0.87964135, 0.72217448],
                                   [ 1.57074343, 1.6709751 , 0.57764321]]))
        assert_almost_equal(nansum(self.A, 2),
                            np.array([[ 0.02940178, 0.86861904, 0.44635648],
                                   [ 0.97452409, 0.87964135, 1.04850343],
                                   [ 1.63772083, 1.47245502, 0.70918589]]))

    def test_nanmin(self):
        assert_almost_equal(nanmin(self.A), 0.01319214)
        assert_almost_equal(nanmin(self.A, 0),
                            np.array([[ 0.59541557, 0.01319214, 0.01620964],
                                   [ 0.11704017, 0.38944762, 0.23913896],
                                   [ 0.28333658, 0.1630199 , 0.33850425]]))
        assert_almost_equal(nanmin(self.A, 1),
                            np.array([[ 0.11704017, 0.01319214, 0.01620964],
                                   [ 0.59541557, 0.87964135, 0.34306596],
                                   [ 0.72687499, 0.37068164, 0.23913896]]))
        assert_almost_equal(nanmin(self.A, 2),
                            np.array([[ 0.01319214, 0.11704017, 0.1630199 ],
                                   [ 0.37910852, 0.87964135, 0.34306596],
                                   [ 0.72687499, 0.23913896, 0.33850425]]))
        assert_(np.isnan(nanmin([np.nan, np.nan])))

    def test_nanargmin(self):
        assert_almost_equal(nanargmin(self.A), 1)
        assert_almost_equal(nanargmin(self.A, 0),
                            np.array([[1, 0, 0],
                                   [0, 2, 2],
                                   [0, 0, 2]]))
        assert_almost_equal(nanargmin(self.A, 1),
                            np.array([[1, 0, 0],
                                   [0, 1, 2],
                                   [0, 2, 1]]))
        assert_almost_equal(nanargmin(self.A, 2),
                            np.array([[1, 0, 1],
                                   [2, 1, 2],
                                   [0, 2, 2]]))

    def test_nanmax(self):
        assert_almost_equal(nanmax(self.A), 0.91084584000000002)
        assert_almost_equal(nanmax(self.A, 0),
                            np.array([[ 0.72687499, 0.91084584, 0.37910852],
                                   [ 0.84386844, 0.87964135, 0.75157887],
                                   [ 0.70543747, 0.37068164, 0.34306596]]))
        assert_almost_equal(nanmax(self.A, 1),
                            np.array([[ 0.28333658, 0.1630199 , 0.75157887],
                                   [ 0.70543747, 0.87964135, 0.37910852],
                                   [ 0.84386844, 0.91084584, 0.33850425]]))
        assert_almost_equal(nanmax(self.A, 2),
                            np.array([[ 0.01620964, 0.75157887, 0.28333658],
                                   [ 0.59541557, 0.87964135, 0.70543747],
                                   [ 0.91084584, 0.84386844, 0.37068164]]))
        assert_(np.isnan(nanmax([np.nan, np.nan])))

    def test_nanmin_allnan_on_axis(self):
        assert_equal(np.isnan(nanmin([[np.nan] * 2] * 3, axis=1)),
                     [True, True, True])

    def test_nanmin_masked(self):
        a = np.ma.fix_invalid([[2, 1, 3, np.nan], [5, 2, 3, np.nan]])
        ctrl_mask = a._mask.copy()
        test = np.nanmin(a, axis=1)
        assert_equal(test, [1, 2])
        assert_equal(a._mask, ctrl_mask)
        assert_equal(np.isinf(a), np.zeros((2, 4), dtype=bool))

    def test_nanmean(self):
        A = [[1, np.nan, np.nan], [np.nan, 4, 5]]
        assert_(nanmean(A) == (10.0 / 3))
        assert_(all(nanmean(A,0) == np.array([1, 4, 5])))
        assert_(all(nanmean(A,1) == np.array([1, 4.5])))

    def test_nanstd(self):
        A = [[1, np.nan, np.nan], [np.nan, 4, 5]]
        assert_almost_equal(nanstd(A), 1.699673171197595)
        assert_almost_equal(nanstd(A,0), np.array([0.0, 0.0, 0.0]))
        assert_almost_equal(nanstd(A,1), np.array([0.0, 0.5]))

    def test_nanvar(self):
        A = [[1, np.nan, np.nan], [np.nan, 4, 5]]
        assert_almost_equal(nanvar(A), 2.88888888889)
        assert_almost_equal(nanvar(A,0), np.array([0.0, 0.0, 0.0]))
        assert_almost_equal(nanvar(A,1), np.array([0.0, 0.25]))


class TestNaNMean(TestCase):
    def setUp(self):
        self.A = np.array([1, np.nan, -1, np.nan, np.nan, 1, -1])
        self.B = np.array([np.nan, np.nan, np.nan, np.nan])
        self.real_mean = 0

    def test_basic(self):
        assert_almost_equal(nanmean(self.A),self.real_mean)

    def test_mutation(self):
        # Because of the "messing around" we do to replace NaNs with zeros
        # this is meant to ensure we don't actually replace the NaNs in the
        # actual _array.
        a_copy = self.A.copy()
        b_copy = self.B.copy()
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            a_ret = nanmean(self.A)
            assert_equal(self.A, a_copy)
            b_ret = nanmean(self.B)
            assert_equal(self.B, b_copy)

    def test_allnans(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(nanmean(self.B)))
            assert_(w[0].category is RuntimeWarning)

    def test_empty(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(nanmean(np.array([]))))
            assert_(w[0].category is RuntimeWarning)


class TestNaNStdVar(TestCase):
    def setUp(self):
        self.A = np.array([np.nan, 1, -1, np.nan, 1, np.nan, -1])
        self.B = np.array([np.nan, np.nan, np.nan, np.nan])
        self.real_var = 1

    def test_basic(self):
        assert_almost_equal(nanvar(self.A),self.real_var)
        assert_almost_equal(nanstd(self.A)**2,self.real_var)

    def test_mutation(self):
        # Because of the "messing around" we do to replace NaNs with zeros
        # this is meant to ensure we don't actually replace the NaNs in the
        # actual array.
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            a_copy = self.A.copy()
            b_copy = self.B.copy()
            a_ret = nanvar(self.A)
            assert_equal(self.A, a_copy)
            b_ret = nanstd(self.B)
            assert_equal(self.B, b_copy)

    def test_ddof1(self):
        mask = ~np.isnan(self.A)
        assert_almost_equal(nanvar(self.A,ddof=1),
                self.real_var*sum(mask)/float(sum(mask) - 1))
        assert_almost_equal(nanstd(self.A,ddof=1)**2,
                self.real_var*sum(mask)/float(sum(mask) - 1))

    def test_ddof2(self):
        mask = ~np.isnan(self.A)
        assert_almost_equal(nanvar(self.A,ddof=2),
                self.real_var*sum(mask)/float(sum(mask) - 2))
        assert_almost_equal(nanstd(self.A,ddof=2)**2,
                self.real_var*sum(mask)/float(sum(mask) - 2))

    def test_allnans(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(nanvar(self.B)))
            assert_(np.isnan(nanstd(self.B)))
            assert_(w[0].category is RuntimeWarning)

    def test_empty(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(nanvar(np.array([]))))
            assert_(np.isnan(nanstd(np.array([]))))
            assert_(w[0].category is RuntimeWarning)


class TestNanFunctsIntTypes(TestCase):

    int_types = (
            np.int8, np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)

    def setUp(self, *args, **kwargs):
        self.A = np.array([127, 39,  93,  87, 46])

    def integer_arrays(self):
        for dtype in self.int_types:
            yield self.A.astype(dtype)

    def test_nanmin(self):
        min_value = min(self.A)
        for A in self.integer_arrays():
            assert_equal(nanmin(A), min_value)

    def test_nanmax(self):
        max_value = max(self.A)
        for A in self.integer_arrays():
            assert_equal(nanmax(A), max_value)

    def test_nanargmin(self):
        min_arg = np.argmin(self.A)
        for A in self.integer_arrays():
            assert_equal(nanargmin(A), min_arg)

    def test_nanargmax(self):
        max_arg = np.argmax(self.A)
        for A in self.integer_arrays():
            assert_equal(nanargmax(A), max_arg)


if __name__ == "__main__":
    run_module_suite()
