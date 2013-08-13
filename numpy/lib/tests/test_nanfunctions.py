from __future__ import division, absolute_import, print_function

import warnings

import numpy as np
from numpy.testing import (
    run_module_suite, TestCase, assert_, assert_equal, assert_almost_equal,
    assert_raises
    )
from numpy.lib import (
    nansum, nanmax, nanargmax, nanargmin, nanmin, nanmean, nanvar, nanstd,
    NanWarning
    )


_ndat = np.array(
        [[ 0.6244, np.nan, 0.2692,  0.0116, np.nan, 0.1170],
         [ 0.5351, 0.9403, np.nan,  0.2100, 0.4759, 0.2833],
         [ np.nan, np.nan, np.nan,  0.1042, np.nan, 0.5954],
         [ 0.161 , np.nan, np.nan,  0.1859, 0.3146, np.nan]]
        )

# rows of _ndat with nans removed
_rdat = [
        np.array([ 0.6244, 0.2692, 0.0116, 0.1170]),
        np.array([ 0.5351, 0.9403, 0.2100, 0.4759, 0.2833]),
        np.array([ 0.1042, 0.5954]),
        np.array([ 0.1610, 0.1859, 0.3146])
       ]


class TestNanFunctions_MinMax(TestCase):

    nanfuncs = [nanmin, nanmax]
    stdfuncs = [np.min, np.max]

    def test_mutation(self):
        # Check that passed array is not modified.
        ndat = _ndat.copy()
        for f in self.nanfuncs:
            f(ndat)
            assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for axis in [None, 0, 1]:
                tgt = rf(mat, axis=axis, keepdims=True)
                res = nf(mat, axis=axis, keepdims=True)
                assert_(res.ndim == tgt.ndim)

    def test_out(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            resout = np.zeros(3)
            tgt = rf(mat, axis=1)
            res = nf(mat, axis=1, out=resout)
            assert_almost_equal(res, resout)
            assert_almost_equal(res, tgt)

    def test_dtype_from_input(self):
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                mat = np.eye(3, dtype=c)
                tgt = rf(mat, axis=1).dtype.type
                res = nf(mat, axis=1).dtype.type
                assert_(res is tgt)
                # scalar case
                tgt = rf(mat, axis=None).dtype.type
                res = nf(mat, axis=None).dtype.type
                assert_(res is tgt)

    def test_result_values(self):
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            tgt = [rf(d) for d in _rdat]
            res = nf(_ndat, axis=1)
            assert_almost_equal(res, tgt)

    def test_allnans(self):
        mat = np.array([np.nan]*9).reshape(3, 3)
        for f in self.nanfuncs:
            for axis in [None, 0, 1]:
                assert_(np.isnan(f(mat, axis=axis)).all())

    def test_masked(self):
        mat = np.ma.fix_invalid(_ndat)
        msk = mat._mask.copy()
        for f in [nanmin]:
            res = f(mat, axis=1)
            tgt = f(_ndat, axis=1)
            assert_equal(res, tgt)
            assert_equal(mat._mask, msk)
            assert_(not np.isinf(mat).any())


class TestNanFunctions_ArgminArgmax(TestCase):

    nanfuncs = [nanargmin, nanargmax]

    def test_mutation(self):
        # Check that passed array is not modified.
        ndat = _ndat.copy()
        for f in self.nanfuncs:
            f(ndat)
            assert_equal(ndat, _ndat)

    def test_result_values(self):
        for f, fcmp in zip(self.nanfuncs, [np.greater, np.less]):
            for row in _ndat:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ind = f(row)
                    val = row[ind]
                    # comparing with NaN is tricky as the result
                    # is always false except for NaN != NaN
                    assert_(not np.isnan(val))
                    assert_(not fcmp(val, row).any())
                    assert_(not np.equal(val, row[:ind]).any())

    def test_allnans(self):
        mat = np.array([np.nan]*9).reshape(3, 3)
        tgt = np.iinfo(np.intp).min
        for f in self.nanfuncs:
            for axis in [None, 0, 1]:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    res = f(mat, axis=axis)
                    assert_((res == tgt).all())
                    assert_(len(w) == 1)
                    assert_(issubclass(w[0].category, NanWarning))

    def test_empty(self):
        mat = np.zeros((0,3))
        for f in self.nanfuncs:
            for axis in [0, None]:
                assert_raises(ValueError, f, mat, axis=axis)
            for axis in [1]:
                res = f(mat, axis=axis)
                assert_equal(res, np.zeros(0))


class TestNanFunctions_IntTypes(TestCase):

    int_types = (
            np.int8, np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)

    def setUp(self, *args, **kwargs):
        self.mat = np.array([127, 39,  93,  87, 46])

    def integer_arrays(self):
        for dtype in self.int_types:
            yield self.mat.astype(dtype)

    def test_nanmin(self):
        min_value = min(self.mat)
        for mat in self.integer_arrays():
            assert_equal(nanmin(mat), min_value)

    def test_nanmax(self):
        max_value = max(self.mat)
        for mat in self.integer_arrays():
            assert_equal(nanmax(mat), max_value)

    def test_nanargmin(self):
        min_arg = np.argmin(self.mat)
        for mat in self.integer_arrays():
            assert_equal(nanargmin(mat), min_arg)

    def test_nanargmax(self):
        max_arg = np.argmax(self.mat)
        for mat in self.integer_arrays():
            assert_equal(nanargmax(mat), max_arg)


class TestNanFunctions_Sum(TestCase):

    def test_mutation(self):
        # Check that passed array is not modified.
        ndat = _ndat.copy()
        nansum(ndat)
        assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for axis in [None, 0, 1]:
            tgt = np.sum(mat, axis=axis, keepdims=True)
            res = nansum(mat, axis=axis, keepdims=True)
            assert_(res.ndim == tgt.ndim)

    def test_out(self):
        mat = np.eye(3)
        resout = np.zeros(3)
        tgt = np.sum(mat, axis=1)
        res = nansum(mat, axis=1, out=resout)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)

    def test_dtype_from_dtype(self):
        mat = np.eye(3)
        codes = 'efdgFDG'
        for c in codes:
            tgt = np.sum(mat, dtype=np.dtype(c), axis=1).dtype.type
            res = nansum(mat, dtype=np.dtype(c), axis=1).dtype.type
            assert_(res is tgt)
            # scalar case
            tgt = np.sum(mat, dtype=np.dtype(c), axis=None).dtype.type
            res = nansum(mat, dtype=np.dtype(c), axis=None).dtype.type
            assert_(res is tgt)

    def test_dtype_from_char(self):
        mat = np.eye(3)
        codes = 'efdgFDG'
        for c in codes:
            tgt = np.sum(mat, dtype=c, axis=1).dtype.type
            res = nansum(mat, dtype=c, axis=1).dtype.type
            assert_(res is tgt)
            # scalar case
            tgt = np.sum(mat, dtype=c, axis=None).dtype.type
            res = nansum(mat, dtype=c, axis=None).dtype.type
            assert_(res is tgt)

    def test_dtype_from_input(self):
        codes = 'efdgFDG'
        for c in codes:
            mat = np.eye(3, dtype=c)
            tgt = np.sum(mat, axis=1).dtype.type
            res = nansum(mat, axis=1).dtype.type
            assert_(res is tgt)
            # scalar case
            tgt = np.sum(mat, axis=None).dtype.type
            res = nansum(mat, axis=None).dtype.type
            assert_(res is tgt)

    def test_result_values(self):
            tgt = [np.sum(d) for d in _rdat]
            res = nansum(_ndat, axis=1)
            assert_almost_equal(res, tgt)

    def test_allnans(self):
        # Check for FutureWarning and later change of return from
        # NaN to zero.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = nansum([np.nan]*3, axis=None)
            if np.__version__[:3] < '1.9':
                assert_(np.isnan(res), 'result is not NaN')
                assert_(len(w) == 1, 'no warning raised')
                assert_(issubclass(w[0].category, FutureWarning))
            else:
                assert_(res == 0, 'result is not 0')
                assert_(len(w) == 0, 'warning raised')

    def test_empty(self):
        mat = np.zeros((0,3))
        if np.__version__[:3] < '1.9':
            tgt = [np.nan]*3
            res = nansum(mat, axis=0)
            assert_equal(res, tgt)
            tgt = []
            res = nansum(mat, axis=1)
            assert_equal(res, tgt)
            tgt = np.nan
            res = nansum(mat, axis=None)
            assert_equal(res, tgt)
        else:
            tgt = [0]*3
            res = nansum(mat, axis=0)
            assert_equal(res, tgt)
            tgt = []
            res = nansum(mat, axis=1)
            assert_equal(res, tgt)
            tgt = 0
            res = nansum(mat, axis=None)
            assert_equal(res, tgt)


class TestNanFunctions_MeanVarStd(TestCase):

    nanfuncs = [nanmean, nanvar, nanstd]
    stdfuncs = [np.mean, np.var, np.std]

    def test_mutation(self):
        # Check that passed array is not modified.
        ndat = _ndat.copy()
        for f in self.nanfuncs:
            f(ndat)
            assert_equal(ndat, _ndat)

    def test_dtype_error(self):
        for f in self.nanfuncs:
            for dtype in [np.bool_, np.int_, np.object]:
                assert_raises( TypeError, f, _ndat, axis=1, dtype=np.int)

    def test_out_dtype_error(self):
        for f in self.nanfuncs:
            for dtype in [np.bool_, np.int_, np.object]:
                out = np.empty(_ndat.shape[0], dtype=dtype)
                assert_raises( TypeError, f, _ndat, axis=1, out=out)

    def test_keepdims(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for axis in [None, 0, 1]:
                tgt = rf(mat, axis=axis, keepdims=True)
                res = nf(mat, axis=axis, keepdims=True)
                assert_(res.ndim == tgt.ndim)

    def test_out(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            resout = np.zeros(3)
            tgt = rf(mat, axis=1)
            res = nf(mat, axis=1, out=resout)
            assert_almost_equal(res, resout)
            assert_almost_equal(res, tgt)

    def test_dtype_from_dtype(self):
        mat = np.eye(3)
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                tgt = rf(mat, dtype=np.dtype(c), axis=1).dtype.type
                res = nf(mat, dtype=np.dtype(c), axis=1).dtype.type
                assert_(res is tgt)
                # scalar case
                tgt = rf(mat, dtype=np.dtype(c), axis=None).dtype.type
                res = nf(mat, dtype=np.dtype(c), axis=None).dtype.type
                assert_(res is tgt)

    def test_dtype_from_char(self):
        mat = np.eye(3)
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                tgt = rf(mat, dtype=c, axis=1).dtype.type
                res = nf(mat, dtype=c, axis=1).dtype.type
                assert_(res is tgt)
                # scalar case
                tgt = rf(mat, dtype=c, axis=None).dtype.type
                res = nf(mat, dtype=c, axis=None).dtype.type
                assert_(res is tgt)

    def test_dtype_from_input(self):
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                mat = np.eye(3, dtype=c)
                tgt = rf(mat, axis=1).dtype.type
                res = nf(mat, axis=1).dtype.type
                assert_(res is tgt)
                # scalar case
                tgt = rf(mat, axis=None).dtype.type
                res = nf(mat, axis=None).dtype.type
                assert_(res is tgt)

    def test_ddof(self):
        nanfuncs = [nanvar, nanstd]
        stdfuncs = [np.var, np.std]
        for nf, rf in zip(nanfuncs, stdfuncs):
            for ddof in [0, 1]:
                tgt = [rf(d, ddof=ddof) for d in _rdat]
                res = nf(_ndat, axis=1, ddof=ddof)
                assert_almost_equal(res, tgt)

    def test_ddof_too_big(self):
        nanfuncs = [nanvar, nanstd]
        stdfuncs = [np.var, np.std]
        dsize = [len(d) for d in _rdat]
        for nf, rf in zip(nanfuncs, stdfuncs):
            for ddof in range(5):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    tgt = [ddof >= d for d in dsize]
                    res = nf(_ndat, axis=1, ddof=ddof)
                    assert_equal(np.isnan(res), tgt)
                    if any(tgt):
                        assert_(len(w) == 1)
                        assert_(issubclass(w[0].category, NanWarning))
                    else:
                        assert_(len(w) == 0)

    def test_result_values(self):
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            tgt = [rf(d) for d in _rdat]
            res = nf(_ndat, axis=1)
            assert_almost_equal(res, tgt)

    def test_allnans(self):
        mat = np.array([np.nan]*9).reshape(3, 3)
        for f in self.nanfuncs:
            for axis in [None, 0, 1]:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    assert_(np.isnan(f(mat, axis=axis)).all())
                    assert_(len(w) == 1)
                    assert_(issubclass(w[0].category, NanWarning))

    def test_empty(self):
        mat = np.zeros((0,3))
        for f in self.nanfuncs:
            for axis in [0, None]:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    assert_(np.isnan(f(mat, axis=axis)).all())
                    assert_(len(w) == 1)
                    assert_(issubclass(w[0].category, NanWarning))
            for axis in [1]:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    assert_equal(f(mat, axis=axis), np.zeros([]))
                    assert_(len(w) == 0)


if __name__ == "__main__":
    run_module_suite()
