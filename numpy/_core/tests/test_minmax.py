"""Tests for the fused ``minimummaximum`` ufunc and the ``np.minmax`` wrapper."""
import pytest

import numpy as np
from numpy._core.umath import minimummaximum
from numpy.testing import assert_array_equal, assert_equal

# integer + floating types covered by the fused loops
INT_DTYPES = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
              'int64', 'uint64']
FLOAT_DTYPES = ['float32', 'float64', 'longdouble']
ALL_DTYPES = INT_DTYPES + FLOAT_DTYPES


def _sample(rng, n, dtype):
    """Random data of the given dtype (no lossy float->int casts)."""
    dt = np.dtype(dtype)
    if dt.kind == 'f':
        return (rng.standard_normal(n) * 50).astype(dt)
    lo = 0 if dt.kind == 'u' else -100
    return rng.integers(lo, 100, size=n, dtype=dt)


class TestMinimumMaximum:
    # lengths chosen to exercise the SIMD main loop and every tail
    LENGTHS = [0, 1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 63, 64, 65, 127, 1000, 4099]

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("n", LENGTHS)
    def test_forward_matches_minimum_maximum(self, dtype, n):
        rng = np.random.default_rng(n)
        a = _sample(rng, n, dtype)
        b = _sample(rng, n, dtype)
        lo, hi = minimummaximum(a, b)
        assert_array_equal(lo, np.minimum(a, b))
        assert_array_equal(hi, np.maximum(a, b))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward_nan_propagation(self, dtype):
        a = np.array([1, np.nan, 3, np.nan], dtype=dtype)
        b = np.array([np.nan, 2, 4, np.nan], dtype=dtype)
        lo, hi = minimummaximum(a, b)
        assert_array_equal(lo, np.minimum(a, b))
        assert_array_equal(hi, np.maximum(a, b))

    def test_forward_broadcasting(self):
        a = np.arange(6).reshape(2, 3)
        lo, hi = minimummaximum(a, np.array([2, 1, 4]))
        assert_array_equal(lo, np.minimum(a, [2, 1, 4]))
        assert_array_equal(hi, np.maximum(a, [2, 1, 4]))

    def test_forward_strided(self):
        a = np.arange(200, dtype=np.float64)[::3]
        b = np.arange(200, 400, dtype=np.float64)[::3][::-1]
        lo, hi = minimummaximum(a, b)
        assert_array_equal(lo, np.minimum(a, b))
        assert_array_equal(hi, np.maximum(a, b))

    def test_forward_out(self):
        a = np.array([2.0, 3.0, 4.0])
        b = np.array([1.0, 5.0, 2.0])
        o1 = np.empty(3)
        o2 = np.empty(3)
        r = minimummaximum(a, b, out=(o1, o2))
        assert r[0] is o1 and r[1] is o2
        assert_array_equal(o1, [1.0, 3.0, 2.0])
        assert_array_equal(o2, [2.0, 5.0, 4.0])

    def test_arity(self):
        assert minimummaximum.nin == 2
        assert minimummaximum.nout == 2

    def test_forward_bool(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 64).astype(bool)
        b = rng.integers(0, 2, 64).astype(bool)
        lo, hi = minimummaximum(a, b)
        assert_array_equal(lo, np.minimum(a, b))
        assert_array_equal(hi, np.maximum(a, b))

    @pytest.mark.parametrize("ct", ['complex64', 'complex128', 'clongdouble'])
    def test_forward_complex(self, ct):
        rng = np.random.default_rng(1)
        a = (rng.standard_normal(50) + 1j * rng.standard_normal(50)).astype(ct)
        b = (rng.standard_normal(50) + 1j * rng.standard_normal(50)).astype(ct)
        a[2] = complex(np.nan, 1)
        lo, hi = minimummaximum(a, b)
        assert_array_equal(lo, np.minimum(a, b))
        assert_array_equal(hi, np.maximum(a, b))

    def test_forward_datetime_nat(self):
        a = np.array(['2020-01-01', 'NaT', '2021-03-03'], dtype='datetime64[D]')
        b = np.array(['2020-06-01', '2019-01-01', 'NaT'], dtype='datetime64[D]')
        lo, hi = minimummaximum(a, b)
        assert_array_equal(lo, np.minimum(a, b))
        assert_array_equal(hi, np.maximum(a, b))

    def test_forward_object(self):
        a = np.array([3, 1, 4, 1, 5], dtype=object)
        b = np.array([2, 7, 0, 9, 5], dtype=object)
        lo, hi = minimummaximum(a, b)
        assert_array_equal(lo, np.minimum(a, b))
        assert_array_equal(hi, np.maximum(a, b))


class TestMinimumMaximumReduce:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("n", [1, 2, 8, 17, 64, 65, 1000, 4099])
    def test_reduce_1d(self, dtype, n):
        rng = np.random.default_rng(n)
        a = _sample(rng, n, dtype)
        lo, hi = minimummaximum.reduce(a)
        assert_equal(lo, np.minimum.reduce(a))
        assert_equal(hi, np.maximum.reduce(a))

    @pytest.mark.parametrize("axis", [0, 1, 2, (0, 1), (1, 2), (0, 2),
                                      (0, 1, 2), None])
    def test_reduce_axes(self, axis):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((13, 17, 23))
        lo, hi = minimummaximum.reduce(a, axis=axis)
        assert_array_equal(lo, np.minimum.reduce(a, axis=axis))
        assert_array_equal(hi, np.maximum.reduce(a, axis=axis))

    @pytest.mark.parametrize("pos", [0, 3, -1])
    def test_reduce_nan(self, pos):
        a = np.arange(64.0)
        a[pos] = np.nan
        lo, hi = minimummaximum.reduce(a)
        assert np.isnan(lo) and np.isnan(hi)

    def test_reduce_strided_and_fortran(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((20, 30))
        for view in (a[::2], a[:, ::3], a.T, np.asfortranarray(a)):
            for axis in (0, 1):
                lo, hi = minimummaximum.reduce(view, axis=axis)
                assert_array_equal(lo, np.minimum.reduce(view, axis=axis))
                assert_array_equal(hi, np.maximum.reduce(view, axis=axis))

    def test_reduce_out_tuple(self):
        a = np.arange(12.0).reshape(3, 4)
        o1 = np.empty(4)
        o2 = np.empty(4)
        minimummaximum.reduce(a, axis=0, out=(o1, o2))
        assert_array_equal(o1, np.minimum.reduce(a, axis=0))
        assert_array_equal(o2, np.maximum.reduce(a, axis=0))

    def test_reduce_keepdims(self):
        a = np.arange(24.0).reshape(2, 3, 4)
        lo, hi = minimummaximum.reduce(a, axis=1, keepdims=True)
        assert lo.shape == (2, 1, 4)
        assert_array_equal(lo, np.minimum.reduce(a, axis=1, keepdims=True))
        assert_array_equal(hi, np.maximum.reduce(a, axis=1, keepdims=True))

    def test_reduce_empty_requires_initial(self):
        with pytest.raises(ValueError):
            minimummaximum.reduce(np.array([], dtype=np.float64))

    def test_reduce_empty_with_initial(self):
        lo, hi = minimummaximum.reduce(
            np.array([], dtype=np.float64), initial=(np.inf, -np.inf))
        assert lo == np.inf and hi == -np.inf

    @pytest.mark.parametrize("dtype", ['bool', 'float16',
                                       'complex64', 'complex128'])
    def test_reduce_extra_dtypes(self, dtype):
        rng = np.random.default_rng(3)
        dt = np.dtype(dtype)
        if dt.kind == 'c':
            a = (rng.standard_normal(200)
                 + 1j * rng.standard_normal(200)).astype(dt)
        elif dt.kind == 'b':
            a = rng.integers(0, 2, 200).astype(dt)
        else:
            a = rng.standard_normal(200).astype(dt)
        lo, hi = minimummaximum.reduce(a)
        assert_equal(lo, np.minimum.reduce(a))
        assert_equal(hi, np.maximum.reduce(a))

    def test_reduce_datetime_nat(self):
        a = np.array(['2020-01-01', '2019-06-01', 'NaT', '2021-03-03'],
                     dtype='datetime64[D]')
        lo, hi = minimummaximum.reduce(a)
        assert_equal(lo, np.minimum.reduce(a))
        assert_equal(hi, np.maximum.reduce(a))
        # without NaT
        b = a[[0, 1, 3]]
        lo, hi = minimummaximum.reduce(b)
        assert_equal(lo, np.minimum.reduce(b))
        assert_equal(hi, np.maximum.reduce(b))

    def test_reduce_object(self):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=object)
        lo, hi = minimummaximum.reduce(a)
        assert lo == 1 and hi == 9

    def test_reduce_object_no_leak(self):
        # exercise the object refcounting path repeatedly; a leak sanitizer
        # would catch a missing decref here.
        objs = np.array([float(i) for i in range(50)], dtype=object)
        for _ in range(500):
            minimummaximum.reduce(objs)
            minimummaximum(objs, objs[::-1])


class TestMinmax:
    @pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 1), (1, 2)])
    def test_matches_min_max(self, axis):
        rng = np.random.default_rng(2)
        a = rng.standard_normal((7, 11, 13))
        lo, hi = np.minmax(a, axis=axis)
        assert_array_equal(lo, np.min(a, axis=axis))
        assert_array_equal(hi, np.max(a, axis=axis))

    def test_returns_plain_tuple(self):
        r = np.minmax(np.arange(5))
        assert type(r) is tuple
        assert len(r) == 2

    def test_scalar_result(self):
        lo, hi = np.minmax(np.array([3, 1, 4, 1, 5]))
        assert lo == 1 and hi == 5

    def test_out(self):
        a = np.arange(12.0).reshape(3, 4)
        o1 = np.empty(4)
        o2 = np.empty(4)
        r = np.minmax(a, axis=0, out=(o1, o2))
        assert r[0] is o1 and r[1] is o2
        assert_array_equal(o1, np.min(a, axis=0))
        assert_array_equal(o2, np.max(a, axis=0))

    def test_keepdims(self):
        a = np.arange(24.0).reshape(2, 3, 4)
        lo, hi = np.minmax(a, axis=1, keepdims=True)
        assert lo.shape == (2, 1, 4)
        assert_array_equal(lo, np.min(a, axis=1, keepdims=True))
        assert_array_equal(hi, np.max(a, axis=1, keepdims=True))

    def test_nan_propagation(self):
        a = np.array([1.0, np.nan, 3.0])
        lo, hi = np.minmax(a)
        assert np.isnan(lo) and np.isnan(hi)

    def test_where_and_initial(self):
        a = np.array([[-50], [10]])
        lo, hi = np.minmax(a, axis=-1, initial=(0, 0), where=True)
        assert_array_equal(lo, [-50, 0])
        assert_array_equal(hi, [0, 10])

    @pytest.mark.parametrize("dtype", ['bool', 'float16', 'complex128',
                                       'datetime64[D]', 'timedelta64[s]',
                                       'object'])
    def test_extra_dtypes_match_min_max(self, dtype):
        rng = np.random.default_rng(4)
        dt = np.dtype(dtype)
        if dt.kind == 'c':
            a = (rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(dt)
        elif dt.kind in 'mM':
            a = rng.integers(0, 10000, 30).astype('int64').astype(dt)
        elif dt.kind == 'b':
            a = rng.integers(0, 2, 30).astype(bool)
        elif dt.kind == 'O':
            a = np.array(list(rng.integers(0, 100, 30)), dtype=object)
        else:
            a = rng.standard_normal(30).astype(dt)
        lo, hi = np.minmax(a)
        assert lo == np.min(a)
        assert hi == np.max(a)
