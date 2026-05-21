import pytest

import numpy as np
from numpy import fix, isneginf, isposinf
from numpy.testing import assert_, assert_array_equal, assert_equal, assert_raises


class TestUfunclike:

    def test_isposinf(self):
        a = np.array([np.inf, -np.inf, np.nan, 0.0, 3.0, -3.0])
        out = np.zeros(a.shape, bool)
        tgt = np.array([True, False, False, False, False, False])

        res = isposinf(a)
        assert_equal(res, tgt)
        res = isposinf(a, out)
        assert_equal(res, tgt)
        assert_equal(out, tgt)

        a = a.astype(np.complex128)
        with assert_raises(TypeError):
            isposinf(a)

    def test_isneginf(self):
        a = np.array([np.inf, -np.inf, np.nan, 0.0, 3.0, -3.0])
        out = np.zeros(a.shape, bool)
        tgt = np.array([False, True, False, False, False, False])

        res = isneginf(a)
        assert_equal(res, tgt)
        res = isneginf(a, out)
        assert_equal(res, tgt)
        assert_equal(out, tgt)

        a = a.astype(np.complex128)
        with assert_raises(TypeError):
            isneginf(a)

    @pytest.mark.filterwarnings("ignore:numpy.fix is deprecated:DeprecationWarning")
    def test_fix(self):
        a = np.array([[1.0, 1.1, 1.5, 1.8], [-1.0, -1.1, -1.5, -1.8]])
        out = np.zeros(a.shape, float)
        tgt = np.array([[1., 1., 1., 1.], [-1., -1., -1., -1.]])

        res = fix(a)
        assert_equal(res, tgt)
        res = fix(a, out)
        assert_equal(res, tgt)
        assert_equal(out, tgt)
        assert_equal(fix(3.14), 3)

    @pytest.mark.filterwarnings("ignore:numpy.fix is deprecated:DeprecationWarning")
    def test_fix_with_subclass(self):
        class MyArray(np.ndarray):
            def __new__(cls, data, metadata=None):
                res = np.array(data, copy=True).view(cls)
                res.metadata = metadata
                return res

            def __array_wrap__(self, obj, context=None, return_scalar=False):
                if not isinstance(obj, MyArray):
                    obj = obj.view(MyArray)
                if obj.metadata is None:
                    obj.metadata = self.metadata
                return obj

            def __array_finalize__(self, obj):
                self.metadata = getattr(obj, 'metadata', None)
                return self

        a = np.array([1.1, -1.1])
        m = MyArray(a, metadata='foo')
        f = fix(m)
        assert_array_equal(f, np.array([1, -1]))
        assert_(isinstance(f, MyArray))
        assert_equal(f.metadata, 'foo')

        # check 0d arrays don't decay to scalars
        m0d = m[0, ...]
        m0d.metadata = 'bar'
        f0d = fix(m0d)
        assert_(isinstance(f0d, MyArray))
        assert_equal(f0d.metadata, 'bar')

    @pytest.mark.filterwarnings("ignore:numpy.fix is deprecated:DeprecationWarning")
    def test_scalar(self):
        x = np.inf
        actual = np.isposinf(x)
        expected = np.True_
        assert_equal(actual, expected)
        assert_equal(type(actual), type(expected))

        x = -3.4
        actual = np.fix(x)
        expected = np.float64(-3.0)
        assert_equal(actual, expected)
        assert_equal(type(actual), type(expected))

        out = np.array(0.0)
        actual = np.fix(x, out=out)
        assert_(actual is out)


class TestFixDeprecation:
    """Test that numpy.fix emits a DeprecationWarning."""

    def test_fix_emits_deprecation_warning(self):
        a = np.array([1.5, 2.7, -1.5, -2.7])
        with pytest.warns(DeprecationWarning, match="numpy.fix is deprecated"):
            fix(a)

    def test_fix_scalar_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="numpy.fix is deprecated"):
            fix(3.14)

    def test_fix_with_out_emits_deprecation_warning(self):
        a = np.array([1.5, 2.7])
        out = np.zeros(a.shape)
        with pytest.warns(DeprecationWarning, match="numpy.fix is deprecated"):
            fix(a, out=out)
