import pytest

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing.overrides import allows_array_function_override


class Container:
    """container that implements basic __array__ protocol"""

    def __init__(self, a):
        self._a = np.asarray(a)

    def __array__(self):
        return self._a

    def __array_wrap__(self, arr, context=None, return_scalar=False):
        return self.__class__(arr)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._a})"


class NonCallable(Container):
    """container with attributes that clash with numpy methods"""

    put = mean = std = var = None
    sum = max = min = all = any = None
    squeeze = None


class Dummy:
    """class with no __array__ interface"""

    max = min = None
    all = any = None


class TestNonCallable:
    """check that non callable attributes are not called by numpy functions"""

    @pytest.fixture
    def arr1D(self):
        l = [1.0 / 7.0, 2.0 / 3.0, 3.0]
        return NonCallable(l), np.array(l)

    @pytest.fixture
    def arr2D(self):
        l = [[1.0 / 7.0, 2.0 / 3.0, 3.0]]
        return NonCallable(l), np.array(l)

    @pytest.fixture
    def singleton(self):
        return Dummy()

    @pytest.mark.parametrize(
        "fun",
        [np.mean, np.std, np.var, np.sum, np.max, np.min, np.all, np.any],
    )
    def test_reductions(self, arr1D, fun):
        # check _wrap_reduction and not wrapped functions 'mean', 'std', 'var'
        assert allows_array_function_override(fun)
        c, a = arr1D
        res = fun(c, axis=0)
        desired = fun(a, axis=0)
        assert np.ndim(desired) == 0
        assert res == desired

    def test_put(self, arr1D):
        c, _ = arr1D
        with pytest.raises(
            TypeError,
            match=r"argument 1 \(NonCallable\) does not have a 'put' method",
        ):
            np.put(c, [0, 2], [-44, -55])

    def test_squeeze(self, arr2D):
        c, a = arr2D
        res = np.squeeze(c)
        desired = np.squeeze(a)
        assert isinstance(res, NonCallable)
        assert isinstance(desired, np.ndarray)
        assert np.shape(res) == np.shape(desired)
        assert_array_equal(res, desired)

    @pytest.mark.parametrize("fun", [np.max, np.min])
    def test_wrapped_singleton(self, singleton, fun):
        res = fun(singleton)
        assert res is singleton

    @pytest.mark.parametrize("fun", [np.all, np.any])
    def test_wrapped_singleton_any_all(self, singleton, fun):
        res = fun(singleton)
        assert res is np.True_
