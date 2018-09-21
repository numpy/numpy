from __future__ import division, absolute_import, print_function

import pickle
import sys

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_raises_regex)
from numpy.core.overrides import (
    get_overloaded_types_and_args, array_function_dispatch)


def _get_overloaded_args(relevant_args):
    types, args = get_overloaded_types_and_args(relevant_args)
    return args


def _return_self(self, *args, **kwargs):
    return self


class TestGetOverloadedTypesAndArgs(object):

    def test_ndarray(self):
        array = np.array(1)

        types, args = get_overloaded_types_and_args([array])
        assert_equal(set(types), {np.ndarray})
        assert_equal(list(args), [])

        types, args = get_overloaded_types_and_args([array, array])
        assert_equal(len(types), 1)
        assert_equal(set(types), {np.ndarray})
        assert_equal(list(args), [])

        types, args = get_overloaded_types_and_args([array, 1])
        assert_equal(set(types), {np.ndarray})
        assert_equal(list(args), [])

        types, args = get_overloaded_types_and_args([1, array])
        assert_equal(set(types), {np.ndarray})
        assert_equal(list(args), [])

    def test_ndarray_subclasses(self):

        class OverrideSub(np.ndarray):
            __array_function__ = _return_self

        class NoOverrideSub(np.ndarray):
            pass

        array = np.array(1).view(np.ndarray)
        override_sub = np.array(1).view(OverrideSub)
        no_override_sub = np.array(1).view(NoOverrideSub)

        types, args = get_overloaded_types_and_args([array, override_sub])
        assert_equal(set(types), {np.ndarray, OverrideSub})
        assert_equal(list(args), [override_sub])

        types, args = get_overloaded_types_and_args([array, no_override_sub])
        assert_equal(set(types), {np.ndarray, NoOverrideSub})
        assert_equal(list(args), [])

        types, args = get_overloaded_types_and_args(
            [override_sub, no_override_sub])
        assert_equal(set(types), {OverrideSub, NoOverrideSub})
        assert_equal(list(args), [override_sub])

    def test_ndarray_and_duck_array(self):

        class Other(object):
            __array_function__ = _return_self

        array = np.array(1)
        other = Other()

        types, args = get_overloaded_types_and_args([other, array])
        assert_equal(set(types), {np.ndarray, Other})
        assert_equal(list(args), [other])

        types, args = get_overloaded_types_and_args([array, other])
        assert_equal(set(types), {np.ndarray, Other})
        assert_equal(list(args), [other])

    def test_many_duck_arrays(self):

        class A(object):
            __array_function__ = _return_self

        class B(A):
            __array_function__ = _return_self

        class C(A):
            __array_function__ = _return_self

        class D(object):
            __array_function__ = _return_self

        a = A()
        b = B()
        c = C()
        d = D()

        assert_equal(_get_overloaded_args([1]), [])
        assert_equal(_get_overloaded_args([a]), [a])
        assert_equal(_get_overloaded_args([a, 1]), [a])
        assert_equal(_get_overloaded_args([a, a, a]), [a])
        assert_equal(_get_overloaded_args([a, d, a]), [a, d])
        assert_equal(_get_overloaded_args([a, b]), [b, a])
        assert_equal(_get_overloaded_args([b, a]), [b, a])
        assert_equal(_get_overloaded_args([a, b, c]), [b, c, a])
        assert_equal(_get_overloaded_args([a, c, b]), [c, b, a])


class TestNDArrayArrayFunction(object):

    def test_method(self):

        class SubOverride(np.ndarray):
            __array_function__ = _return_self

        class NoOverrideSub(np.ndarray):
            pass

        array = np.array(1)

        def func():
            return 'original'

        result = array.__array_function__(
            func=func, types=(np.ndarray,), args=(), kwargs={})
        assert_equal(result, 'original')

        result = array.__array_function__(
            func=func, types=(np.ndarray, SubOverride), args=(), kwargs={})
        assert_(result is NotImplemented)

        result = array.__array_function__(
            func=func, types=(np.ndarray, NoOverrideSub), args=(), kwargs={})
        assert_equal(result, 'original')


# need to define this at the top level to test pickling
@array_function_dispatch(lambda array: (array,))
def dispatched_one_arg(array):
    """Docstring."""
    return 'original'


class TestArrayFunctionDispatch(object):

    def test_pickle(self):
        roundtripped = pickle.loads(pickle.dumps(dispatched_one_arg))
        assert_(roundtripped is dispatched_one_arg)

    def test_name_and_docstring(self):
        assert_equal(dispatched_one_arg.__name__, 'dispatched_one_arg')
        if sys.flags.optimize < 2:
            assert_equal(dispatched_one_arg.__doc__, 'Docstring.')

    def test_interface(self):

        class MyArray(object):
            def __array_function__(self, func, types, args, kwargs):
                return (self, func, types, args, kwargs)

        original = MyArray()
        (obj, func, types, args, kwargs) = dispatched_one_arg(original)
        assert_(obj is original)
        assert_(func is dispatched_one_arg)
        assert_equal(set(types), {MyArray})
        assert_equal(args, (original,))
        assert_equal(kwargs, {})


def _new_duck_type_and_implements():
    """Create a duck array type and implements functions."""
    HANDLED_FUNCTIONS = {}

    class MyArray(object):
        def __array_function__(self, func, types, args, kwargs):
            if func not in HANDLED_FUNCTIONS:
                return NotImplemented
            if not all(issubclass(t, MyArray) for t in types):
                return NotImplemented
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def implements(numpy_function):
        """Register an __array_function__ implementations."""
        def decorator(func):
            HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator

    return (MyArray, implements)


class TestArrayFunctionImplementation(object):

    def test_one_arg(self):
        MyArray, implements = _new_duck_type_and_implements()

        @implements(dispatched_one_arg)
        def _(array):
            return 'myarray'

        assert_equal(dispatched_one_arg(1), 'original')
        assert_equal(dispatched_one_arg(MyArray()), 'myarray')

    def test_optional_args(self):
        MyArray, implements = _new_duck_type_and_implements()

        @array_function_dispatch(lambda array, option=None: (array,))
        def func_with_option(array, option='default'):
            return option

        @implements(func_with_option)
        def my_array_func_with_option(array, new_option='myarray'):
            return new_option

        # we don't need to implement every option on __array_function__
        # implementations
        assert_equal(func_with_option(1), 'default')
        assert_equal(func_with_option(1, option='extra'), 'extra')
        assert_equal(func_with_option(MyArray()), 'myarray')
        with assert_raises(TypeError):
            func_with_option(MyArray(), option='extra')

        # but new options on implementations can't be used
        result = my_array_func_with_option(MyArray(), new_option='yes')
        assert_equal(result, 'yes')
        with assert_raises(TypeError):
            func_with_option(MyArray(), new_option='no')

    def test_unimplemented(self):
        MyArray, implements = _new_duck_type_and_implements()

        @array_function_dispatch(lambda array: (array,))
        def func(array):
            return array

        array = np.array(1)
        assert_(func(array) is array)

        with assert_raises_regex(TypeError, 'no implementation found'):
            func(MyArray())
