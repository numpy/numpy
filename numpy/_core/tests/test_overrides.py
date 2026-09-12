import inspect
import os
import pickle
import sys
import tempfile
from io import StringIO
from unittest import mock

import pytest

import numpy as np
from numpy._core._multiarray_umath import _ArrayFunctionDispatcher
from numpy._core.overrides import (
    _get_implementing_args,
    array_function_dispatch,
    verify_matching_signatures,
)
from numpy.testing import assert_, assert_equal, assert_raises, assert_raises_regex
from numpy.testing.overrides import get_overridable_numpy_array_functions


def _return_not_implemented(self, *args, **kwargs):
    return NotImplemented


# need to define this at the top level to test pickling
@array_function_dispatch(lambda array: (array,))
def dispatched_one_arg(array):
    """Docstring."""
    return 'original'


@array_function_dispatch(lambda array1, array2: (array1, array2))
def dispatched_two_arg(array1, array2):
    """Docstring."""
    return 'original'


class TestGetImplementingArgs:

    def test_ndarray(self):
        array = np.array(1)

        args = _get_implementing_args([array])
        assert_equal(list(args), [array])

        args = _get_implementing_args([array, array])
        assert_equal(list(args), [array])

        args = _get_implementing_args([array, 1])
        assert_equal(list(args), [array])

        args = _get_implementing_args([1, array])
        assert_equal(list(args), [array])

    def test_ndarray_subclasses(self):

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        class NoOverrideSub(np.ndarray):
            pass

        array = np.array(1).view(np.ndarray)
        override_sub = np.array(1).view(OverrideSub)
        no_override_sub = np.array(1).view(NoOverrideSub)

        args = _get_implementing_args([array, override_sub])
        assert_equal(list(args), [override_sub, array])

        args = _get_implementing_args([array, no_override_sub])
        assert_equal(list(args), [no_override_sub, array])

        args = _get_implementing_args(
            [override_sub, no_override_sub])
        assert_equal(list(args), [override_sub, no_override_sub])

    def test_ndarray_and_duck_array(self):

        class Other:
            __array_function__ = _return_not_implemented

        array = np.array(1)
        other = Other()

        args = _get_implementing_args([other, array])
        assert_equal(list(args), [other, array])

        args = _get_implementing_args([array, other])
        assert_equal(list(args), [array, other])

    def test_ndarray_subclass_and_duck_array(self):

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        class Other:
            __array_function__ = _return_not_implemented

        array = np.array(1)
        subarray = np.array(1).view(OverrideSub)
        other = Other()

        assert_equal(_get_implementing_args([array, subarray, other]),
                     [subarray, array, other])
        assert_equal(_get_implementing_args([array, other, subarray]),
                     [subarray, array, other])

    def test_many_duck_arrays(self):

        class A:
            __array_function__ = _return_not_implemented

        class B(A):
            __array_function__ = _return_not_implemented

        class C(A):
            __array_function__ = _return_not_implemented

        class D:
            __array_function__ = _return_not_implemented

        a = A()
        b = B()
        c = C()
        d = D()

        assert_equal(_get_implementing_args([1]), [])
        assert_equal(_get_implementing_args([a]), [a])
        assert_equal(_get_implementing_args([a, 1]), [a])
        assert_equal(_get_implementing_args([a, a, a]), [a])
        assert_equal(_get_implementing_args([a, d, a]), [a, d])
        assert_equal(_get_implementing_args([a, b]), [b, a])
        assert_equal(_get_implementing_args([b, a]), [b, a])
        assert_equal(_get_implementing_args([a, b, c]), [b, c, a])
        assert_equal(_get_implementing_args([a, c, b]), [c, b, a])

    def test_too_many_duck_arrays(self):
        namespace = {'__array_function__': _return_not_implemented}
        types = [type('A' + str(i), (object,), namespace) for i in range(65)]
        relevant_args = [t() for t in types]

        actual = _get_implementing_args(relevant_args[:64])
        assert_equal(actual, relevant_args[:64])

        with assert_raises_regex(TypeError, 'distinct argument types'):
            _get_implementing_args(relevant_args)


class TestNDArrayArrayFunction:

    def test_method(self):

        class Other:
            __array_function__ = _return_not_implemented

        class NoOverrideSub(np.ndarray):
            pass

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        array = np.array([1])
        other = Other()
        no_override_sub = array.view(NoOverrideSub)
        override_sub = array.view(OverrideSub)

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray,),
                                          args=(array, 1.), kwargs={})
        assert_equal(result, 'original')

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, Other),
                                          args=(array, other), kwargs={})
        assert_(result is NotImplemented)

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, NoOverrideSub),
                                          args=(array, no_override_sub),
                                          kwargs={})
        assert_equal(result, 'original')

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, OverrideSub),
                                          args=(array, override_sub),
                                          kwargs={})
        assert_equal(result, 'original')

        with assert_raises_regex(TypeError, 'no implementation found'):
            np.concatenate((array, other))

        expected = np.concatenate((array, array))
        result = np.concatenate((array, no_override_sub))
        assert_equal(result, expected.view(NoOverrideSub))
        result = np.concatenate((array, override_sub))
        assert_equal(result, expected.view(OverrideSub))

    def test_no_wrapper(self):
        # Regular numpy functions have wrappers, but do not presume
        # all functions do (array creation ones do not): check that
        # we just call the function in that case.
        array = np.array(1)
        func = lambda x: x * 2
        result = array.__array_function__(func=func, types=(np.ndarray,),
                                          args=(array,), kwargs={})
        assert_equal(result, array * 2)

    def test_wrong_arguments(self):
        # Check our implementation guards against wrong arguments.
        a = np.array([1, 2])
        with pytest.raises(TypeError, match="args must be a tuple"):
            a.__array_function__(np.reshape, (np.ndarray,), a, (2, 1))
        with pytest.raises(TypeError, match="kwargs must be a dict"):
            a.__array_function__(np.reshape, (np.ndarray,), (a,), (2, 1))


class TestArrayFunctionDispatch:

    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            roundtripped = pickle.loads(
                    pickle.dumps(dispatched_one_arg, protocol=proto))
            assert_(roundtripped is dispatched_one_arg)

    def test_name_and_docstring(self):
        assert_equal(dispatched_one_arg.__name__, 'dispatched_one_arg')
        if sys.flags.optimize < 2:
            assert_equal(dispatched_one_arg.__doc__, 'Docstring.')

    def test_interface(self):

        class MyArray:
            def __array_function__(self, func, types, args, kwargs):
                return (self, func, types, args, kwargs)

        original = MyArray()
        (obj, func, types, args, kwargs) = dispatched_one_arg(original)
        assert_(obj is original)
        assert_(func is dispatched_one_arg)
        assert_equal(set(types), {MyArray})
        # assert_equal uses the overloaded np.iscomplexobj() internally
        assert_(args == (original,))
        assert_equal(kwargs, {})

    def test_not_implemented(self):

        class MyArray:
            def __array_function__(self, func, types, args, kwargs):
                return NotImplemented

        array = MyArray()
        with assert_raises_regex(TypeError, 'no implementation found'):
            dispatched_one_arg(array)

    def test_where_dispatch(self):

        class DuckArray:
            def __array_function__(self, ufunc, method, *inputs, **kwargs):
                return "overridden"

        array = np.array(1)
        duck_array = DuckArray()

        result = np.std(array, where=duck_array)

        assert_equal(result, "overridden")


class TestVerifyMatchingSignatures:

    def test_verify_matching_signatures(self):

        verify_matching_signatures(lambda x: 0, lambda x: 0)
        verify_matching_signatures(lambda x=None: 0, lambda x=None: 0)
        verify_matching_signatures(lambda x=1: 0, lambda x=None: 0)

        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda a: 0, lambda b: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x: 0, lambda x=None: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x=None: 0, lambda y=None: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x=1: 0, lambda y=1: 0)

    def test_array_function_dispatch(self):

        with assert_raises(RuntimeError):
            @array_function_dispatch(lambda x: (x,))
            def f(y):
                pass

        # should not raise
        @array_function_dispatch(lambda x: (x,), verify=False)
        def f(y):
            pass

    def test_reduction_configuration_errors(self):
        # reduction= requires a tuple-spec dispatcher
        with pytest.raises(TypeError, match="tuple-spec"):
            @array_function_dispatch(lambda a, out=None: (a, out),
                                     reduction=np.add)
            def _f(a, out=None):
                return a

        # a parameter with no ufunc.reduce slot declines the fast path
        # when passed (the Python implementation runs instead)
        @array_function_dispatch(("a", "out"), reduction=np.add)
        def _g(a, out=None, bogus=None):
            return "python-impl"

        assert _g(np.array([1, 2])) == 3
        assert _g(np.array([1, 2]), bogus=1) == "python-impl"

        # unknown reduce argument names in the defaults are rejected
        with pytest.raises(KeyError):
            @array_function_dispatch(
                    ("a", "out"), reduction=np.add,
                    reduction_defaults={"bogus": 1})
            def _h(a, out=None):
                return a

        # reduction_defaults without reduction= would be a silent no-op
        with pytest.raises(TypeError, match="reduction_defaults"):
            @array_function_dispatch(
                    ("a",), reduction_defaults={"dtype": bool})
            def _i(a):
                return a

        # forward= and reduction= are mutually exclusive
        with pytest.raises(TypeError, match="mutually exclusive"):
            @array_function_dispatch(
                    ("a",), reduction=np.add,
                    forward=(np.ndarray.ravel, ("a", "order")))
            def _j(a, axis=None):
                return a

        # forward_defaults without forward= would be a silent no-op
        with pytest.raises(TypeError, match="forward_defaults"):
            @array_function_dispatch(
                    ("a",), forward_defaults={"order": "C"})
            def _k(a):
                return a

        # a default that could bypass an override is rejected
        with pytest.raises(ValueError, match="bypass"):
            @array_function_dispatch(
                    ("a", "out"), reduction=np.add,
                    reduction_defaults={"where": [True]})
            def _l(a, out=None, where=np._NoValue):
                return a

    def test_reduction_explicit_novalue_argument(self):
        # explicit _NoValue for a parameter not defaulting to _NoValue
        # must be rejected, matching the Python wrapper (gh-31943)
        arr = np.array([1, 2, 3])
        with pytest.raises(TypeError):
            np.sum(arr, dtype=np._NoValue)
        with pytest.raises(TypeError):
            np.sum([1, 2, 3], dtype=np._NoValue)
        with pytest.raises(TypeError):
            np.sum(arr, axis=np._NoValue)
        # parameters whose default is _NoValue still treat it as "not passed"
        assert np.sum(arr, keepdims=np._NoValue) == 6
        assert np.sum(arr, initial=np._NoValue) == 6

    def test_reduce_slots_match_ufunc_reduce_signature(self):
        # the fast path scatters args into ufunc.reduce's positional
        # slots; pin that order down to catch drift
        from numpy._core.overrides import _REDUCE_DEFAULTS, _REDUCE_SLOT_NAMES

        assert set(_REDUCE_DEFAULTS) == set(_REDUCE_SLOT_NAMES) - {"a"}

        by_name = {"a": np.array([1, 2, 3]), "axis": 0, "dtype": np.int64,
                   "out": None, "keepdims": False, "initial": 1,
                   "where": True}
        assert set(by_name) == set(_REDUCE_SLOT_NAMES)
        # every slot name (with `a` -> `array`) must be a valid
        # ufunc.reduce keyword...
        expected = np.add.reduce(
            **{("array" if k == "a" else k): v for k, v in by_name.items()})
        # ...and the all-positional call in slot order must mean the same
        positional = [by_name[name] for name in _REDUCE_SLOT_NAMES]
        assert np.add.reduce(*positional) == expected == 7

    def test_dispatcher_constructor_argument_error(self):
        dispatcher = lambda x: (x,)

        with pytest.raises(TypeError, match=r"_ArrayFunctionDispatcher\(\)"):
            _ArrayFunctionDispatcher(dispatcher)

    def test_reduction_dispatches_and_falls_back(self):
        # exact-ndarray calls take the direct ufunc.reduce path; overrides
        # and invalid signatures fall back to normal dispatch
        @array_function_dispatch(("a", "out"), reduction=np.add)
        def my_sum(a, axis=None, dtype=None, out=None):
            return "python-impl"

        # exact ndarray: direct reduce call, wrapper never runs
        assert my_sum(np.array([1, 2, 3])) == 6
        assert my_sum(np.array([1, 2, 3]), axis=0) == 6

        # non-exact arrays fall back to the Python implementation
        assert my_sum([1, 2, 3]) == "python-impl"

        # overrides still dispatch
        class Duck:
            def __array_function__(self, func, types, args, kwargs):
                return "duck"
        assert my_sum(Duck()) == "duck"


class TestForwardDispatch:
    # forward=(target, slots): exact-ndarray calls go straight to the
    # target callable; everything else falls back to the Python wrapper.

    def test_forward_matches_wrapper(self):
        m = np.array([float(i) for i in range(6)]).reshape(2, 3)
        f = np.array([2.0, 0.5, 1.5])
        cases = [
            (np.transpose, (m,), {}),
            (np.ravel, (m,), {"order": "F"}),
            (np.take, (m, [0, 2]), {"axis": 1}),
            (np.argsort, (f,), {"kind": "stable"}),
            (np.round, (f,), {"decimals": 1}),
            (np.swapaxes, (m, 0, 1), {}),
            (np.argmax, (f,), {}),
            (np.searchsorted, (np.array([0.0, 1.0, 2.0]), 1.5), {}),
        ]
        for func, args, kwargs in cases:
            assert_equal(func(*args, **kwargs),
                         func._implementation(*args, **kwargs))

    def test_forward_declines_to_wrapper(self):
        recorded = []

        @array_function_dispatch(
            ("a",), forward=(np.ndarray.ravel, ("a", "order")))
        def my_ravel(a, order='C', upper=False):
            recorded.append("wrapper")
            return "wrapper"

        arr = np.array([[1, 2], [3, 4]])
        # exact ndarray: target called directly, wrapper never runs
        assert_equal(my_ravel(arr), arr.ravel())
        assert recorded == []
        # a parameter with no target slot declines when passed
        assert my_ravel(arr, upper=True) == "wrapper"

        # subclasses never take the fast path
        class Sub(np.ndarray):
            pass
        assert my_ravel(arr.view(Sub)) == "wrapper"

    def test_forward_missing_required_argument(self):
        # missing required arguments decline the fast path so the wrapper
        # raises the standard TypeError
        with pytest.raises(TypeError, match="missing"):
            np.swapaxes(np.array([[1, 2]]))

    def test_forward_default_for_array_slot_rejected(self):
        # a default for slot 0 (`a`) would be silently dead: the fast
        # path declines whenever the array argument is missing
        with pytest.raises(RuntimeError, match="no effect"):
            @array_function_dispatch(
                ("a",), forward=(np.ndarray.ravel, ("a", "order")),
                forward_defaults={"a": None})
            def _f(a, order='C'):
                return a

    def test_forward_default_for_required_parameter_rejected(self):
        # the fast path would succeed where the wrapper raises TypeError
        with pytest.raises(RuntimeError, match="required parameter"):
            @array_function_dispatch(
                ("a",), forward=(np.ndarray.repeat, ("a", "repeats")),
                forward_defaults={"repeats": 1})
            def _f(a, repeats):
                return a

    def test_inconsistent_signature_info_rejected(self):
        # direct constructor calls with n_pos_max/n_required out of range
        # must be rejected (they would index past the parameter table)
        for n_pos_max, n_required in [(2, 0), (1, 2), (0, -1)]:
            with pytest.raises(ValueError, match="inconsistent"):
                _ArrayFunctionDispatcher(
                    ((0,), (("a",), n_pos_max, n_required, False), None),
                    lambda a: a)

    def test_forward_gates_all_relevant_args(self):
        # Every dispatch-relevant arg must decline the fast path when it
        # carries an override -- searchsorted also dispatches on `v` and
        # `sorter`, not only on `a` (gh-31943 review).
        calls = []

        class Duck:
            def __array_function__(self, func, types, args, kwargs):
                calls.append(func)
                return "duck"

        a = np.array([0.0, 1.0, 2.0])
        assert np.searchsorted(a, Duck()) == "duck"
        assert calls == [np.searchsorted]

        calls.clear()
        assert np.searchsorted(a, 1.5, sorter=Duck()) == "duck"
        assert calls == [np.searchsorted]

    def test_forward_hides_filled_defaults_from_array_ufunc(self):
        # __array_ufunc__ via out/where must not see the filled defaults
        class Duck:
            kwargs = None

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                self.kwargs = sorted(kwargs)
                return NotImplemented

        a = np.arange(3.0)
        for make in (lambda d: {"where": d}, lambda d: {"out": (d,)}):
            fast, slow = Duck(), Duck()
            with pytest.raises(TypeError):
                np.sum(a, **make(fast))
            with pytest.raises(TypeError):
                np.sum._implementation(a, **make(slow))
            assert fast.kwargs == slow.kwargs

    def test_forward_checks_passed_values_only(self):
        @array_function_dispatch(("a", "out"), reduction=np.add)
        def my_sum(a, axis=None, dtype=None, out=None, where=np._NoValue):
            return "python-impl"

        arr = np.array([1, 2, 3])
        # defaults and explicit _NoValue take the fast path
        assert my_sum(arr) == 6
        assert my_sum(arr, where=np._NoValue) == 6
        assert my_sum(arr, where=True) == 6
        # passed values that could carry an override decline
        assert my_sum(arr, where=[True, False, True]) == "python-impl"
        assert my_sum(arr, out=(np.zeros(()),)) == "python-impl"
        assert my_sum(arr, None, None, (np.zeros(()),)) == "python-impl"

    def test_forward_keyword_slots(self):
        # keepdims exists only as a keyword slot on ndarray.argmax
        m = np.array([[1, 5], [3, 2]])
        assert_equal(np.argmax(m, axis=0, keepdims=True),
                     np.argmax._implementation(m, axis=0, keepdims=True))
        assert np.argmax(m) == 1
        # explicit np._NoValue means "not passed" (public default)
        assert np.argmax(np.array([3, 1]), keepdims=np._NoValue) == 0

    def test_forward_configuration_errors(self):
        with pytest.raises(TypeError, match="tuple-spec"):
            @array_function_dispatch(
                lambda a: (a,), forward=(np.ndarray.ravel, ("a", "order")))
            def _f(a, order=None):
                return a

        # keyword slots must be trailing
        with pytest.raises(RuntimeError, match="trailing"):
            @array_function_dispatch(
                ("a",), forward=(np.ndarray.ravel, ("a", "*x", "y")))
            def _g(a, x=None, y=None):
                return a

        # np._NoValue defaults require an explicit override
        with pytest.raises(RuntimeError, match="_NoValue"):
            @array_function_dispatch(
                ("a",), forward=(np.ndarray.ravel, ("a", "order")))
            def _h(a, order=np._NoValue):
                return a

        # slots without a public parameter need a default override
        with pytest.raises(RuntimeError, match="no public parameter"):
            @array_function_dispatch(
                ("a",), forward=(np.ndarray.take, ("a", "indices")))
            def _i(a):
                return a


class TestPositionalOnlyKeywordCalls:
    # Implementations with positional-only array parameters reject
    # keyword-passed arrays identically for plain ndarrays and for
    # __array_function__ implementors.  Before the tuple-spec conversion,
    # duck arrays dispatched on such calls while plain ndarrays raised.
    @pytest.mark.parametrize("func", [
        np.matrix_transpose,
        np.linalg.matrix_transpose,
        np.linalg.svdvals,
        # array-API mandated (x, /) signatures (gh-31943 review)
        np.unique_all,
        np.unique_counts,
        np.unique_inverse,
        np.unique_values,
    ])
    def test_keyword_call_raises_without_dispatch(self, func):
        calls = []

        class Duck:
            def __array_function__(self, func, types, args, kwargs):
                calls.append(func)
                return "duck"

        with pytest.raises(TypeError):
            func(x=np.ones((2, 2)))
        with pytest.raises(TypeError):
            func(x=Duck())
        assert calls == []

    def test_linalg_outer_keyword_call_raises(self):
        with pytest.raises(TypeError):
            np.linalg.outer(x1=np.ones(2), x2=np.ones(2))


def _new_duck_type_and_implements():
    """Create a duck array type and implements functions."""
    HANDLED_FUNCTIONS = {}

    class MyArray:
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


class TestArrayFunctionImplementation:

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

    def test_not_implemented(self):
        MyArray, implements = _new_duck_type_and_implements()

        @array_function_dispatch(lambda array: (array,), module='my')
        def func(array):
            return array

        array = np.array(1)
        assert_(func(array) is array)
        assert_equal(func.__module__, 'my')

        with assert_raises_regex(
                TypeError, "no implementation found for 'my.func'"):
            func(MyArray())

    @pytest.mark.parametrize("name", ["concatenate", "mean", "asarray"])
    def test_signature_error_message_simple(self, name):
        func = getattr(np, name)
        try:
            # all of these functions need an argument:
            func()
        except TypeError as e:
            exc = e

        assert exc.args[0].startswith(f"{name}()")

    def test_signature_error_message(self):
        # The lambda function will be named "<lambda>", but the TypeError
        # should show the name as "func"
        def _dispatcher():
            return ()

        @array_function_dispatch(_dispatcher)
        def func():
            pass

        try:
            func._implementation(bad_arg=3)
        except TypeError as e:
            expected_exception = e

        try:
            func(bad_arg=3)
            raise AssertionError("must fail")
        except TypeError as exc:
            if exc.args[0].startswith("_dispatcher"):
                # We replace the qualname currently, but it used `__name__`
                # (relevant functions have the same name and qualname anyway)
                pytest.skip("Python version is not using __qualname__ for "
                            "TypeError formatting.")

            assert exc.args == expected_exception.args

    @pytest.mark.parametrize(
        "func, args, kwargs",
        [
            (np.sum, (), {}),
            (np.sum, (np.ones(1),), {"out": object()}),
            (np.max, (np.ones(1),), {"dtype": np.float64}),
            (np.any, (np.ones(1),), {"dtype": np.bool_, "initial": False}),
            (np.sum, (np.ones(1), 0), {"axis": 0}),  # name and position
            (np.any, (np.ones(1), 0, None, False, True), {}),  # too many pos.
        ],
    )
    def test_reduction_error_message(self, func, args, kwargs):
        with pytest.raises(TypeError) as expected:
            func._implementation(*args, **kwargs)

        with pytest.raises(TypeError) as actual:
            func(*args, **kwargs)

        assert actual.value.args == expected.value.args

    @pytest.mark.parametrize(
        "args, name",
        [
            ((np.ones(1),), "bad"),  # unknown keyword
            ((np.ones(1), 0), "axis"),  # name and position
        ],
    )
    def test_reduction_signature_format_error(self, args, name):
        # Formatting the signature mismatch message calls `str()` on the
        # keyword, which can run arbitrary Python code.  If that raises, the
        # error must propagate instead of being cleared as a mismatch.
        class K(str):
            calls = 0

            def __str__(self):
                K.calls += 1
                if K.calls == 1:
                    raise RuntimeError
                return super().__str__()

        with pytest.raises(RuntimeError):
            np.sum(*args, **{K(name): 1})

        # The failure must happen while formatting, not somewhere later.
        assert K.calls == 1

    @pytest.mark.parametrize("value", [234, "this func is not replaced"])
    def test_dispatcher_error(self, value):
        # If the dispatcher raises an error, we must not attempt to mutate it
        error = TypeError(value)

        def dispatcher():
            raise error

        @array_function_dispatch(dispatcher)
        def func():
            return 3

        try:
            func()
            raise AssertionError("must fail")
        except TypeError as exc:
            assert exc is error  # unmodified exception

    def test_properties(self):
        # Check that str and repr are sensible
        func = dispatched_two_arg
        assert str(func) == str(func._implementation)
        repr_no_id = repr(func).split("at ")[0]
        repr_no_id_impl = repr(func._implementation).split("at ")[0]
        assert repr_no_id == repr_no_id_impl

    @pytest.mark.parametrize("func", [
            lambda x, y: 0,  # no like argument
            lambda like=None: 0,  # not keyword only
            lambda *, like=None, a=3: 0,  # not last (not that it matters)
        ])
    def test_bad_like_sig(self, func):
        # We sanity check the signature, and these should fail.
        with pytest.raises(RuntimeError):
            array_function_dispatch()(func)

    def test_bad_like_passing(self):
        # Cover internal sanity check for passing like as first positional arg
        def func(*, like=None):
            pass

        func_with_like = array_function_dispatch()(func)
        with pytest.raises(TypeError):
            func_with_like()
        with pytest.raises(TypeError):
            func_with_like(like=234)

    def test_too_many_args(self):
        # Mainly a unit-test to increase coverage
        objs = []
        for i in range(80):
            class MyArr:
                def __array_function__(self, *args, **kwargs):
                    return NotImplemented

            objs.append(MyArr())

        def _dispatch(*args):
            return args

        @array_function_dispatch(_dispatch)
        def func(*args):
            pass

        with pytest.raises(TypeError, match="maximum number"):
            func(*objs)


class TestNDArrayMethods:

    def test_repr(self):
        # gh-12162: should still be defined even if __array_function__ doesn't
        # implement np.array_repr()

        class MyArray(np.ndarray):
            def __array_function__(*args, **kwargs):
                return NotImplemented

        array = np.array(1).view(MyArray)
        assert_equal(repr(array), 'MyArray(1)')
        assert_equal(str(array), '1')


class TestNumPyFunctions:

    def test_set_module(self):
        assert_equal(np.sum.__module__, 'numpy')
        assert_equal(np.char.equal.__module__, 'numpy.char')
        assert_equal(np.fft.fft.__module__, 'numpy.fft')
        assert_equal(np.linalg.solve.__module__, 'numpy.linalg')

    def test_inspect_sum(self):
        signature = inspect.signature(np.sum)
        assert_('axis' in signature.parameters)
        assert_equal(signature, inspect.signature(np.sum._implementation))

    @pytest.mark.parametrize("func", [
        np.sum,            # tuple-spec dispatcher
        np.reshape,        # tuple-spec, positional-only relevant arg
        np.any,            # tuple-spec, keyword-only relevant arg
        pytest.param(
            np.concatenate,  # legacy callable dispatcher, C implementation
            marks=pytest.mark.skipif(
                sys.flags.optimize > 1,
                reason="the C implementation's signature comes from its "
                       "docstring, which -OO strips")),
        np.fft.fft,        # tuple-spec in a submodule
        np.linalg.solve,   # tuple-spec in a submodule
    ])
    def test_inspect_signature_matches_implementation(self, func):
        # Both dispatcher forms expose the implementation's signature via
        # __wrapped__ (set by functools.update_wrapper).
        assert_equal(inspect.signature(func),
                     inspect.signature(func._implementation))

    def test_override_sum(self):
        MyArray, implements = _new_duck_type_and_implements()

        @implements(np.sum)
        def _(array):
            return 'yes'

        assert_equal(np.sum(MyArray()), 'yes')

    def test_sum_on_mock_array(self):

        # We need a proxy for mocks because __array_function__ is only looked
        # up in the class dict
        class ArrayProxy:
            def __init__(self, value):
                self.value = value

            def __array_function__(self, *args, **kwargs):
                return self.value.__array_function__(*args, **kwargs)

            def __array__(self, *args, **kwargs):
                return self.value.__array__(*args, **kwargs)

        proxy = ArrayProxy(mock.Mock(spec=ArrayProxy))
        proxy.value.__array_function__.return_value = 1
        result = np.sum(proxy)
        assert_equal(result, 1)
        proxy.value.__array_function__.assert_called_once_with(
            np.sum, (ArrayProxy,), (proxy,), {})
        proxy.value.__array__.assert_not_called()

    def test_sum_forwarding_implementation(self):

        class MyArray(np.ndarray):

            def sum(self, axis, out):
                return 'summed'

            def __array_function__(self, func, types, args, kwargs):
                return super().__array_function__(func, types, args, kwargs)

        # note: the internal implementation of np.sum() calls the .sum() method
        array = np.array(1).view(MyArray)
        assert_equal(np.sum(array), 'summed')


class TestArrayLike:
    def _create_MyArray(self):
        class MyArray:
            def __init__(self, function=None):
                self.function = function

            def __array_function__(self, func, types, args, kwargs):
                assert func is getattr(np, func.__name__)
                try:
                    my_func = getattr(self, func.__name__)
                except AttributeError:
                    return NotImplemented
                return my_func(*args, **kwargs)

        return MyArray

    def _create_MyNoArrayFunctionArray(self):
        class MyNoArrayFunctionArray:
            def __init__(self, function=None):
                self.function = function

        return MyNoArrayFunctionArray

    def _create_MySubclass(self):
        class MySubclass(np.ndarray):
            def __array_function__(self, func, types, args, kwargs):
                result = super().__array_function__(func, types, args, kwargs)
                return result.view(self.__class__)

        return MySubclass

    def add_method(self, name, arr_class, enable_value_error=False):
        def _definition(*args, **kwargs):
            # Check that `like=` isn't propagated downstream
            assert 'like' not in kwargs

            if enable_value_error and 'value_error' in kwargs:
                raise ValueError

            return arr_class(getattr(arr_class, name))
        setattr(arr_class, name, _definition)

    def func_args(*args, **kwargs):
        return args, kwargs

    def test_array_like_not_implemented(self):
        MyArray = self._create_MyArray()
        self.add_method('array', MyArray)

        ref = MyArray.array()

        with assert_raises_regex(TypeError, 'no implementation found'):
            array_like = np.asarray(1, like=ref)

    _array_tests = [
        ('array', *func_args((1,))),
        ('asarray', *func_args((1,))),
        ('asanyarray', *func_args((1,))),
        ('ascontiguousarray', *func_args((2, 3))),
        ('asfortranarray', *func_args((2, 3))),
        ('require', *func_args((np.arange(6).reshape(2, 3),),
                               requirements=['A', 'F'])),
        ('empty', *func_args((1,))),
        ('full', *func_args((1,), 2)),
        ('ones', *func_args((1,))),
        ('zeros', *func_args((1,))),
        ('arange', *func_args(3)),
        ('frombuffer', *func_args(b'\x00' * 8, dtype=int)),
        ('fromiter', *func_args(range(3), dtype=int)),
        ('fromstring', *func_args('1,2', dtype=int, sep=',')),
        ('loadtxt', *func_args(lambda: StringIO('0 1\n2 3'))),
        ('genfromtxt', *func_args(lambda: StringIO('1,2.1'),
                                  dtype=[('int', 'i8'), ('float', 'f8')],
                                  delimiter=',')),
    ]

    def test_nep35_functions_as_array_functions(self,):
        all_array_functions = get_overridable_numpy_array_functions()
        like_array_functions_subset = {
            getattr(np, func_name) for func_name, *_ in self.__class__._array_tests
        }
        assert like_array_functions_subset.issubset(all_array_functions)

        nep35_python_functions = {
            np.eye, np.fromfunction, np.full, np.genfromtxt,
            np.identity, np.loadtxt, np.ones, np.require, np.tri,
        }
        assert nep35_python_functions.issubset(all_array_functions)

        nep35_C_functions = {
            np.arange, np.array, np.asanyarray, np.asarray,
            np.ascontiguousarray, np.asfortranarray, np.empty,
            np.frombuffer, np.fromfile, np.fromiter, np.fromstring,
            np.zeros,
        }
        assert nep35_C_functions.issubset(all_array_functions)

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    @pytest.mark.parametrize('numpy_ref', [True, False])
    def test_array_like(self, function, args, kwargs, numpy_ref):
        MyArray = self._create_MyArray()
        self.add_method('array', MyArray)
        self.add_method(function, MyArray)
        np_func = getattr(np, function)
        my_func = getattr(MyArray, function)

        if numpy_ref is True:
            ref = np.array(1)
        else:
            ref = MyArray.array()

        like_args = tuple(a() if callable(a) else a for a in args)
        array_like = np_func(*like_args, **kwargs, like=ref)

        if numpy_ref is True:
            assert type(array_like) is np.ndarray

            np_args = tuple(a() if callable(a) else a for a in args)
            np_arr = np_func(*np_args, **kwargs)

            # Special-case np.empty to ensure values match
            if function == "empty":
                np_arr.fill(1)
                array_like.fill(1)

            assert_equal(array_like, np_arr)
        else:
            assert type(array_like) is MyArray
            assert array_like.function is my_func

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    @pytest.mark.parametrize('ref', [1, [1], "MyNoArrayFunctionArray"])
    def test_no_array_function_like(self, function, args, kwargs, ref):
        MyNoArrayFunctionArray = self._create_MyNoArrayFunctionArray()
        self.add_method('array', MyNoArrayFunctionArray)
        self.add_method(function, MyNoArrayFunctionArray)
        np_func = getattr(np, function)

        # Instantiate ref if it's the MyNoArrayFunctionArray class
        if ref == "MyNoArrayFunctionArray":
            ref = MyNoArrayFunctionArray.array()

        like_args = tuple(a() if callable(a) else a for a in args)

        with assert_raises_regex(TypeError,
                'The `like` argument must be an array-like that implements'):
            np_func(*like_args, **kwargs, like=ref)

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    def test_subclass(self, function, args, kwargs):
        MySubclass = self._create_MySubclass()
        ref = np.array(1).view(MySubclass)
        np_func = getattr(np, function)
        like_args = tuple(a() if callable(a) else a for a in args)
        array_like = np_func(*like_args, **kwargs, like=ref)
        assert type(array_like) is MySubclass
        if np_func is np.empty:
            return
        np_args = tuple(a() if callable(a) else a for a in args)
        np_arr = np_func(*np_args, **kwargs)
        assert_equal(array_like.view(np.ndarray), np_arr)

    @pytest.mark.parametrize('numpy_ref', [True, False])
    def test_array_like_fromfile(self, numpy_ref):
        MyArray = self._create_MyArray()
        self.add_method('array', MyArray)
        self.add_method("fromfile", MyArray)

        if numpy_ref is True:
            ref = np.array(1)
        else:
            ref = MyArray.array()

        data = np.random.random(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "testfile")
            data.tofile(fname)

            array_like = np.fromfile(fname, like=ref)
            if numpy_ref is True:
                assert type(array_like) is np.ndarray
                np_res = np.fromfile(fname, like=ref)
                assert_equal(np_res, data)
                assert_equal(array_like, np_res)
            else:
                assert type(array_like) is MyArray
                assert array_like.function is MyArray.fromfile

    def test_exception_handling(self):
        MyArray = self._create_MyArray()
        self.add_method('array', MyArray, enable_value_error=True)

        ref = MyArray.array()

        with assert_raises(TypeError):
            # Raises the error about `value_error` being invalid first
            np.array(1, value_error=True, like=ref)

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    def test_like_as_none(self, function, args, kwargs):
        MyArray = self._create_MyArray()
        self.add_method('array', MyArray)
        self.add_method(function, MyArray)
        np_func = getattr(np, function)

        like_args = tuple(a() if callable(a) else a for a in args)
        # required for loadtxt and genfromtxt to init w/o error.
        like_args_exp = tuple(a() if callable(a) else a for a in args)

        array_like = np_func(*like_args, **kwargs, like=None)
        expected = np_func(*like_args_exp, **kwargs)
        # Special-case np.empty to ensure values match
        if function == "empty":
            array_like.fill(1)
            expected.fill(1)
        assert_equal(array_like, expected)


def test_function_like():
    # We provide a `__get__` implementation, make sure it works
    assert type(np.mean) is np._core._multiarray_umath._ArrayFunctionDispatcher

    class MyClass:
        def __array__(self, dtype=None, copy=None):
            # valid argument to mean:
            return np.arange(3)

        func1 = staticmethod(np.mean)
        func2 = np.mean
        func3 = classmethod(np.mean)

    m = MyClass()
    assert m.func1([10]) == 10
    assert m.func2() == 1  # mean of the arange
    with pytest.raises(TypeError, match="unsupported operand type"):
        # Tries to operate on the class
        m.func3()

    # Manual binding also works (the above may shortcut):
    bound = np.mean.__get__(m, MyClass)
    assert bound() == 1

    bound = np.mean.__get__(None, MyClass)  # unbound actually
    assert bound([10]) == 10

    bound = np.mean.__get__(MyClass)  # classmethod
    with pytest.raises(TypeError, match="unsupported operand type"):
        bound()


def _make_duck(result, calls=None):
    """A minimal class implementing __array_function__."""
    class Duck:
        def __array_function__(self, func, types, args, kwargs):
            if calls is not None:
                calls.append(func)
            return result
    return Duck


# The same dispatch scenarios must behave identically for tuple-spec and
# legacy callable dispatchers; tests using `dispatched_op` run through
# both C code paths.
@pytest.fixture(params=["tuple-spec", "callable"])
def dispatched_op(request):
    """An (a, out=None) function dispatched via each dispatcher form."""
    if request.param == "tuple-spec":
        decorator = array_function_dispatch(("a", "out"))
    else:
        decorator = array_function_dispatch(lambda a, out=None: (a, out))

    @decorator
    def op(a, out=None):
        return "original"
    return op


class TestTupleSpecDispatch:
    """Tuple-spec dispatcher: spec validation and construction."""

    def test_construction_failure_does_not_crash(self):
        from numpy._core._multiarray_umath import _ArrayFunctionDispatcher
        with pytest.raises(TypeError):
            _ArrayFunctionDispatcher(42, 42, 42)
        with pytest.raises(TypeError):
            _ArrayFunctionDispatcher(42)
        with pytest.raises(TypeError):
            _ArrayFunctionDispatcher()

    def test_empty_tuple_spec_rejected(self):
        with pytest.raises(ValueError, match="at least one relevant"):
            @array_function_dispatch(())
            def _f(a):
                return a

    def test_docs_from_dispatcher_incompatible_with_tuple_spec(self):
        with pytest.raises(TypeError, match="docs_from_dispatcher"):
            @array_function_dispatch(("a",), docs_from_dispatcher=True)
            def _f(a):
                return a

    def test_var_positional_rejected(self):
        # *args makes tuple-spec positions ambiguous at call time.
        with pytest.raises(RuntimeError, match=r"\*args"):
            @array_function_dispatch(("a", "out"))
            def _f(a, *args, out=None):
                return a

    def test_required_keyword_only_rejected(self):
        # validate_call_signature does not track required keyword-only
        # parameters, so they cannot be allowed at decoration time.
        with pytest.raises(RuntimeError, match="keyword-only"):
            @array_function_dispatch(("a", "out"))
            def _f(a, *, out):
                return a

    def test_unknown_arg_name_rejected(self):
        with pytest.raises(RuntimeError, match="not found"):
            @array_function_dispatch(("a", "missing"))
            def _f(a, out=None):
                return a

    def test_non_string_spec_rejected(self):
        with pytest.raises(TypeError, match="only strings"):
            @array_function_dispatch(("a", 42))
            def _f(a, out=None):
                return a

    def test_namedtuple_not_treated_as_tuple_spec(self):
        # Match C-side PyTuple_CheckExact; tuple subclasses take callable path.
        from collections import namedtuple
        NT = namedtuple("NT", ["a", "b"])
        spec = NT("a", "out")

        with pytest.raises(TypeError):
            @array_function_dispatch(spec)
            def _f(a, out=None):
                return a

    def test_wrapped_impl_signature_followed(self):
        # inspect.signature follows __wrapped__; reading __code__ would fail.
        import functools

        def _wrap(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        @array_function_dispatch(("a", "out"))
        @_wrap
        def _f(a, out=None):
            return a

        assert _f(np.arange(3)) is not None

    def test_name_and_docstring_preserved(self):
        @array_function_dispatch(("a",))
        def my_func(a):
            """My docstring."""
            return a

        assert my_func.__name__ == "my_func"
        if sys.flags.optimize < 2:
            assert my_func.__doc__ == "My docstring."

    def test_kwonly_relevant_arg_excess_positional(self):
        # A kwonly relevant arg (where) must never be read positionally:
        # an invalid call with excess positional args raises TypeError
        # instead of dispatching on the excess arg.
        duck = _make_duck("duck")()
        with pytest.raises(TypeError, match="positional"):
            np.any(np.arange(3), 0, None, True, duck)
        # valid keyword use still dispatches
        assert np.any(np.arange(3), where=duck) == "duck"

    def test_posonly_relevant_arg_by_keyword(self):
        # A positional-only relevant arg (a in reshape) must never be
        # matched by keyword: the invalid call raises TypeError instead
        # of dispatching.
        duck = _make_duck("duck")()
        with pytest.raises(TypeError, match="positional-only"):
            np.reshape(a=duck, shape=(2,))
        # valid positional use still dispatches
        assert np.reshape(duck, (2,)) == "duck"

    def test_invalid_signature_with_override(self):
        # A signature-invalid call must raise TypeError even when an
        # __array_function__ override is present, instead of forwarding
        # the invalid call to the override unchecked.
        duck = _make_duck("duck")()
        with pytest.raises(TypeError, match="unexpected keyword"):
            np.sum(duck, bogus=3)
        with pytest.raises(TypeError, match="multiple values"):
            np.sum(duck, a="duplicate")
        with pytest.raises(TypeError, match="positional arguments"):
            np.sum(duck, 0, None, None, False, 0, True, "extra")
        # the valid call still dispatches
        assert np.sum(duck) == "duck"


class TestDispatchFormEquivalence:
    """The same scenarios through both tuple-spec and callable dispatchers."""

    def test_all_safe_args_call_implementation(self, dispatched_op):
        assert dispatched_op(np.arange(3)) == "original"
        assert dispatched_op(np.arange(3), out=None) == "original"

    def test_dispatches_ndarray_subclass(self, dispatched_op):
        calls = []

        class Sub(np.ndarray):
            def __array_function__(self, func, types, args, kwargs):
                calls.append(func)
                return "subclass-handled"

        assert dispatched_op(np.arange(3).view(Sub)) == "subclass-handled"
        assert calls == [dispatched_op]

    def test_dispatches_duck_array(self, dispatched_op):
        calls = []
        duck = _make_duck("duck-handled", calls)()
        assert dispatched_op(duck) == "duck-handled"
        assert calls == [dispatched_op]

    def test_out_keyword_dispatch(self, dispatched_op):
        calls = []
        duck = _make_duck("duck-out", calls)()
        assert dispatched_op(np.arange(3), out=duck) == "duck-out"
        assert calls == [dispatched_op]

    def test_not_implemented_no_fallback(self, dispatched_op):
        duck = _make_duck(NotImplemented)()
        with pytest.raises(TypeError, match="no implementation found"):
            dispatched_op(duck)

    def test_mixed_ndarray_and_duck(self, dispatched_op):
        # ndarray fallback must stay in the chain when a subclass's
        # override returns NotImplemented.
        class Sub(np.ndarray):
            def __array_function__(self, func, types, args, kwargs):
                return NotImplemented

        out = np.arange(3).view(Sub)
        assert dispatched_op(np.arange(3), out=out) == "original"
