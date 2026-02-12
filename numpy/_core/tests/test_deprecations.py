"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.

"""
import contextlib
import warnings

import pytest

import numpy as np
from numpy._core._multiarray_tests import fromstring_null_term_c_api  # noqa: F401
from numpy.testing import assert_raises


class _DeprecationTestCase:
    # Just as warning: warnings uses re.match, so the start of this message
    # must match.
    message = ''
    warning_cls = DeprecationWarning

    @contextlib.contextmanager
    def filter_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            # Do *not* ignore other DeprecationWarnings. Ignoring warnings
            # can give very confusing results because of
            # https://bugs.python.org/issue4180 and it is probably simplest to
            # try to keep the tests cleanly giving only the right warning type.
            # (While checking them set to "error" those are ignored anyway)
            # We still have them show up, because otherwise they would be raised
            warnings.filterwarnings("always", category=self.warning_cls)
            warnings.filterwarnings("always", message=self.message,
                                    category=self.warning_cls)
            yield w
        return

    def assert_deprecated(self, function, num=1, ignore_others=False,
                          function_fails=False,
                          exceptions=np._NoValue,
                          args=(), kwargs={}):
        """Test if DeprecationWarnings are given and raised.

        This first checks if the function when called gives `num`
        DeprecationWarnings, after that it tries to raise these
        DeprecationWarnings and compares them with `exceptions`.
        The exceptions can be different for cases where this code path
        is simply not anticipated and the exception is replaced.

        Parameters
        ----------
        function : callable
            The function to test
        num : int
            Number of DeprecationWarnings to expect. This should normally be 1.
        ignore_others : bool
            Whether warnings of the wrong type should be ignored (note that
            the message is not checked)
        function_fails : bool
            If the function would normally fail, setting this will check for
            warnings inside a try/except block.
        exceptions : Exception or tuple of Exceptions
            Exception to expect when turning the warnings into an error.
            The default checks for DeprecationWarnings. If exceptions is
            empty the function is expected to run successfully.
        args : tuple
            Arguments for `function`
        kwargs : dict
            Keyword arguments for `function`
        """
        __tracebackhide__ = True  # Hide traceback for py.test

        if exceptions is np._NoValue:
            exceptions = (self.warning_cls,)

        if function_fails:
            context_manager = contextlib.suppress(Exception)
        else:
            context_manager = contextlib.nullcontext()
        with context_manager:
            with self.filter_warnings() as w_context:
                function(*args, **kwargs)

        # just in case, clear the registry
        num_found = 0
        for warning in w_context:
            if warning.category is self.warning_cls:
                num_found += 1
            elif not ignore_others:
                raise AssertionError(
                        "expected %s but got: %s" %
                        (self.warning_cls.__name__, warning.category))
        if num is not None and num_found != num:
            msg = f"{len(w_context)} warnings found but {num} expected."
            lst = [str(w) for w in w_context]
            raise AssertionError("\n".join([msg] + lst))

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=self.message,
                                    category=self.warning_cls)
            try:
                function(*args, **kwargs)
                if exceptions != ():
                    raise AssertionError(
                            "No error raised during function call")
            except exceptions:
                if exceptions == ():
                    raise AssertionError(
                            "Error raised during function call")

    def assert_not_deprecated(self, function, args=(), kwargs={}):
        """Test that warnings are not raised.

        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=(), args=args, kwargs=kwargs)


class _VisibleDeprecationTestCase(_DeprecationTestCase):
    warning_cls = np.exceptions.VisibleDeprecationWarning


class TestTestDeprecated:
    def test_assert_deprecated(self):
        test_case_instance = _DeprecationTestCase()
        assert_raises(AssertionError,
                      test_case_instance.assert_deprecated,
                      lambda: None)

        def foo():
            warnings.warn("foo", category=DeprecationWarning, stacklevel=2)

        test_case_instance.assert_deprecated(foo)


class TestCtypesGetter(_DeprecationTestCase):
    ctypes = np.array([1]).ctypes

    @pytest.mark.parametrize("name", ["data", "shape", "strides", "_as_parameter_"])
    def test_not_deprecated(self, name: str) -> None:
        self.assert_not_deprecated(lambda: getattr(self.ctypes, name))


class TestPyIntConversion(_DeprecationTestCase):
    message = r".*stop allowing conversion of out-of-bound.*"

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_deprecated_scalar(self, dtype):
        dtype = np.dtype(dtype)
        info = np.iinfo(dtype)

        # Cover the most common creation paths (all end up in the
        # same place):
        def scalar(value, dtype):
            dtype.type(value)

        def assign(value, dtype):
            arr = np.array([0, 0, 0], dtype=dtype)
            arr[2] = value

        def create(value, dtype):
            np.array([value], dtype=dtype)

        for creation_func in [scalar, assign, create]:
            try:
                self.assert_deprecated(
                        lambda: creation_func(info.min - 1, dtype))
            except OverflowError:
                pass  # OverflowErrors always happened also before and are OK.

            try:
                self.assert_deprecated(
                        lambda: creation_func(info.max + 1, dtype))
            except OverflowError:
                pass  # OverflowErrors always happened also before and are OK.


@pytest.mark.parametrize("name", ["str", "bytes", "object"])
def test_future_scalar_attributes(name):
    # FutureWarning added 2022-11-17, NumPy 1.24,
    assert name not in dir(np)  # we may want to not add them
    with pytest.warns(FutureWarning,
            match=f"In the future .*{name}"):
        assert not hasattr(np, name)

    # Unfortunately, they are currently still valid via `np.dtype()`
    np.dtype(name)
    name in np._core.sctypeDict


# Ignore the above future attribute warning for this test.
@pytest.mark.filterwarnings("ignore:In the future:FutureWarning")
class TestRemovedGlobals:
    # Removed 2023-01-12, NumPy 1.24.0
    # Not a deprecation, but the large error was added to aid those who missed
    # the previous deprecation, and should be removed similarly to one
    # (or faster).
    @pytest.mark.parametrize("name",
            ["object", "float", "complex", "str", "int"])
    def test_attributeerror_includes_info(self, name):
        msg = f".*\n`np.{name}` was a deprecated alias for the builtin"
        with pytest.raises(AttributeError, match=msg):
            getattr(np, name)


class TestCharArray(_DeprecationTestCase):
    def test_deprecated_chararray(self):
        self.assert_deprecated(lambda: np.char.chararray)


class TestDeprecatedDTypeAliases(_DeprecationTestCase):
    @pytest.mark.parametrize("dtype_code", ["a", "a10"])
    def test_a_dtype_alias(self, dtype_code: str):
        # Deprecated in 2.0, removed in 2.5, 2025-12
        with pytest.raises(TypeError):
            np.dtype(dtype_code)


class TestDeprecatedArrayWrap(_DeprecationTestCase):
    message = "__array_wrap__.*"

    def test_deprecated(self):
        class Test1:
            def __array__(self, dtype=None, copy=None):
                return np.arange(4)

            def __array_wrap__(self, arr, context=None):
                self.called = True
                return 'pass context'

        class Test2(Test1):
            def __array_wrap__(self, arr):
                self.called = True
                return 'pass'

        test1 = Test1()
        test2 = Test2()
        self.assert_deprecated(lambda: np.negative(test1))
        assert test1.called
        self.assert_deprecated(lambda: np.negative(test2))
        assert test2.called

class TestDeprecatedArrayAttributeSetting(_DeprecationTestCase):
    message = "Setting the .*on a NumPy array has been deprecated.*"

    def test_deprecated_strides_set(self):
        x = np.eye(2)
        self.assert_deprecated(setattr, args=(x, 'strides', x.strides))

    def test_deprecated_shape_set(self):
        x = np.eye(2)
        self.assert_deprecated(setattr, args=(x, "shape", (4, 1)))

class TestDeprecatedDTypeParenthesizedRepeatCount(_DeprecationTestCase):
    message = "Passing in a parenthesized single number"

    @pytest.mark.parametrize("string", ["(2)i,", "(3)3S,", "f,(2)f"])
    def test_parenthesized_repeat_count(self, string):
        self.assert_deprecated(np.dtype, args=(string,))


class TestDTypeAlignBool(_VisibleDeprecationTestCase):
    # Deprecated in Numpy 2.4, 2025-07
    # NOTE: As you can see, finalizing this deprecation breaks some (very) old
    # pickle files.  This may be fine, but needs to be done with some care since
    # it breaks all of them and not just some.
    # (Maybe it should be a 3.0 or only after warning more explicitly around pickles.)
    message = r"dtype\(\): align should be passed as Python or NumPy boolean but got "

    def test_deprecated(self):
        # in particular integers should be rejected because one may think they mean
        # alignment, or pass them accidentally as a subarray shape (meaning to pass
        # a tuple).
        self.assert_deprecated(lambda: np.dtype("f8", align=3))

    @pytest.mark.parametrize("align", [True, False, np.True_, np.False_])
    def test_not_deprecated(self, align):
        # if the user passes a bool, it is accepted.
        self.assert_not_deprecated(lambda: np.dtype("f8", align=align))


class TestFlatiterIndexing0dBoolIndex(_DeprecationTestCase):
    # Deprecated in Numpy 2.4, 2025-07
    message = r"Indexing flat iterators with a 0-dimensional boolean index"

    def test_0d_boolean_index_deprecated(self):
        arr = np.arange(3)
        # 0d boolean indices on flat iterators are deprecated
        self.assert_deprecated(lambda: arr.flat[True])

    def test_0d_boolean_assign_index_deprecated(self):
        arr = np.arange(3)

        def assign_to_index():
            arr.flat[True] = 10

        self.assert_deprecated(assign_to_index)


class TestFlatiterIndexingFloatIndex(_DeprecationTestCase):
    # Deprecated in NumPy 2.4, 2025-07
    message = r"Invalid non-array indices for iterator objects"

    def test_float_index_deprecated(self):
        arr = np.arange(3)
        # float indices on flat iterators are deprecated
        self.assert_deprecated(lambda: arr.flat[[1.]])

    def test_float_assign_index_deprecated(self):
        arr = np.arange(3)

        def assign_to_index():
            arr.flat[[1.]] = 10

        self.assert_deprecated(assign_to_index)


@pytest.mark.thread_unsafe(
    reason="warning control utilities are deprecated due to being thread-unsafe"
)
class TestWarningUtilityDeprecations(_DeprecationTestCase):
    # Deprecation in NumPy 2.4, 2025-08
    message = r"NumPy warning suppression and assertion utilities are deprecated."

    def test_assert_warns_deprecated(self):
        def use_assert_warns():
            with np.testing.assert_warns(RuntimeWarning):
                warnings.warn("foo", RuntimeWarning, stacklevel=1)

        self.assert_deprecated(use_assert_warns)

    def test_suppress_warnings_deprecated(self):
        def use_suppress_warnings():
            with np.testing.suppress_warnings() as sup:
                sup.filter(RuntimeWarning, 'invalid value encountered in divide')

        self.assert_deprecated(use_suppress_warnings)


class TestTooManyArgsExtremum(_DeprecationTestCase):
    # Deprecated in Numpy 2.4, 2025-08, gh-27639
    message = "Passing more than 2 positional arguments to np.maximum and np.minimum "

    @pytest.mark.parametrize("ufunc", [np.minimum, np.maximum])
    def test_extremem_3_args(self, ufunc):
        self.assert_deprecated(ufunc, args=(np.ones(1), np.zeros(1), np.empty(1)))


class TestTypenameDeprecation(_DeprecationTestCase):
    # Deprecation in Numpy 2.5, 2026-02

    def test_typename_emits_deprecation_warning(self):
        self.assert_deprecated(lambda: np.typename("S1"))
        self.assert_deprecated(lambda: np.typename("h"))
