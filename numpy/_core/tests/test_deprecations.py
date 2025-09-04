"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.

"""
import contextlib
import warnings

import pytest

import numpy as np
import numpy._core._struct_ufunc_tests as struct_ufunc
from numpy._core._multiarray_tests import fromstring_null_term_c_api  # noqa: F401
from numpy.testing import assert_raises, temppath


class _DeprecationTestCase:
    # Just as warning: warnings uses re.match, so the start of this message
    # must match.
    message = ''
    warning_cls = DeprecationWarning

    def _setup(self):
        warn_ctx = warnings.catch_warnings(record=True)
        log = warn_ctx.__enter__()

        # Do *not* ignore other DeprecationWarnings. Ignoring warnings
        # can give very confusing results because of
        # https://bugs.python.org/issue4180 and it is probably simplest to
        # try to keep the tests cleanly giving only the right warning type.
        # (While checking them set to "error" those are ignored anyway)
        # We still have them show up, because otherwise they would be raised
        warnings.filterwarnings("always", category=self.warning_cls)
        warnings.filterwarnings("always", message=self.message,
                                category=self.warning_cls)
        return log, warn_ctx

    def _teardown(self, warn_ctx):
        warn_ctx.__exit__()

    def assert_deprecated(self, log, function, num=1, ignore_others=False,
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

        # reset the log
        log[:] = []

        if exceptions is np._NoValue:
            exceptions = (self.warning_cls,)

        if function_fails:
            context_manager = contextlib.suppress(Exception)
        else:
            context_manager = contextlib.nullcontext()
        with context_manager:
            function(*args, **kwargs)

        # just in case, clear the registry
        num_found = 0
        for warning in log:
            if warning.category is self.warning_cls:
                num_found += 1
            elif not ignore_others:
                raise AssertionError(
                        "expected %s but got: %s" %
                        (self.warning_cls.__name__, warning.category))
        if num is not None and num_found != num:
            msg = f"{len(log)} warnings found but {num} expected."
            lst = [str(w) for w in log]
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

    def assert_not_deprecated(self, log, function, args=(), kwargs={}):
        """Test that warnings are not raised.

        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
        self.assert_deprecated(log, function, num=0, ignore_others=True,
                        exceptions=(), args=args, kwargs=kwargs)


class _VisibleDeprecationTestCase(_DeprecationTestCase):
    warning_cls = np.exceptions.VisibleDeprecationWarning


class TestTestDeprecated:
    def test_assert_deprecated(self):
        test_case_instance = _DeprecationTestCase()
        log, warn_ctx = test_case_instance._setup()
        assert_raises(AssertionError,
                      test_case_instance.assert_deprecated,
                      log,
                      lambda: None)

        def foo():
            warnings.warn("foo", category=DeprecationWarning, stacklevel=2)

        test_case_instance.assert_deprecated(log, foo)
        test_case_instance._teardown(warn_ctx)


class TestBincount(_DeprecationTestCase):
    # 2024-07-29, 2.1.0
    @pytest.mark.parametrize('badlist', [[0.5, 1.2, 1.5],
                                         ['0', '1', '1']])
    def test_bincount_bad_list(self, badlist):
        log, warn_ctx = self._setup()
        self.assert_deprecated(log, lambda: np.bincount(badlist))
        self._teardown(warn_ctx)


class TestGeneratorSum(_DeprecationTestCase):
    # 2018-02-25, 1.15.0
    def test_generator_sum(self):
        log, warn_ctx = self._setup()
        self.assert_deprecated(log, np.sum, args=((i for i in range(5)),))
        self._teardown(warn_ctx)


class BuiltInRoundComplexDType(_DeprecationTestCase):
    # 2020-03-31 1.19.0
    deprecated_types = [np.csingle, np.cdouble, np.clongdouble]
    not_deprecated_types = [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
    ]

    def test_deprecated(self):
        log, warn_ctx = self._setup()
        for scalar_type in self.deprecated_types:
            scalar = scalar_type(0)
            self.assert_deprecated(log, round, args=(scalar,))
            self.assert_deprecated(log, round, args=(scalar, 0))
            self.assert_deprecated(log, round, args=(scalar,), kwargs={'ndigits': 0})
        self._teardown(warn_ctx)

    def test_not_deprecated(self):
        log, warn_ctx = self._setup()
        for scalar_type in self.not_deprecated_types:
            scalar = scalar_type(0)
            self.assert_not_deprecated(log, round, args=(scalar,))
            self.assert_not_deprecated(log, round, args=(scalar, 0))
            self.assert_not_deprecated(log, round, args=(scalar,),
                                       kwargs={'ndigits': 0})
        self._teardown(warn_ctx)


class FlatteningConcatenateUnsafeCast(_DeprecationTestCase):
    # NumPy 1.20, 2020-09-03
    message = "concatenate with `axis=None` will use same-kind casting"

    def test_deprecated(self):
        log, warn_ctx = self._setup()
        self.assert_deprecated(log, np.concatenate,
                args=(([0.], [1.]),),
                kwargs={'axis': None, 'out': np.empty(2, dtype=np.int64)})
        self._teardown(warn_ctx)

    def test_not_deprecated(self):
        log, warn_ctx = self._setup()
        self.assert_not_deprecated(log, np.concatenate,
                args=(([0.], [1.]),),
                kwargs={'axis': None, 'out': np.empty(2, dtype=np.int64),
                        'casting': "unsafe"})

        with assert_raises(TypeError):
            # Tests should notice if the deprecation warning is given first...
            np.concatenate(([0.], [1.]), out=np.empty(2, dtype=np.int64),
                           casting="same_kind")

        self._teardown(warn_ctx)


class TestCtypesGetter(_DeprecationTestCase):
    # Deprecated 2021-05-18, Numpy 1.21.0
    warning_cls = DeprecationWarning
    ctypes = np.array([1]).ctypes

    @pytest.mark.parametrize(
        "name", ["get_data", "get_shape", "get_strides", "get_as_parameter"]
    )
    def test_deprecated(self, name: str) -> None:
        log, warn_ctx = self._setup()
        func = getattr(self.ctypes, name)
        self.assert_deprecated(log, func)
        self._teardown(warn_ctx)

    @pytest.mark.parametrize(
        "name", ["data", "shape", "strides", "_as_parameter_"]
    )
    def test_not_deprecated(self, name: str) -> None:
        log, warn_ctx = self._setup()
        self.assert_not_deprecated(log, lambda: getattr(self.ctypes, name))
        self._teardown(warn_ctx)


class TestMachAr(_DeprecationTestCase):
    # Deprecated 2022-11-22, NumPy 1.25
    warning_cls = DeprecationWarning

    def test_deprecated_module(self):
        log, warn_ctx = self._setup()
        self.assert_deprecated(log, lambda: np._core.MachAr)
        self._teardown(warn_ctx)


class TestQuantileInterpolationDeprecation(_DeprecationTestCase):
    # Deprecated 2021-11-08, NumPy 1.22
    @pytest.mark.parametrize("func",
        [np.percentile, np.quantile, np.nanpercentile, np.nanquantile])
    def test_deprecated(self, func):
        log, warn_ctx = self._setup()
        self.assert_deprecated(log,
            lambda: func([0., 1.], 0., interpolation="linear"))
        self.assert_deprecated(log,
            lambda: func([0., 1.], 0., interpolation="nearest"))
        self._teardown(warn_ctx)

    @pytest.mark.parametrize("func",
            [np.percentile, np.quantile, np.nanpercentile, np.nanquantile])
    def test_both_passed(self, func):
        log, warn_ctx = self._setup()

        with warnings.catch_warnings():
            # catch the DeprecationWarning so that it does not raise:
            warnings.simplefilter("always", DeprecationWarning)
            with pytest.raises(TypeError):
                func([0., 1.], 0., interpolation="nearest", method="nearest")

        self._teardown(warn_ctx)


class TestScalarConversion(_DeprecationTestCase):
    # 2023-01-02, 1.25.0
    def test_float_conversion(self):
        log, warn_ctx = self._setup()
        self.assert_deprecated(log, float, args=(np.array([3.14]),))
        self._teardown(warn_ctx)

    def test_behaviour(self):
        b = np.array([[3.14]])
        c = np.zeros(5)
        with pytest.warns(DeprecationWarning):
            c[0] = b


class TestPyIntConversion(_DeprecationTestCase):
    message = r".*stop allowing conversion of out-of-bound.*"

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_deprecated_scalar(self, dtype):
        log, warn_ctx = self._setup()
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
                self.assert_deprecated(log,
                        lambda: creation_func(info.min - 1, dtype))
            except OverflowError:
                pass  # OverflowErrors always happened also before and are OK.

            try:
                self.assert_deprecated(log,
                        lambda: creation_func(info.max + 1, dtype))
            except OverflowError:
                pass  # OverflowErrors always happened also before and are OK.

        self._teardown(warn_ctx)

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


class TestDeprecatedFinfo(_DeprecationTestCase):
    # Deprecated in NumPy 1.25, 2023-01-16
    def test_deprecated_none(self):
        log, warn_ctx = self._setup()
        self.assert_deprecated(log, np.finfo, args=(None,))
        self._teardown(warn_ctx)


class TestMathAlias(_DeprecationTestCase):
    def test_deprecated_np_lib_math(self):
        log, warn_ctx = self._setup()
        self.assert_deprecated(log, lambda: np.lib.math)
        self._teardown(warn_ctx)


class TestLibImports(_DeprecationTestCase):
    # Deprecated in Numpy 1.26.0, 2023-09
    def test_lib_functions_deprecation_call(self):
        from numpy import in1d, row_stack, trapz
        from numpy._core.numerictypes import maximum_sctype
        from numpy.lib._function_base_impl import disp
        from numpy.lib._npyio_impl import recfromcsv, recfromtxt
        from numpy.lib._shape_base_impl import get_array_wrap
        from numpy.lib._utils_impl import safe_eval
        from numpy.lib.tests.test_io import TextIO

        log, warn_ctx = self._setup()
        self.assert_deprecated(log, lambda: safe_eval("None"))

        data_gen = lambda: TextIO('A,B\n0,1\n2,3')
        kwargs = {'delimiter': ",", 'missing_values': "N/A", 'names': True}
        self.assert_deprecated(log, lambda: recfromcsv(data_gen()))
        self.assert_deprecated(log, lambda: recfromtxt(data_gen(), **kwargs))

        self.assert_deprecated(log, lambda: disp("test"))
        self.assert_deprecated(log, get_array_wrap)
        self.assert_deprecated(log, lambda: maximum_sctype(int))

        self.assert_deprecated(log, lambda: in1d([1], [1]))
        self.assert_deprecated(log, lambda: row_stack([[]]))
        self.assert_deprecated(log, lambda: trapz([1], [1]))
        self.assert_deprecated(log, lambda: np.chararray)
        self._teardown(warn_ctx)


class TestDeprecatedDTypeAliases(_DeprecationTestCase):

    def _check_for_warning(self, func):
        log, warn_ctx = self._setup()
        with warnings.catch_warnings(record=True) as caught_warnings:
            func()
        assert len(caught_warnings) == 1
        w = caught_warnings[0]
        assert w.category is DeprecationWarning
        assert "alias 'a' was deprecated in NumPy 2.0" in str(w.message)
        self._teardown(warn_ctx)

    def test_a_dtype_alias(self):
        log, warn_ctx = self._setup()
        for dtype in ["a", "a10"]:
            f = lambda: np.dtype(dtype)
            self._check_for_warning(f)
            self.assert_deprecated(log, f)
            f = lambda: np.array(["hello", "world"]).astype("a10")
            self._check_for_warning(f)
            self.assert_deprecated(log, f)
        self._teardown(warn_ctx)


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

        log, warn_ctx = self._setup()
        test1 = Test1()
        test2 = Test2()
        self.assert_deprecated(log, lambda: np.negative(test1))
        assert test1.called
        self.assert_deprecated(log, lambda: np.negative(test2))
        assert test2.called
        self._teardown(warn_ctx)

class TestDeprecatedArrayAttributeSetting(_DeprecationTestCase):
    message = "Setting the .*on a NumPy array has been deprecated.*"

    def test_deprecated_strides_set(self):
        log, warn_ctx = self._setup()
        x = np.eye(2)
        self.assert_deprecated(log, setattr, args=(x, 'strides', x.strides))
        self._teardown(warn_ctx)


class TestDeprecatedDTypeParenthesizedRepeatCount(_DeprecationTestCase):
    message = "Passing in a parenthesized single number"

    @pytest.mark.parametrize("string", ["(2)i,", "(3)3S,", "f,(2)f"])
    def test_parenthesized_repeat_count(self, string):
        log, warn_ctx = self._setup()
        self.assert_deprecated(log, np.dtype, args=(string,))
        self._teardown(warn_ctx)


class TestDeprecatedSaveFixImports(_DeprecationTestCase):
    # Deprecated in Numpy 2.1, 2024-05
    message = "The 'fix_imports' flag is deprecated and has no effect."

    def test_deprecated(self):
        log, warn_ctx = self._setup()
        with temppath(suffix='.npy') as path:
            sample_args = (path, np.array(np.zeros((1024, 10))))
            self.assert_not_deprecated(log, np.save, args=sample_args)
            self.assert_deprecated(log, np.save, args=sample_args,
                                kwargs={'fix_imports': True})
            self.assert_deprecated(log, np.save, args=sample_args,
                                kwargs={'fix_imports': False})
            for allow_pickle in [True, False]:
                self.assert_not_deprecated(log, np.save, args=sample_args,
                                        kwargs={'allow_pickle': allow_pickle})
                self.assert_deprecated(log, np.save, args=sample_args,
                                    kwargs={'allow_pickle': allow_pickle,
                                            'fix_imports': True})
                self.assert_deprecated(log, np.save, args=sample_args,
                                    kwargs={'allow_pickle': allow_pickle,
                                            'fix_imports': False})
        self._teardown(warn_ctx)


class TestAddNewdocUFunc(_DeprecationTestCase):
    # Deprecated in Numpy 2.2, 2024-11
    def test_deprecated(self):
        log, warn_ctx = self._setup()
        doc = struct_ufunc.add_triplet.__doc__
        # gh-26718
        # This test mutates the C-level docstring pointer for add_triplet,
        # which is permanent once set. Skip when re-running tests.
        if doc is not None and "new docs" in doc:
            pytest.skip("Cannot retest deprecation, otherwise ValueError: "
                "Cannot change docstring of ufunc with non-NULL docstring")
        self.assert_deprecated(log,
            lambda: np._core.umath._add_newdoc_ufunc(
                struct_ufunc.add_triplet, "new docs"
            )
        )
        self._teardown(warn_ctx)


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
        log, warn_ctx = self._setup()
        self.assert_deprecated(log, lambda: np.dtype("f8", align=3))
        self._teardown(warn_ctx)

    @pytest.mark.parametrize("align", [True, False, np.True_, np.False_])
    def test_not_deprecated(self, align):
        # if the user passes a bool, it is accepted.
        log, warn_ctx = self._setup()
        self.assert_not_deprecated(log, lambda: np.dtype("f8", align=align))
        self._teardown(warn_ctx)


class TestFlatiterIndexing0dBoolIndex(_DeprecationTestCase):
    # Deprecated in Numpy 2.4, 2025-07
    message = r"Indexing flat iterators with a 0-dimensional boolean index"

    def test_0d_boolean_index_deprecated(self):
        log, warn_ctx = self._setup()
        arr = np.arange(3)
        # 0d boolean indices on flat iterators are deprecated
        self.assert_deprecated(log, lambda: arr.flat[True])
        self._teardown(warn_ctx)

    def test_0d_boolean_assign_index_deprecated(self):
        log, warn_ctx = self._setup()
        arr = np.arange(3)

        def assign_to_index():
            arr.flat[True] = 10

        self.assert_deprecated(log, assign_to_index)
        self._teardown(warn_ctx)


class TestFlatiterIndexingFloatIndex(_DeprecationTestCase):
    # Deprecated in NumPy 2.4, 2025-07
    message = r"Invalid non-array indices for iterator objects"

    def test_float_index_deprecated(self):
        log, warn_ctx = self._setup()
        arr = np.arange(3)
        # float indices on flat iterators are deprecated
        self.assert_deprecated(log, lambda: arr.flat[[1.]])
        self._teardown(warn_ctx)

    def test_float_assign_index_deprecated(self):
        log, warn_ctx = self._setup()
        arr = np.arange(3)

        def assign_to_index():
            arr.flat[[1.]] = 10

        self.assert_deprecated(log, assign_to_index)
        self._teardown(warn_ctx)


class TestWarningUtilityDeprecations(_DeprecationTestCase):
    # Deprecation in NumPy 2.4, 2025-08
    message = r"NumPy warning suppression and assertion utilities are deprecated."

    def test_assert_warns_deprecated(self):
        def use_assert_warns():
            with np.testing.assert_warns(RuntimeWarning):
                warnings.warn("foo", RuntimeWarning, stacklevel=1)

        log, warn_ctx = self._setup()
        self.assert_deprecated(log, use_assert_warns)
        self._teardown(warn_ctx)

    def test_suppress_warnings_deprecated(self):
        def use_suppress_warnings():
            with np.testing.suppress_warnings() as sup:
                sup.filter(RuntimeWarning, 'invalid value encountered in divide')

        log, warn_ctx = self._setup()
        self.assert_deprecated(log, use_suppress_warnings)
        self._teardown(warn_ctx)


class TestTooManyArgsExtremum(_DeprecationTestCase):
    # Deprecated in Numpy 2.4, 2025-08, gh-27639
    message = "Passing more than 2 positional arguments to np.maximum and np.minimum "

    @pytest.mark.parametrize("ufunc", [np.minimum, np.maximum])
    def test_extremem_3_args(self, ufunc):
        log, warn_ctx = self._setup()
        self.assert_deprecated(log, ufunc, args=(np.ones(1), np.zeros(1), np.empty(1)))
        self._teardown(warn_ctx)
