"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.

"""
from __future__ import division, absolute_import, print_function

import sys
import operator
import warnings

import numpy as np
from numpy.testing import (run_module_suite, assert_raises,
                           assert_warns, assert_array_equal, assert_)


class _DeprecationTestCase(object):
    # Just as warning: warnings uses re.match, so the start of this message
    # must match.
    message = ''

    def setUp(self):
        self.warn_ctx = warnings.catch_warnings(record=True)
        self.log = self.warn_ctx.__enter__()

        # Do *not* ignore other DeprecationWarnings. Ignoring warnings
        # can give very confusing results because of
        # http://bugs.python.org/issue4180 and it is probably simplest to
        # try to keep the tests cleanly giving only the right warning type.
        # (While checking them set to "error" those are ignored anyway)
        # We still have them show up, because otherwise they would be raised
        warnings.filterwarnings("always", category=DeprecationWarning)
        warnings.filterwarnings("always", message=self.message,
                                    category=DeprecationWarning)

    def tearDown(self):
        self.warn_ctx.__exit__()

    def assert_deprecated(self, function, num=1, ignore_others=False,
                        function_fails=False,
                        exceptions=(DeprecationWarning,), args=(), kwargs={}):
        """Test if DeprecationWarnings are given and raised.

        This first checks if the function when called gives `num`
        DeprecationWarnings, after that it tries to raise these
        DeprecationWarnings and compares them with `exceptions`.
        The exceptions can be different for cases where this code path
        is simply not anticipated and the exception is replaced.

        Parameters
        ----------
        f : callable
            The function to test
        num : int
            Number of DeprecationWarnings to expect. This should normally be 1.
        ignore_other : bool
            Whether warnings of the wrong type should be ignored (note that
            the message is not checked)
        function_fails : bool
            If the function would normally fail, setting this will check for
            warnings inside a try/except block.
        exceptions : Exception or tuple of Exceptions
            Exception to expect when turning the warnings into an error.
            The default checks for DeprecationWarnings. If exceptions is
            empty the function is expected to run successfull.
        args : tuple
            Arguments for `f`
        kwargs : dict
            Keyword arguments for `f`
        """
        # reset the log
        self.log[:] = []

        try:
            function(*args, **kwargs)
        except (Exception if function_fails else tuple()):
            pass

        # just in case, clear the registry
        num_found = 0
        for warning in self.log:
            if warning.category is DeprecationWarning:
                num_found += 1
            elif not ignore_others:
                raise AssertionError(
                        "expected DeprecationWarning but got: %s" %
                        (warning.category,))
        if num is not None and num_found != num:
            msg = "%i warnings found but %i expected." % (len(self.log), num)
            lst = [w.category for w in self.log]
            raise AssertionError("\n".join([msg] + [lst]))

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=self.message,
                                    category=DeprecationWarning)
            try:
                function(*args, **kwargs)
                if exceptions != tuple():
                    raise AssertionError(
                            "No error raised during function call")
            except exceptions:
                if exceptions == tuple():
                    raise AssertionError(
                            "Error raised during function call")

    def assert_not_deprecated(self, function, args=(), kwargs={}):
        """Test if DeprecationWarnings are given and raised.

        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)


class TestBooleanUnaryMinusDeprecation(_DeprecationTestCase):
    """Test deprecation of unary boolean `-`. While + and * are well
    defined, unary - is not and even a corrected form seems to have
    no real uses.

    The deprecation process was started in NumPy 1.9.
    """
    message = r"numpy boolean negative, the `-` operator, .*"

    def test_unary_minus_operator_deprecation(self):
        array = np.array([True])
        generic = np.bool_(True)

        # Unary minus/negative ufunc:
        self.assert_deprecated(operator.neg, args=(array,))
        self.assert_deprecated(operator.neg, args=(generic,))


class TestBooleanBinaryMinusDeprecation(_DeprecationTestCase):
    """Test deprecation of binary boolean `-`. While + and * are well
    defined, binary  - is not and even a corrected form seems to have
    no real uses.

    The deprecation process was started in NumPy 1.9.
    """
    message = r"numpy boolean subtract, the `-` operator, .*"

    def test_operator_deprecation(self):
        array = np.array([True])
        generic = np.bool_(True)

        # Minus operator/subtract ufunc:
        self.assert_deprecated(operator.sub, args=(array, array))
        self.assert_deprecated(operator.sub, args=(generic, generic))


class TestRankDeprecation(_DeprecationTestCase):
    """Test that np.rank is deprecated. The function should simply be
    removed. The VisibleDeprecationWarning may become unnecessary.
    """

    def test(self):
        a = np.arange(10)
        assert_warns(np.VisibleDeprecationWarning, np.rank, a)


class TestComparisonDeprecations(_DeprecationTestCase):
    """This tests the deprecation, for non-elementwise comparison logic.
    This used to mean that when an error occured during element-wise comparison
    (i.e. broadcasting) NotImplemented was returned, but also in the comparison
    itself, False was given instead of the error.

    Also test FutureWarning for the None comparison.
    """

    message = "elementwise.* comparison failed; .*"

    def test_normal_types(self):
        for op in (operator.eq, operator.ne):
            # Broadcasting errors:
            self.assert_deprecated(op, args=(np.zeros(3), []))
            a = np.zeros(3, dtype='i,i')
            # (warning is issued a couple of times here)
            self.assert_deprecated(op, args=(a, a[:-1]), num=None)

            # Element comparison error (numpy array can't be compared).
            a = np.array([1, np.array([1,2,3])], dtype=object)
            b = np.array([1, np.array([1,2,3])], dtype=object)
            self.assert_deprecated(op, args=(a, b), num=None)

    def test_string(self):
        # For two string arrays, strings always raised the broadcasting error:
        a = np.array(['a', 'b'])
        b = np.array(['a', 'b', 'c'])
        assert_raises(ValueError, lambda x, y: x == y, a, b)

        # The empty list is not cast to string, this is only to document
        # that fact (it likely should be changed). This means that the
        # following works (and returns False) due to dtype mismatch:
        a == []

    def test_none_comparison(self):
        # Test comparison of None, which should result in elementwise
        # comparison in the future. [1, 2] == None should be [False, False].
        with warnings.catch_warnings():
            warnings.filterwarnings('always', '', FutureWarning)
            assert_warns(FutureWarning, operator.eq, np.arange(3), None)
            assert_warns(FutureWarning, operator.ne, np.arange(3), None)

        with warnings.catch_warnings():
            warnings.filterwarnings('error', '', FutureWarning)
            assert_raises(FutureWarning, operator.eq, np.arange(3), None)
            assert_raises(FutureWarning, operator.ne, np.arange(3), None)

    def test_scalar_none_comparison(self):
        # Scalars should still just return false and not give a warnings.
        # The comparisons are flagged by pep8, ignore that.
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', FutureWarning)
            assert_(not np.float32(1) == None)
            assert_(not np.str_('test') == None)
            # This is dubious (see below):
            assert_(not np.datetime64('NaT') == None)

            assert_(np.float32(1) != None)
            assert_(np.str_('test') != None)
            # This is dubious (see below):
            assert_(np.datetime64('NaT') != None)
        assert_(len(w) == 0)

        # For documentaiton purpose, this is why the datetime is dubious.
        # At the time of deprecation this was no behaviour change, but
        # it has to be considered when the deprecations is done.
        assert_(np.equal(np.datetime64('NaT'), None))

    def test_void_dtype_equality_failures(self):
        class NotArray(object):
            def __array__(self):
                raise TypeError

            # Needed so Python 3 does not raise DeprecationWarning twice.
            def __ne__(self, other):
                return NotImplemented

        self.assert_deprecated(lambda: np.arange(2) == NotArray())
        self.assert_deprecated(lambda: np.arange(2) != NotArray())

        struct1 = np.zeros(2, dtype="i4,i4")
        struct2 = np.zeros(2, dtype="i4,i4,i4")

        assert_warns(FutureWarning, lambda: struct1 == 1)
        assert_warns(FutureWarning, lambda: struct1 == struct2)
        assert_warns(FutureWarning, lambda: struct1 != 1)
        assert_warns(FutureWarning, lambda: struct1 != struct2)

    def test_array_richcompare_legacy_weirdness(self):
        # It doesn't really work to use assert_deprecated here, b/c part of
        # the point of assert_deprecated is to check that when warnings are
        # set to "error" mode then the error is propagated -- which is good!
        # But here we are testing a bunch of code that is deprecated *because*
        # it has the habit of swallowing up errors and converting them into
        # different warnings. So assert_warns will have to be sufficient.
        assert_warns(FutureWarning, lambda: np.arange(2) == "a")
        assert_warns(FutureWarning, lambda: np.arange(2) != "a")
        # No warning for scalar comparisons
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            assert_(not (np.array(0) == "a"))
            assert_(np.array(0) != "a")
            assert_(not (np.int16(0) == "a"))
            assert_(np.int16(0) != "a")

        for arg1 in [np.asarray(0), np.int16(0)]:
            struct = np.zeros(2, dtype="i4,i4")
            for arg2 in [struct, "a"]:
                for f in [operator.lt, operator.le, operator.gt, operator.ge]:
                    if sys.version_info[0] >= 3:
                        # py3
                        with warnings.catch_warnings() as l:
                            warnings.filterwarnings("always")
                            assert_raises(TypeError, f, arg1, arg2)
                            assert not l
                    else:
                        # py2
                        assert_warns(DeprecationWarning, f, arg1, arg2)


class TestIdentityComparisonDeprecations(_DeprecationTestCase):
    """This tests the equal and not_equal object ufuncs identity check
    deprecation. This was due to the usage of PyObject_RichCompareBool.

    This tests that for example for `a = np.array([np.nan], dtype=object)`
    `a == a` it is warned that False and not `np.nan is np.nan` is returned.

    Should be kept in sync with TestComparisonDeprecations and new tests
    added when the deprecation is over. Requires only removing of @identity@
    (and blocks) from the ufunc loops.c.src of the OBJECT comparisons.
    """

    message = "numpy .* will not check object identity in the future."

    def test_identity_equality_mismatch(self):
        a = np.array([np.nan], dtype=object)

        with warnings.catch_warnings():
            warnings.filterwarnings('always', '', FutureWarning)
            assert_warns(FutureWarning, np.equal, a, a)
            assert_warns(FutureWarning, np.not_equal, a, a)

        with warnings.catch_warnings():
            warnings.filterwarnings('error', '', FutureWarning)
            assert_raises(FutureWarning, np.equal, a, a)
            assert_raises(FutureWarning, np.not_equal, a, a)
            # And the other do not warn:
            with np.errstate(invalid='ignore'):
                np.less(a, a)
                np.greater(a, a)
                np.less_equal(a, a)
                np.greater_equal(a, a)

    def test_comparison_error(self):
        class FunkyType(object):
            def __eq__(self, other):
                raise TypeError("I won't compare")

            def __ne__(self, other):
                raise TypeError("I won't compare")

        a = np.array([FunkyType()])
        self.assert_deprecated(np.equal, args=(a, a))
        self.assert_deprecated(np.not_equal, args=(a, a))

    def test_bool_error(self):
        # The comparison result cannot be interpreted as a bool
        a = np.array([np.array([1, 2, 3]), None], dtype=object)
        self.assert_deprecated(np.equal, args=(a, a))
        self.assert_deprecated(np.not_equal, args=(a, a))


class TestAlterdotRestoredotDeprecations(_DeprecationTestCase):
    """The alterdot/restoredot functions are deprecated.

    These functions no longer do anything in numpy 1.10, so should not be
    used.

    """

    def test_alterdot_restoredot_deprecation(self):
        self.assert_deprecated(np.alterdot)
        self.assert_deprecated(np.restoredot)


class TestBooleanIndexShapeMismatchDeprecation():
    """Tests deprecation for boolean indexing where the boolean array
    does not match the input array along the given diemsions.
    """
    message = r"boolean index did not match indexed array"

    def test_simple(self):
        arr = np.ones((5, 4, 3))
        index = np.array([True])
        #self.assert_deprecated(arr.__getitem__, args=(index,))
        assert_warns(np.VisibleDeprecationWarning,
                     arr.__getitem__, index)

        index = np.array([False] * 6)
        #self.assert_deprecated(arr.__getitem__, args=(index,))
        assert_warns(np.VisibleDeprecationWarning,
             arr.__getitem__, index)

        index = np.zeros((4, 4), dtype=bool)
        #self.assert_deprecated(arr.__getitem__, args=(index,))
        assert_warns(np.VisibleDeprecationWarning,
             arr.__getitem__, index)
        #self.assert_deprecated(arr.__getitem__, args=((slice(None), index),))
        assert_warns(np.VisibleDeprecationWarning,
             arr.__getitem__, (slice(None), index))


class TestFullDefaultDtype:
    """np.full defaults to float when dtype is not set.  In the future, it will
    use the fill value's dtype.
    """

    def test_full_default_dtype(self):
        assert_warns(FutureWarning, np.full, 1, 1)
        assert_warns(FutureWarning, np.full, 1, None)


if __name__ == "__main__":
    run_module_suite()
