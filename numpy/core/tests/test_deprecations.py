"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.

"""
from __future__ import division, absolute_import, print_function

import sys
import operator
import warnings
from nose.plugins.skip import SkipTest

import numpy as np
from numpy.testing import dec, run_module_suite, assert_raises


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
                raise AssertionError("expected DeprecationWarning but %s given"
                                                            % warning.category)
        if num_found != num:
            raise AssertionError("%i warnings found but %i expected"
                                                        % (len(self.log), num))

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=self.message,
                                        category=DeprecationWarning)

            try:
                function(*args, **kwargs)
                if exceptions != tuple():
                    raise AssertionError("No error raised during function call")
            except exceptions:
                if exceptions == tuple():
                    raise AssertionError("Error raised during function call")


    def assert_not_deprecated(self, function, args=(), kwargs={}):
        """Test if DeprecationWarnings are given and raised.

        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)


class TestFloatNonIntegerArgumentDeprecation(_DeprecationTestCase):
    """
    These test that ``DeprecationWarning`` is given when you try to use
    non-integers as arguments to for indexing and slicing e.g. ``a[0.0:5]``
    and ``a[0.5]``, or other functions like ``array.reshape(1., -1)``.

    After deprecation, changes need to be done inside conversion_utils.c
    in PyArray_PyIntAsIntp and possibly PyArray_IntpConverter.
    In iterators.c the function slice_GetIndices could be removed in favor
    of its python equivalent and in mapping.c the function _tuple_of_integers
    can be simplified (if ``np.array([1]).__index__()`` is also deprecated).

    As for the deprecation time-frame: via Ralf Gommers,

    "Hard to put that as a version number, since we don't know if the
    version after 1.8 will be 6 months or 2 years after. I'd say 2
    years is reasonable."

    I interpret this to mean 2 years after the 1.8 release. Possibly
    giving a PendingDeprecationWarning before that (which is visible
    by default)

    """
    message = "using a non-integer number instead of an integer " \
              "will result in an error in the future"

    def test_indexing(self):
        a = np.array([[[5]]])
        def assert_deprecated(*args, **kwargs):
            self.assert_deprecated(*args, exceptions=(IndexError,), **kwargs)

        assert_deprecated(lambda: a[0.0])
        assert_deprecated(lambda: a[0, 0.0])
        assert_deprecated(lambda: a[0.0, 0])
        assert_deprecated(lambda: a[0.0, :])
        assert_deprecated(lambda: a[:, 0.0])
        assert_deprecated(lambda: a[:, 0.0, :])
        assert_deprecated(lambda: a[0.0, :, :], num=2) # [1]
        assert_deprecated(lambda: a[0, 0, 0.0])
        assert_deprecated(lambda: a[0.0, 0, 0])
        assert_deprecated(lambda: a[0, 0.0, 0])
        assert_deprecated(lambda: a[-1.4])
        assert_deprecated(lambda: a[0, -1.4])
        assert_deprecated(lambda: a[-1.4, 0])
        assert_deprecated(lambda: a[-1.4, :])
        assert_deprecated(lambda: a[:, -1.4])
        assert_deprecated(lambda: a[:, -1.4, :])
        assert_deprecated(lambda: a[-1.4, :, :], num=2) # [1]
        assert_deprecated(lambda: a[0, 0, -1.4])
        assert_deprecated(lambda: a[-1.4, 0, 0])
        assert_deprecated(lambda: a[0, -1.4, 0])
        # [1] These are duplicate because of the _tuple_of_integers quick check

        # Test that the slice parameter deprecation warning doesn't mask
        # the scalar index warning.
        assert_deprecated(lambda: a[0.0:, 0.0], num=2)
        assert_deprecated(lambda: a[0.0:, 0.0, :], num=2)


    def test_valid_indexing(self):
        a = np.array([[[5]]])
        assert_not_deprecated = self.assert_not_deprecated

        assert_not_deprecated(lambda: a[np.array([0])])
        assert_not_deprecated(lambda: a[[0, 0]])
        assert_not_deprecated(lambda: a[:, [0, 0]])
        assert_not_deprecated(lambda: a[:, 0, :])
        assert_not_deprecated(lambda: a[:, :, :])


    def test_slicing(self):
        a = np.array([[5]])
        def assert_deprecated(*args, **kwargs):
            self.assert_deprecated(*args, exceptions=(IndexError,), **kwargs)

        # start as float.
        assert_deprecated(lambda: a[0.0:])
        assert_deprecated(lambda: a[0:, 0.0:2])
        assert_deprecated(lambda: a[0.0::2, :0])
        assert_deprecated(lambda: a[0.0:1:2, :])
        assert_deprecated(lambda: a[:, 0.0:])
        # stop as float.
        assert_deprecated(lambda: a[:0.0])
        assert_deprecated(lambda: a[:0, 1:2.0])
        assert_deprecated(lambda: a[:0.0:2, :0])
        assert_deprecated(lambda: a[:0.0, :])
        assert_deprecated(lambda: a[:, 0:4.0:2])
        # step as float.
        assert_deprecated(lambda: a[::1.0])
        assert_deprecated(lambda: a[0:, :2:2.0])
        assert_deprecated(lambda: a[1::4.0, :0])
        assert_deprecated(lambda: a[::5.0, :])
        assert_deprecated(lambda: a[:, 0:4:2.0])
        # mixed.
        assert_deprecated(lambda: a[1.0:2:2.0], num=2)
        assert_deprecated(lambda: a[1.0::2.0], num=2)
        assert_deprecated(lambda: a[0:, :2.0:2.0], num=2)
        assert_deprecated(lambda: a[1.0:1:4.0, :0], num=2)
        assert_deprecated(lambda: a[1.0:5.0:5.0, :], num=3)
        assert_deprecated(lambda: a[:, 0.4:4.0:2.0], num=3)
        # should still get the DeprecationWarning if step = 0.
        assert_deprecated(lambda: a[::0.0], function_fails=True)


    def test_valid_slicing(self):
        a = np.array([[[5]]])
        assert_not_deprecated = self.assert_not_deprecated

        assert_not_deprecated(lambda: a[::])
        assert_not_deprecated(lambda: a[0:])
        assert_not_deprecated(lambda: a[:2])
        assert_not_deprecated(lambda: a[0:2])
        assert_not_deprecated(lambda: a[::2])
        assert_not_deprecated(lambda: a[1::2])
        assert_not_deprecated(lambda: a[:2:2])
        assert_not_deprecated(lambda: a[1:2:2])


    def test_non_integer_argument_deprecations(self):
        a = np.array([[5]])

        self.assert_deprecated(np.reshape, args=(a, (1., 1., -1)), num=2)
        self.assert_deprecated(np.reshape, args=(a, (np.array(1.), -1)))
        self.assert_deprecated(np.take, args=(a, [0], 1.))
        self.assert_deprecated(np.take, args=(a, [0], np.float64(1.)))


class TestBooleanArgumentDeprecation(_DeprecationTestCase):
    """This tests that using a boolean as integer argument/indexing is
    deprecated.

    This should be kept in sync with TestFloatNonIntegerArgumentDeprecation
    and like it is handled in PyArray_PyIntAsIntp.
    """
    message = "using a boolean instead of an integer " \
              "will result in an error in the future"

    def test_bool_as_int_argument(self):
        a = np.array([[[1]]])

        self.assert_deprecated(np.reshape, args=(a, (True, -1)))
        self.assert_deprecated(np.reshape, args=(a, (np.bool_(True), -1)))
        # Note that operator.index(np.array(True)) does not work, a boolean
        # array is thus also deprecated, but not with the same message:
        assert_raises(TypeError, operator.index, np.array(True))
        self.assert_deprecated(np.take, args=(a, [0], False))
        self.assert_deprecated(lambda: a[False:True:True], exceptions=IndexError, num=3)
        self.assert_deprecated(lambda: a[False,0], exceptions=IndexError)
        self.assert_deprecated(lambda: a[False,0,0], exceptions=IndexError)


class TestArrayToIndexDeprecation(_DeprecationTestCase):
    """This tests that creating an an index from an array is deprecated
    if the array is not 0d.

    This can probably be deprecated somewhat faster then the integer
    deprecations. The deprecation period started with NumPy 1.8.
    For deprecation this needs changing of array_index in number.c
    """
    message = "converting an array with ndim \> 0 to an index will result " \
              "in an error in the future"

    def test_array_to_index_deprecation(self):
        # This drops into the non-integer deprecation, which is ignored here,
        # so no exception is expected. The raising is effectively tested above.
        a = np.array([[[1]]])

        self.assert_deprecated(operator.index, args=(np.array([1]),))
        self.assert_deprecated(np.reshape, args=(a, (a, -1)), exceptions=())
        self.assert_deprecated(np.take, args=(a, [0], a), exceptions=())
        # Check slicing. Normal indexing checks arrays specifically.
        self.assert_deprecated(lambda: a[a:a:a], exceptions=(), num=3)


if __name__ == "__main__":
    run_module_suite()
