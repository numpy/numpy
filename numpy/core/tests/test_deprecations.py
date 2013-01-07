"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.
"""
import sys

import warnings
from nose.plugins.skip import SkipTest

import numpy as np
from numpy.testing import dec, run_module_suite, assert_raises
from numpy.testing.utils import WarningManager


class _DeprecationTestCase(object):
    # Just as warning: warnings uses re.match, so the start of this message
    # must match.
    message = ''

    def setUp(self):
        self.warn_ctx = WarningManager(record=True)
        self.log = self.warn_ctx.__enter__()

        # make sure we are ignoring other types of DeprecationWarnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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
        # reset the log
        if num_found != num:
            raise AssertionError("%i warnings found but %i expected"
                                                        % (len(self.log), num))
        self.log[:] = []

        warnings.filterwarnings("error", message=self.message,
                                    category=DeprecationWarning)

        try:
            function(*args, **kwargs)
            if exceptions != tuple():
                raise AssertionError("No error raised during function call")
        except exceptions:
            if exceptions == tuple():
                raise AssertionError("Error raised during function call")
        finally:
            warnings.filters.pop(0)


    def assert_not_deprecated(self, function, args=(), kwargs={}):
        """Test if DeprecationWarnings are given and raised.
        
        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)


class TestFloatScalarIndexDeprecation(_DeprecationTestCase):
    """
    These test that ``DeprecationWarning`` gets raised when you try to use
    scalar indices that are not integers e.g. ``a[0.0]``, ``a[1.5, 0]``.

    grep "non-integer scalar" numpy/core/src/multiarray/* for all the calls
    to ``DEPRECATE()``, except the one inside ``_validate_slice_parameter``
    which handles slicing (but see also
    `TestFloatSliceParameterDeprecation`).

    When 2.4 support is dropped ``PyIndex_Check_Or_Unsupported`` should be
    removed from ``npy_pycompat.h`` and changed to just ``PyIndex_Check``.

    As for the deprecation time-frame: via Ralf Gommers,

    "Hard to put that as a version number, since we don't know if the
    version after 1.8 will be 6 months or 2 years after. I'd say 2
    years is reasonable."

    I interpret this to mean 2 years after the 1.8 release.

    """
    message = "using a non-integer number instead of an integer will result in an error in the future"

    def test_deprecations(self):
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


    def test_valid_not_deprecated(self):
        a = np.array([[[5]]])
        assert_not_deprecated = self.assert_not_deprecated

        assert_not_deprecated(lambda: a[np.array([0])])
        assert_not_deprecated(lambda: a[[0, 0]])
        assert_not_deprecated(lambda: a[:, [0, 0]])
        assert_not_deprecated(lambda: a[:, 0, :])
        assert_not_deprecated(lambda: a[:, :, :])


class TestFloatSliceParameterDeprecation(_DeprecationTestCase):
    """
    These test that ``DeprecationWarning`` gets raised when you try to use
    non-integers for slicing, e.g. ``a[0.0:5]``, ``a[::1.5]``, etc.

    When this is changed to an error, ``slice_GetIndices`` and
    ``_validate_slice_parameter`` should probably be removed. Calls to
    ``slice_GetIndices`` should be replaced by the standard Python API call
    ``PySlice_GetIndicesEx``, since ``slice_GetIndices`` implements the
    same thing but with int coercion and Python < 2.3 backwards
    compatibility (which we have long since dropped as of this writing).

    As for the deprecation time-frame: via Ralf Gommers,

    "Hard to put that as a version number, since we don't know if the
    version after 1.8 will be 6 months or 2 years after. I'd say 2 years is
    reasonable."

    I interpret this to mean 2 years after the 1.8 release.

    """
    message = "using a non-integer number instead of an integer will result in an error in the future"

    def test_deprecations(self):
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


    def test_valid_not_deprecated(self):
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


if __name__ == "__main__":
    run_module_suite()
