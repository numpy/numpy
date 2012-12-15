"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.
"""
import sys
import warnings

from nose.plugins.skip import SkipTest
import numpy as np


def check_for_warning(f, category):
    raised = False
    try:
        f()
    except category:
        raised = True
    print np.__file__
    np.testing.assert_(raised)


def check_does_not_raise(f, category):
    raised = False
    try:
        f()
    except category:
        raised = True
    np.testing.assert_(not raised)


class TestFloatScalarIndexDeprecation(object):
    """
    These test that `DeprecationWarning`s get raised when you
    try to use scalar indices that are not integers e.g. `a[0.0]`,
    `a[1.5, 0]`.

    grep "non-integer scalar" numpy/core/src/multiarray/* for
    all the calls to `DEPRECATE()`, except the one inside
    `_validate_slice_parameter` which handles slicing (but see also
    `TestFloatSliceParameterDeprecation`).

    When 2.4 support is dropped `PyIndex_Check_Or_Unsupported` should
    be removed from npy_pycompat.h and changed to just `PyIndex_Check`.

    As for the deprecation time-frame: via Ralf Gommers,

    "Hard to put that as a version number, since we don't know if the
    version after 1.8 will be 6 months or 2 years after. I'd say 2
    years is reasonable."

    I interpret this to mean 2 years after the 1.8 release.
    """
    def setUp(self):
        warnings.filterwarnings("error", message="non-integer scalar index",
                                category=DeprecationWarning)

    def tearDown(self):
        warnings.filterwarnings("default", message="non-integer scalar index",
                                category=DeprecationWarning)

    def test_deprecations(self):
        a = np.array([[[5]]])
        if sys.version_info[:2] < (2, 5):
            raise SkipTest()
        yield check_for_warning, lambda: a[0.0], DeprecationWarning
        yield check_for_warning, lambda: a[0, 0.0], DeprecationWarning
        yield check_for_warning, lambda: a[0.0, 0], DeprecationWarning
        yield check_for_warning, lambda: a[0.0, :], DeprecationWarning
        yield check_for_warning, lambda: a[:, 0.0], DeprecationWarning
        yield check_for_warning, lambda: a[:, 0.0, :], DeprecationWarning
        yield check_for_warning, lambda: a[0.0, :, :], DeprecationWarning
        yield check_for_warning, lambda: a[0, 0, 0.0], DeprecationWarning
        yield check_for_warning, lambda: a[0.0, 0, 0], DeprecationWarning
        yield check_for_warning, lambda: a[0, 0.0, 0], DeprecationWarning
        yield check_for_warning, lambda: a[-1.4], DeprecationWarning
        yield check_for_warning, lambda: a[0, -1.4], DeprecationWarning
        yield check_for_warning, lambda: a[-1.4, 0], DeprecationWarning
        yield check_for_warning, lambda: a[-1.4, :], DeprecationWarning
        yield check_for_warning, lambda: a[:, -1.4], DeprecationWarning
        yield check_for_warning, lambda: a[:, -1.4, :], DeprecationWarning
        yield check_for_warning, lambda: a[-1.4, :, :], DeprecationWarning
        yield check_for_warning, lambda: a[0, 0, -1.4], DeprecationWarning
        yield check_for_warning, lambda: a[-1.4, 0, 0], DeprecationWarning
        yield check_for_warning, lambda: a[0, -1.4, 0], DeprecationWarning
        # Test that the slice parameter deprecation warning doesn't mask
        # the scalar index warning.
        yield check_for_warning, lambda: a[0.0:, 0.0], DeprecationWarning
        yield check_for_warning, lambda: a[0.0:, 0.0, :], DeprecationWarning

    def test_valid_not_deprecated(self):
        a = np.array([[[5]]])
        yield (check_does_not_raise, lambda: a[np.array([0])],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[[0, 0]],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[:, [0, 0]],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[:, 0, :],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[:, :, :],
               DeprecationWarning)


class TestFloatSliceParameterDeprecation(object):
    """
    These test that `DeprecationWarning`s get raised when you
    try to use  non-integers for slicing, e.g `a[0.0:5]`, `a[::1.5]`,
    etc.

    When this is changed to an error, `slice_GetIndices` and
    `_validate_slice_parameter` should probably be removed. Calls to
    `slice_GetIndices` should be replaced by the standard Python
    API call `PySlice_GetIndicesEx`, since `slice_GetIndices`
    implements the same thing but with a) int coercion and b) Python
    < 2.3 backwards compatibility (which we have long since dropped
    as of this writing).

    As for the deprecation time-frame: via Ralf Gommers,

    "Hard to put that as a version number, since we don't know if the
    version after 1.8 will be 6 months or 2 years after. I'd say 2
    years is reasonable."

    I interpret this to mean 2 years after the 1.8 release.
    """
    def setUp(self):
        warnings.filterwarnings("error", message="non-integer slice param",
                                category=DeprecationWarning)

    def tearDown(self):
        warnings.filterwarnings("default", message="non-integer slice param",
                                category=DeprecationWarning)

    def test_deprecations(self):
        a = np.array([[5]])
        if sys.version_info[:2] < (2, 5):
            raise SkipTest()
        # start as float.
        yield check_for_warning, lambda: a[0.0:], DeprecationWarning
        yield check_for_warning, lambda: a[0:, 0.0:2], DeprecationWarning
        yield check_for_warning, lambda: a[0.0::2, :0], DeprecationWarning
        yield check_for_warning, lambda: a[0.0:1:2, :], DeprecationWarning
        yield check_for_warning, lambda: a[:, 0.0:], DeprecationWarning
        # stop as float.
        yield check_for_warning, lambda: a[:0.0], DeprecationWarning
        yield check_for_warning, lambda: a[:0, 1:2.0], DeprecationWarning
        yield check_for_warning, lambda: a[:0.0:2, :0], DeprecationWarning
        yield check_for_warning, lambda: a[:0.0, :], DeprecationWarning
        yield check_for_warning, lambda: a[:, 0:4.0:2], DeprecationWarning
        # step as float.
        yield check_for_warning, lambda: a[::1.0], DeprecationWarning
        yield check_for_warning, lambda: a[0:, :2:2.0], DeprecationWarning
        yield check_for_warning, lambda: a[1::4.0, :0], DeprecationWarning
        yield check_for_warning, lambda: a[::5.0, :], DeprecationWarning
        yield check_for_warning, lambda: a[:, 0:4:2.0], DeprecationWarning
        # mixed.
        yield check_for_warning, lambda: a[1.0:2:2.0], DeprecationWarning
        yield check_for_warning, lambda: a[1.0::2.0], DeprecationWarning
        yield check_for_warning, lambda: a[0:, :2.0:2.0], DeprecationWarning
        yield check_for_warning, lambda: a[1.0:1:4.0, :0], DeprecationWarning
        yield check_for_warning, lambda: a[1.0:5.0:5.0, :], DeprecationWarning
        yield check_for_warning, lambda: a[:, 0.4:4.0:2.0], DeprecationWarning
        # should still get the DeprecationWarning if step = 0.
        yield check_for_warning, lambda: a[::0.0], DeprecationWarning

    def test_valid_not_deprecated(self):
        a = np.array([[[5]]])
        yield (check_does_not_raise, lambda: a[::],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[0:],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[:2],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[0:2],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[::2],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[1::2],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[:2:2],
               DeprecationWarning)
        yield (check_does_not_raise, lambda: a[1:2:2],
               DeprecationWarning)
