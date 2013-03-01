"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.

"""
from __future__ import division

import sys
import warnings
from nose.plugins.skip import SkipTest

import numpy as np
from numpy.testing import dec, run_module_suite, assert_raises


def assert_deprecated(f, *args, **kwargs):
    """Check if DeprecationWarning raised as error.

    The warning environment is assumed to have been set up so that the
    appropriate DeprecationWarning has been turned into an error. We do not
    use assert_warns here as the desire is to check that an error will be
    raised if the deprecation is changed to an error and there may be other
    errors that would override a warning. It is a fine point as to which
    error should appear first.

    Parameters
    ----------
    f : callable
       A function that will exhibit the deprecation. It need not be
       deprecated itself, but can be used to execute deprecated code.

    """
    assert_raises(DeprecationWarning, f, *args, **kwargs)


def assert_not_deprecated(f, *args, **kwargs):
    """Check that DeprecationWarning not raised as error.

    The warning environment is assumed to have been set up so that the
    appropriate DeprecationWarning has been turned into an error. This
    function checks that no warning is raised when `f` is executed.

    Parameters
    ----------
    f : callable
       A function that will exhibit no deprecation. It can be used to
       execute code that should not raise DeprecationWarning.

    """
    try:
        f(*args, **kwargs)
    except DeprecationWarning:
        raise AssertionError()


class TestFloatScalarIndexDeprecation(object):
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

    def setUp(self):
        warnings.filterwarnings("error", message="non-integer scalar index",
                                category=DeprecationWarning)


    def tearDown(self):
        warnings.filterwarnings("default", message="non-integer scalar index",
                                category=DeprecationWarning)


    @dec.skipif(sys.version_info[:2] < (2, 5))
    def test_deprecations(self):
        a = np.array([[[5]]])

        assert_deprecated(lambda: a[0.0])
        assert_deprecated(lambda: a[0.0])
        assert_deprecated(lambda: a[0, 0.0])
        assert_deprecated(lambda: a[0.0, 0])
        assert_deprecated(lambda: a[0.0, :])
        assert_deprecated(lambda: a[:, 0.0])
        assert_deprecated(lambda: a[:, 0.0, :])
        assert_deprecated(lambda: a[0.0, :, :])
        assert_deprecated(lambda: a[0, 0, 0.0])
        assert_deprecated(lambda: a[0.0, 0, 0])
        assert_deprecated(lambda: a[0, 0.0, 0])
        assert_deprecated(lambda: a[-1.4])
        assert_deprecated(lambda: a[0, -1.4])
        assert_deprecated(lambda: a[-1.4, 0])
        assert_deprecated(lambda: a[-1.4, :])
        assert_deprecated(lambda: a[:, -1.4])
        assert_deprecated(lambda: a[:, -1.4, :])
        assert_deprecated(lambda: a[-1.4, :, :])
        assert_deprecated(lambda: a[0, 0, -1.4])
        assert_deprecated(lambda: a[-1.4, 0, 0])
        assert_deprecated(lambda: a[0, -1.4, 0])
        # Test that the slice parameter deprecation warning doesn't mask
        # the scalar index warning.
        assert_deprecated(lambda: a[0.0:, 0.0])
        assert_deprecated(lambda: a[0.0:, 0.0, :])


    def test_valid_not_deprecated(self):
        a = np.array([[[5]]])

        assert_not_deprecated(lambda: a[np.array([0])])
        assert_not_deprecated(lambda: a[[0, 0]])
        assert_not_deprecated(lambda: a[:, [0, 0]])
        assert_not_deprecated(lambda: a[:, 0, :])
        assert_not_deprecated(lambda: a[:, :, :])


class TestFloatSliceParameterDeprecation(object):
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

    def setUp(self):
        warnings.filterwarnings("error", message="non-integer slice param",
                                category=DeprecationWarning)


    def tearDown(self):
        warnings.filterwarnings("default", message="non-integer slice param",
                                category=DeprecationWarning)


    @dec.skipif(sys.version_info[:2] < (2, 5))
    def test_deprecations(self):
        a = np.array([[5]])

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
        assert_deprecated(lambda: a[1.0:2:2.0])
        assert_deprecated(lambda: a[1.0::2.0])
        assert_deprecated(lambda: a[0:, :2.0:2.0])
        assert_deprecated(lambda: a[1.0:1:4.0, :0])
        assert_deprecated(lambda: a[1.0:5.0:5.0, :])
        assert_deprecated(lambda: a[:, 0.4:4.0:2.0])
        # should still get the DeprecationWarning if step = 0.
        assert_deprecated(lambda: a[::0.0])


    def test_valid_not_deprecated(self):
        a = np.array([[[5]]])

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
