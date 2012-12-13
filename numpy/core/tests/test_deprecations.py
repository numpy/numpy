"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.
"""
import numpy as np
import warnings


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


class TestFloatIndexDeprecation(object):
    """
    These test that DeprecationWarnings get raised when you
    try to use scalar indices that are not integers e.g. a[0.0],
    a[1.5, 0].

    grep PyIndex_Check_Or_Unsupported numpy/core/src/multiarray/* for
    all the calls to DEPRECATE().

    When 2.4 support is dropped PyIndex_Check_Or_Unsupported should
    be removed from npy_pycompat.h and changed to just PyIndex_Check.
    """
    def setUp(self):
        warnings.filterwarnings("error", message="non-integer",
                                category=DeprecationWarning)

    def tearDown(self):
        warnings.filterwarnings("default", message="non-integer",
                                category=DeprecationWarning)

    def test_deprecations(self):
        a = np.array([[[5]]])
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
