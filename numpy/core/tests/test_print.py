import numpy as np
from numpy.testing import *

import locale
import sys

def check_float_type(tp):
    for x in [0, 1,-1, 1e10, 1e20] :
        assert_equal(str(tp(x)), str(float(x)))

def test_float_types():
    """ Check formatting.

        This is only for the str function, and only for simple types.
        The precision of np.float and np.longdouble aren't the same as the
        python float precision.

    """
    for t in [np.float32, np.double, np.longdouble] :
        yield check_float_type, t

def check_nan_inf_float(tp):
    for x in [float('inf'), float('-inf'), float('nan')]:
        assert_equal(str(tp(x)), str(float(x)))

def test_nan_inf_float():
    """ Check formatting.

        This is only for the str function, and only for simple types.
        The precision of np.float and np.longdouble aren't the same as the
        python float precision.

    """
    for t in [np.float32, np.double, np.longdouble] :
        yield check_nan_inf_float, t

def check_complex_type(tp):
    for x in [0, 1,-1, 1e10, 1e20] :
        assert_equal(str(tp(x)), str(complex(x)))
        assert_equal(str(tp(x*1j)), str(complex(x*1j)))
        assert_equal(str(tp(x + x*1j)), str(complex(x + x*1j)))

def test_complex_types():
    """Check formatting.

        This is only for the str function, and only for simple types.
        The precision of np.float and np.longdouble aren't the same as the
        python float precision.

    """
    for t in [np.complex64, np.cdouble, np.clongdouble] :
        yield check_complex_type, t

def has_french_locale():
    curloc = locale.getlocale(locale.LC_NUMERIC)
    try:
        if not sys.platform == 'win32':
            locale.setlocale(locale.LC_NUMERIC, 'fr_FR')
        else:
            locale.setlocale(locale.LC_NUMERIC, 'FRENCH')

        st = True
    except:
        st = False
    finally:
        locale.setlocale(locale.LC_NUMERIC, locale=curloc)

    return st

def _test_locale_independance(tp):
    # XXX: How to query locale on a given system ?

    # French is one language where the decimal is ',' not '.', and should be
    # relatively common on many systems
    curloc = locale.getlocale(locale.LC_NUMERIC)
    try:
        if not sys.platform == 'win32':
            locale.setlocale(locale.LC_NUMERIC, 'fr_FR')
        else:
            locale.setlocale(locale.LC_NUMERIC, 'FRENCH')

        assert_equal(str(tp(1.2)), str(float(1.2)))
    finally:
        locale.setlocale(locale.LC_NUMERIC, locale=curloc)

@np.testing.dec.skipif(not has_french_locale(),
                       "Skipping locale test, French locale not found")
def test_locale_single():
    return _test_locale_independance(np.float32)

@np.testing.dec.skipif(not has_french_locale(),
                       "Skipping locale test, French locale not found")
def test_locale_double():
    return _test_locale_independance(np.float64)

@np.testing.dec.skipif(not has_french_locale(),
                       "Skipping locale test, French locale not found")
def test_locale_longdouble():
    return _test_locale_independance(np.float96)

if __name__ == "__main__":
    run_module_suite()
