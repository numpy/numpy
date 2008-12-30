import numpy as np
from numpy.testing import *

import locale
import sys
from StringIO import StringIO

_REF = {np.inf: 'inf', -np.inf: '-inf', np.nan: 'nan', complex(np.inf, 1):
        '(inf+1j)', complex(np.nan, 1): '(nan+1j)', complex(-np.inf, 1):
        '(-inf+1j)'}

if sys.platform == 'win32' and sys.version_info[0] <= 2 and sys.version_info[1] <= 5:
    _REF[np.float32(1e10)] = '1e+010'
else:
    _REF[np.float32(1e10)] = '1e+10'

def check_float_type(tp):
    for x in [0, 1,-1, 1e20] :
        assert_equal(str(tp(x)), str(float(x)),
                     err_msg='Failed str formatting for type %s' % tp)

    if tp(1e10).itemsize > 4:
        assert_equal(str(tp(1e10)), str(float('1e10')),
                     err_msg='Failed str formatting for type %s' % tp)
    else:
        assert_equal(str(tp(1e10)), _REF[tp(1e10)],
                     err_msg='Failed str formatting for type %s' % tp)

def test_float_types():
    """ Check formatting.

        This is only for the str function, and only for simple types.
        The precision of np.float and np.longdouble aren't the same as the
        python float precision.

    """
    for t in [np.float32, np.double, np.longdouble] :
        yield check_float_type, t

def check_nan_inf_float(tp):
    for x in [np.inf, -np.inf, np.nan]:
        assert_equal(str(tp(x)), _REF[x],
                     err_msg='Failed str formatting for type %s' % tp)

def test_nan_inf_float():
    """ Check formatting.

        This is only for the str function, and only for simple types.
        The precision of np.float and np.longdouble aren't the same as the
        python float precision.

    """
    for t in [np.float32, np.double, np.longdouble] :
        yield check_nan_inf_float, t

def check_complex_type(tp):
    for x in [0, 1,-1, 1e20] :
        assert_equal(str(tp(x)), str(complex(x)),
                     err_msg='Failed str formatting for type %s' % tp)
        assert_equal(str(tp(x*1j)), str(complex(x*1j)),
                     err_msg='Failed str formatting for type %s' % tp)
        assert_equal(str(tp(x + x*1j)), str(complex(x + x*1j)),
                     err_msg='Failed str formatting for type %s' % tp)

    if tp(1e10).itemsize > 8:
        assert_equal(str(tp(1e10)), str(complex(1e10)),
                     err_msg='Failed str formatting for type %s' % tp)
    else:
        assert_equal(str(tp(1e10)), _REF[tp(1e10)],
                     err_msg='Failed str formatting for type %s' % tp)

def test_complex_types():
    """Check formatting.

        This is only for the str function, and only for simple types.
        The precision of np.float and np.longdouble aren't the same as the
        python float precision.

    """
    for t in [np.complex64, np.cdouble, np.clongdouble] :
        yield check_complex_type, t

# print tests
def _test_redirected_print(x, tp):
    file = StringIO()
    file_tp = StringIO()
    stdout = sys.stdout
    try:
        sys.stdout = file_tp
        print tp(x)
        sys.stdout = file
        if x in _REF:
            print _REF[x]
        else:
            print x
    finally:
        sys.stdout = stdout

    assert_equal(file.getvalue(), file_tp.getvalue(),
                 err_msg='print failed for type%s' % tp)

def check_float_type_print(tp):
    for x in [0, 1,-1, 1e10, 1e20, np.inf, -np.inf, np.nan]:
        _test_redirected_print(float(x), tp)

def check_complex_type_print(tp):
    # We do not create complex with inf/nan directly because the feature is
    # missing in python < 2.6
    for x in [0, 1, -1, 1e10, 1e20, complex(np.inf, 1),
              complex(np.nan, 1), complex(-np.inf, 1)] :
        _test_redirected_print(complex(x), tp)

def test_float_type_print():
    """Check formatting when using print """
    for t in [np.float32, np.double, np.longdouble] :
        yield check_float_type_print, t

def test_complex_type_print():
    """Check formatting when using print """
    for t in [np.complex64, np.cdouble, np.clongdouble] :
        yield check_complex_type_print, t

# Locale tests: scalar types formatting should be independant of the locale
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

        assert_equal(str(tp(1.2)), str(float(1.2)),
                     err_msg='Failed locale test for type %s' % tp)
    finally:
        locale.setlocale(locale.LC_NUMERIC, locale=curloc)

@np.testing.dec.skipif(not has_french_locale(),
                       "Skipping locale test, French locale not found")
def test_locale_single():
    return _test_locale_independance(np.float32)

@np.testing.dec.skipif(not has_french_locale(),
                       "Skipping locale test, French locale not found")
def test_locale_double():
    return _test_locale_independance(np.double)

@np.testing.dec.skipif(not has_french_locale(),
                       "Skipping locale test, French locale not found")
def test_locale_longdouble():
    return _test_locale_independance(np.longdouble)

if __name__ == "__main__":
    run_module_suite()
