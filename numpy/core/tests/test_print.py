import numpy as np
from numpy.testing import *

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

if __name__ == "__main__":
    run_module_suite()
