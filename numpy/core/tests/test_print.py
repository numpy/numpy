import sys

import numpy as np
from numpy.testing import *

class TestPrint(TestCase):

    def _test_float_type(self, type):
        """ Check formatting.

            This is only for the str function, and only for simple types.
            The precision of np.float and np.longdouble aren't the same as the
            python float precision.

        """
        for x in [0, 1,-1, 1e10, 1e20] :
            assert_equal(str(type(x)), str(float(x)))

    def test_float(self) :
        self._test_float_type(np.float)

    def test_double(self) :
        self._test_float_type(np.double)

    # Not really failure on win32, but with mingw. Get this information this
    # information is more than I want to do.
    @dec.knownfailureif(sys.platform == 'win32', 
                      "long double print is known to fail on windows")
    def test_longdouble(self) :
        self._test_float_type(np.longdouble)

    def _test_complex_type(self, type) :
        """Check formatting.

            This is only for the str function, and only for simple types.
            The precision of np.float and np.longdouble aren't the same as the
            python float precision.

        """
        for x in [0, 1,-1, 1e10, 1e20] :
            assert_equal(str(type(x)), str(complex(x)))
            assert_equal(str(type(x*1j)), str(complex(x*1j)))
            assert_equal(str(type(x + x*1j)), str(complex(x + x*1j)))

    def test_complex_float(self) :
        self._test_complex_type(np.cfloat)

    def test_complex_double(self) :
        self._test_complex_type(np.cdouble)

    # Not really failure on win32, but with mingw. Get this information this
    # information is more than I want to do.
    @dec.knownfailureif(sys.platform == 'win32', 
                      "complex long double print is known to fail on windows")
    def test_complex_longdouble(self) :
        self._test_complex_type(np.clongdouble)


if __name__ == "__main__":
    run_module_suite()
