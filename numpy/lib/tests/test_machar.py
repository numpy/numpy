from numpy.testing import NumpyTestCase, NumpyTest

from numpy.lib.machar import MachAr
import numpy.core.numerictypes as ntypes
from numpy import seterr, array

class TestMachAr(NumpyTestCase):
    def _run_machar_highprec(self):
        # Instanciate MachAr instance with high enough precision to cause
        # underflow
        try:
            hiprec = ntypes.float96
            machar = MachAr(lambda v:array([v], hiprec))
        except AttributeError:
            print "Skipping test: no nyptes.float96 available on this" \
                  " platform."

    def test_underlow(self):
        """Regression testing for #759: instanciating MachAr for dtype =
        np.float96 raises spurious warning."""
        serrstate = seterr(all='raise')
        try:
            try:
                self._run_machar_highprec()
            except FloatingPointError, e:
                self.fail("Caught %s exception, should not have been raised." % e)
        finally:
            seterr(**serrstate)

if __name__ == "__main__":
    NumpyTest().run()
