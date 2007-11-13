import sys
from numpy.testing import *
set_package_path()
from numpy.fft import *
restore_path()

class TestFFTShift(NumpyTestCase):
    def check_fft_n(self):
        self.failUnlessRaises(ValueError,fft,[1,2,3],0)

if __name__ == "__main__":
    NumpyTest().run()
