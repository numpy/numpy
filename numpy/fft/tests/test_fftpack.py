import sys
from numpy.testing import *
set_package_path()
import numpy as N
restore_path()

def fft1(x):
    L = len(x)
    phase = -2j*N.pi*(N.arange(L)/float(L))
    phase = N.arange(L).reshape(-1,1) * phase
    return N.sum(x*N.exp(phase),axis=1)

class TestFFTShift(NumpyTestCase):
    def check_fft_n(self):
        self.failUnlessRaises(ValueError,N.fft.fft,[1,2,3],0)

class TestFFT1D(NumpyTestCase):
    def check_basic(self):
        rand = N.random.random
        x = rand(30) + 1j*rand(30)
        assert_array_almost_equal(fft1(x), N.fft.fft(x))

if __name__ == "__main__":
    NumpyTest().run()
