# To get sub-modules
from info import __doc__

from fftpack import *
from helper import *

try:
    import scipy.fftpack
    fft = scipy.fftpack.fft
    ifft = scipy.fftpack.ifft
except ImportError:
    pass

from scipy.test.testing import ScipyTest 
test = ScipyTest().test
