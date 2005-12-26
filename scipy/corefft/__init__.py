# To get sub-modules
from info import __doc__

from fftpack import *
from helper import *

# re-define duplicated functions if full scipy installed.
try:
    import scipy.fftpack
except ImportError:
    pass
else:
    fft = scipy.fftpack.fft
    ifft = scipy.fftpack.ifft
    fftn = scipy.fftpack.fftn
    ifftn = scipy.fftpack.ifftn
    fft2 = scipy.fftpack.fft2
    ifft2 = scipy.fftpack.ifft2


from scipy.test.testing import ScipyTest 
test = ScipyTest().test
