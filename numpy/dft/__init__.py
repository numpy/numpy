# To get sub-modules
from info import __doc__

from fftpack import *
from helper import *

# re-define duplicated functions if full numpy installed.
try:
    import numpy.fftpack
except ImportError:
    pass
else:
    fft = numpy.fftpack.fft
    ifft = numpy.fftpack.ifft
    fftn = numpy.fftpack.fftn
    ifftn = numpy.fftpack.ifftn
    fft2 = numpy.fftpack.fft2
    ifft2 = numpy.fftpack.ifft2


from numpy.testing import ScipyTest 
test = ScipyTest().test
