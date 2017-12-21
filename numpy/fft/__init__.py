from __future__ import division, absolute_import, print_function

# To get sub-modules
from .info import __doc__

from .fftpack import (absolute_import, array, asarray, conjugate, division,
                      fft, fft2, fftn, fftpack, hfft, ifft, ifft2, ifftn,
                      ihfft, irfft, irfft2, irfftn, print_function, rfft,
                      rfft2, rfftn, shape, sqrt, swapaxes, take, zeros)
from .helper import (absolute_import, arange, asarray, collections,
                     concatenate, division, empty, fftfreq, fftshift,
                     ifftshift, integer, integer_types, print_function,
                     rfftfreq, take, threading)

from numpy.testing import _numpy_tester
test = _numpy_tester().test
bench = _numpy_tester().bench
