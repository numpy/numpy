"""
Discrete Fast Fourier Transform (FFT)
=====================================

========= =========================================================
Standard FFTs
===================================================================
fft       Discrete Fourier transform.
ifft      Inverse discrete Fourier transform.
fft2      Discrete Fourier transform in two dimensions.
ifft2     Inverse discrete Fourier transform in two dimensions.
fftn      Discrete Fourier transform in N-dimensions.
ifftn     Inverse discrete Fourier transform in N dimensions.
========= =========================================================

========= ==========================================================
Real FFTs
====================================================================
rfft      Real discrete Fourier transform.
irfft     Inverse real discrete Fourier transform.
rfft2     Real discrete Fourier transform in two dimensions.
irfft2    Inverse real discrete Fourier transform in two dimensions.
rfftn     Real discrete Fourier transform in N dimensions.
irfftn    Inverse real discrete Fourier transform in N dimensions.
========= ==========================================================

========= =========================================================
Hermite FFTs
===================================================================
hfft      Hermite discrete Fourier transform.
ihfft     Inverse hermite discrete Fourier transform.
========= =========================================================

"""
# To get sub-modules
from info import __doc__

from fftpack import *
from helper import *

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
