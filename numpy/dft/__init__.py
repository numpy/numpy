import warnings
warnings.warn("The dft subpackage will be removed by 1.0 final -- it is now called fft")
from numpy.fft import *
del warnings
