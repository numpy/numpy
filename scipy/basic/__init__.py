# To get sub-modules
from info import __doc__

import fftpack
from fftpack import fft, ifft
import linalg
import random
from random import rand, randn

from scipy.test.testing import ScipyTest 
test = ScipyTest('scipy.basic').test
