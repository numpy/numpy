"""Don't add these to the __all__ variable though

"""
from __future__ import division

from numpy import *

def _move_axis_to_0(a, axis):
    if axis == 0:
        return a
    n = len(a.shape)
    if axis < 0:
        axis += n
    axes = range(1, axis+1) + [0,] + range(axis+1, n)
    return transpose(a, axes)

# Add these
from compat import *
from functions import *
from precision import *
from ufuncs import *
from misc import *

import compat
import precision
import functions
import misc
import ufuncs
import warnings
import sys

import numpy
__version__ = numpy.__version__
del numpy

__all__ = ['__version__']
__all__ += compat.__all__
__all__ += precision.__all__
__all__ += functions.__all__
__all__ += ufuncs.__all__
__all__ += misc.__all__

del compat
del functions
del precision
del ufuncs
del misc

if sys.version_info[0] > 2:
    raise ImportError("oldnumeric is not supported in Python 3.x")

_msg = "oldnumeric will be dropped in numpy 1.8"
warnings.warn(_msg, DeprecationWarning)

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
