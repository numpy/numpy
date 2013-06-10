"""Don't add these to the __all__ variable though

"""
from __future__ import division, absolute_import, print_function

import warnings

from numpy import *

_msg = "The oldnumeric module will be dropped in Numpy 1.9"
warnings.warn(_msg, ModuleDeprecationWarning)


def _move_axis_to_0(a, axis):
    if axis == 0:
        return a
    n = len(a.shape)
    if axis < 0:
        axis += n
    axes = list(range(1, axis+1)) + [0,] + list(range(axis+1, n))
    return transpose(a, axes)

# Add these
from .compat import *
from .functions import *
from .precision import *
from .ufuncs import *
from .misc import *

from . import compat
from . import precision
from . import functions
from . import misc
from . import ufuncs

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

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
