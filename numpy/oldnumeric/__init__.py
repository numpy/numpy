
from numpy import *
from compat import *

import numpy
import compat

__version__ = numpy.__version__

__all__ = ['__version__']
__all__ += numpy.__all__
__all__ += compat.__all__

del numpy
del compat
