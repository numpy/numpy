
from numpy import *
from compat import *
from olddefaults import *

import numpy
import compat
import olddefaults

__version__ = numpy.__version__

__all__ = ['__version__']
__all__ += numpy.__all__
__all__ += compat.__all__
for name in olddefaults.__all__:
    if name not in __all__:
        __all__.append(name)

del numpy
del compat
del olddefaults
