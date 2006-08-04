
from numpy import *
from compat import *
from olddefaults import *
from typeconv import *
from functions import *

import numpy
import compat
import olddefaults
import typeconv
import functions

__version__ = numpy.__version__

__all__ = ['__version__']
__all__ += numpy.__all__
__all__ += compat.__all__
__all__ += typeconv.__all__
for name in olddefaults.__all__:
    if name not in __all__:
        __all__.append(name)

for name in functions.__all__:
    if name not in __all__:
        __all__.apend(name)
        
del name
del numpy
del compat
del olddefaults
del functions
del typeconv
