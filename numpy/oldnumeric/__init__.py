
# Don't add these to the __all__ variable
from numpy import *

# Add these
from compat import *
from olddefaults import *
from functions import *

import compat
import olddefaults
import functions

import numpy
__version__ = numpy.__version__
del numpy

__all__ = ['__version__']
__all__ += compat.__all__
__all__ += olddefaults.__all__
__all__ += functions.__all__
        
del compat
del olddefaults
del functions
