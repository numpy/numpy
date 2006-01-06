
from info import __doc__
from numpy.version import version as __version__

import multiarray
import umath
import numerictypes as nt
multiarray.set_typeDict(nt.typeDict)
import _sort
from numeric import *
from oldnumeric import *
from defmatrix import *
import ma
import chararray as char
import records as rec
from records import *
from memmap import *
from chararray import *
del nt

__all__ = ['char','rec','memmap','ma']
__all__ += numeric.__all__
__all__ += oldnumeric.__all__
__all__ += defmatrix.__all__
__all__ += records.__all__
__all__ += char.__all__

from numpy.testing import ScipyTest 
test = ScipyTest().test
