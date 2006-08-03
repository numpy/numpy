
from info import __doc__
from numpy.version import version as __version__

import multiarray
import umath
import numerictypes as nt
multiarray.set_typeDict(nt.sctypeDict)
import _sort
from numeric import *
from fromnumeric import *
from defmatrix import *
import ma
import defchararray as char
import records as rec
from records import *
from memmap import *
from defchararray import *
import scalarmath
del nt

__all__ = ['char','rec','memmap','ma']
__all__ += numeric.__all__
__all__ += fromnumeric.__all__
__all__ += defmatrix.__all__
__all__ += rec.__all__
__all__ += char.__all__



def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
