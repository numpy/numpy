
from info import __doc__
from numpy.core_version import version as __version__

import multiarray
import umath
import numerictypes as nt
multiarray.set_typeDict(nt.typeDict)
import _sort
from numeric import *
from oldnumeric import *
from matrix import *
import ma
import chararray as char
import records as rec
from records import *
from memmap import *
del nt

__all__ = filter(lambda s:not s.startswith('_'),dir())

from numpy.testing import ScipyTest 
test = ScipyTest().test
