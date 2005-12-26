
from info import __doc__
from scipy.core_version import version as __version__

import multiarray
import umath
import numerictypes as nt
multiarray.set_typeDict(nt.typeDict)
from numeric import *
from oldnumeric import *
from matrix import *
from type_check import *
from index_tricks import *
from function_base import *
from shape_base import *
from twodim_base import *
from ufunclike import *

import scimath as math
from polynomial import *
from machar import *
from getlimits import *
import ma
import chararray as char
import records as rec
from records import *
from memmap import *
import convertcode
del nt

from utils import *

__all__ = filter(lambda s:not s.startswith('_'),dir())

from scipy.testing import ScipyTest 
test = ScipyTest().test
