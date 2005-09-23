
from info_scipy_base import __doc__
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

import scimath as math
from polynomial import *
from machar import *
from getlimits import *
import ma
import convertcode

del nt

