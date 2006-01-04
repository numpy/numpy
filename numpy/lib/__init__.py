
from info import __doc__
from numpy.version import version as __version__

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
import convertcode
del nt

from utils import *

__all__ = filter(lambda s:not s.startswith('_'),dir())

from numpy.testing import ScipyTest 
test = ScipyTest().test
