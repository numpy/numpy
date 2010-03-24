from info import __doc__
from numpy.version import version as __version__

from type_check import *
from index_tricks import *
from function_base import *
from shape_base import *
from stride_tricks import *
from twodim_base import *
from ufunclike import *

import scimath as emath
from polynomial import *
#import convertcode
from utils import *
from arraysetops import *
from npyio import *
from financial import *
import math
from arrayterator import *

__all__ = ['emath','math']
__all__ += type_check.__all__
__all__ += index_tricks.__all__
__all__ += function_base.__all__
__all__ += shape_base.__all__
__all__ += stride_tricks.__all__
__all__ += twodim_base.__all__
__all__ += ufunclike.__all__
__all__ += polynomial.__all__
__all__ += utils.__all__
__all__ += arraysetops.__all__
__all__ += npyio.__all__
__all__ += financial.__all__

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
