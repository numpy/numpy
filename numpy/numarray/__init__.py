from __future__ import division

from util import *
from numerictypes import *
from functions import *
from ufuncs import *
from compat import *
from session import *

import util
import numerictypes
import functions
import ufuncs
import compat
import session
import warnings

__all__ = ['session', 'numerictypes']
__all__ += util.__all__
__all__ += numerictypes.__all__
__all__ += functions.__all__
__all__ += ufuncs.__all__
__all__ += compat.__all__
__all__ += session.__all__

del util
del functions
del ufuncs
del compat

_msg = "oldnumeric will be dropped in 1.8"
warnings.warn(_msg, DeprecationWarning)

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
