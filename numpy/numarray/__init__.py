from __future__ import division, absolute_import, print_function

from .util import *
from .numerictypes import *
from .functions import *
from .ufuncs import *
from .compat import *
from .session import *

from . import util
from . import numerictypes
from . import functions
from . import ufuncs
from . import compat
from . import session

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

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
