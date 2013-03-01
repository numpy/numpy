from __future__ import division

from numpy.oldnumeric import *
from numpy.lib.user_array import container as UserArray

import numpy.oldnumeric as nold
__all__ = nold.__all__[:]
__all__ += ['UserArray']
del nold
