
from info_scipy_base import __doc__
from scipy_base_version import scipy_base_version as __version__

from ppimport import ppimport, ppimport_attr

# The following statement is equivalent to
#
#   from Matrix import Matrix as mat
#
# but avoids expensive LinearAlgebra import when
# Matrix is not used.
mat = ppimport_attr(ppimport('Matrix'), 'Matrix')

# Force Numeric to use scipy_base.fastumath instead of Numeric.umath.
import fastumath  # no need to use scipy_base.fastumath
import sys as _sys
_sys.modules['umath'] = fastumath

import Numeric
from Numeric import *

import limits
from type_check import *
from index_tricks import *
from function_base import *
from shape_base import *
from matrix_base import *



from polynomial import *
from scimath import *
from machar import *
from pexec import *

if Numeric.__version__ < '23.5':
    matrixmultiply=dot

Inf = inf = fastumath.PINF
try:
    NAN = NaN = nan = fastumath.NAN
except AttributeError:
    NaN = NAN = nan = fastumath.PINF/fastumath.PINF

from scipy_test.testing import ScipyTest
test = ScipyTest('scipy_base').test

if _sys.modules.has_key('scipy_base.Matrix') \
   and _sys.modules['scipy_base.Matrix'] is None:
    del _sys.modules['scipy_base.Matrix']

