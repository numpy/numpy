"""Imports from Numeric for numerix, the numarray/Numeric interchangeability
module.  These array functions are used when Numeric is chosen.
"""
import Numeric
from Numeric import *

import fastumath
from fastumath import *
from fastumath import PINF as inf
from fastumath import alter_numeric, restore_numeric

import _compiled_base
from _compiled_base import arraymap, _unique, _insert

from ppimport import ppimport, ppimport_attr

class _TypeNamespace:
    """Numeric compatible type aliases for use with extension functions."""
    Int8          = Int8
    UInt8         = UInt8
    Int16         = Int16
    UInt16        = UInt16
    Int32         = Int32
    UInt32        = UInt32
    Float32       = Float32
    Float64       = Float64
    Complex32     = Complex32
    Complex64     = Complex64

nx = _TypeNamespace()

# inf is useful for testing infinities in results of array divisions
# (which don't raise exceptions)

inf = infty = Infinity = (array([1])/0.0)[0]

# The following import statements are equivalent to
#
#   from Matrix import Matrix as mat
#
# but avoids expensive LinearAlgebra import when
# Matrix is not used.
#
LinearAlgebra = ppimport('LinearAlgebra')
inverse = ppimport_attr(LinearAlgebra, 'inverse')
eigenvectors = ppimport_attr(LinearAlgebra, 'eigenvectors')
Matrix = mat = ppimport_attr(ppimport('Matrix'), 'Matrix')
fft = ppimport_attr(ppimport('FFT'), 'fft')
RandomArray =  ppimport('RandomArray')
MLab = ppimport('MLab')

NUMERIX_HEADER = "Numeric/arrayobject.h"

#
# Force numerix to use scipy_base.fastumath instead of numerix.umath.
#
import sys as _sys
_sys.modules['umath'] = fastumath


if Numeric.__version__ < '23.5':
    matrixmultiply=dot

Inf = inf = fastumath.PINF
try:
    NAN = NaN = nan = fastumath.NAN
except AttributeError:
    NaN = NAN = nan = fastumath.PINF/fastumath.PINF

try:
    from Numeric import UfuncType
except ImportError:
    UfuncType = type(Numeric.sin)

__all__ = []
for k in globals().keys():
    if k[0] != "_":
        __all__.append(k)
__all__.append("_insert")
__all__.append("_unique")

