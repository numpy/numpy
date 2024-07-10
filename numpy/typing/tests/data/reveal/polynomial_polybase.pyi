import sys
from collections.abc import Sequence
from fractions import Fraction
from typing import Any, Literal as L, TypeAlias

import numpy as np
import numpy.typing as npt
import numpy.polynomial.polyutils as pu
from numpy.polynomial._polybase import ABCPolyBase

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

_Arr1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.inexact[Any] | np.object_]]
_ArrFloat1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
_ArrComplex1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.complexfloating[Any, Any]]]
_ArrObject1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.object_]]

_Arr1D_2: TypeAlias = np.ndarray[tuple[L[2]], np.dtype[np.inexact[Any] | np.object_]]
_ArrFloat1D_2: TypeAlias = np.ndarray[tuple[L[2]], np.dtype[np.floating[Any]]]
_ArrComplex1D_2: TypeAlias = np.ndarray[tuple[L[2]], np.dtype[np.complexfloating[Any, Any]]]
_ArrObject1D_2: TypeAlias = np.ndarray[tuple[L[2]], np.dtype[np.object_]]

_BasisName: TypeAlias = L["X"]

AR_u1: npt.NDArray[np.uint8]
AR_i2: npt.NDArray[np.int16]
AR_f4: npt.NDArray[np.float32]
AR_c8: npt.NDArray[np.complex64]
AR_O: npt.NDArray[np.object_]

poly_obj: ABCPolyBase[_BasisName]

assert_type(poly_obj.basis_name, _BasisName)
assert_type(poly_obj.coef, _Arr1D)
assert_type(poly_obj.domain, _Arr1D_2)
assert_type(poly_obj.window, _Arr1D_2)

# TODO: ABCPolyBase methods
# TODO: ABCPolyBase operators
