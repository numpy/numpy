from _typeshed import Incomplete
from collections.abc import Container, Iterable
from typing import Any, Literal as L, Protocol, overload, type_check_only

import numpy as np
from numpy._typing import (
    ArrayLike,
    NDArray,
    _16Bit,
    _32Bit,
    _64Bit,
    _ArrayLike,
    _NestedSequence,
    _ScalarLike_co,
    _SupportsArray,
)

__all__ = [
    "common_type",
    "imag",
    "iscomplex",
    "iscomplexobj",
    "isreal",
    "isrealobj",
    "mintypecode",
    "nan_to_num",
    "real",
    "real_if_close",
    "typename",
]

type _FloatMax32 = np.float32 | np.float16
type _ComplexMax128 = np.complex128 | np.complex64
type _RealMax64 = np.float64 | np.float32 | np.float16 | np.integer
type _Real = np.floating | np.integer
type _ToReal = _Real | np.bool
type _InexactMax32 = np.inexact[_32Bit] | np.float16
type _NumberMax64 = np.number[_64Bit] | np.number[_32Bit] | np.number[_16Bit] | np.integer

@type_check_only
class _HasReal[T](Protocol):
    @property
    def real(self, /) -> T: ...

@type_check_only
class _HasImag[T](Protocol):
    @property
    def imag(self, /) -> T: ...

@type_check_only
class _HasDType[ScalarT: np.generic](Protocol):
    @property
    def dtype(self, /) -> np.dtype[ScalarT]: ...

###

def mintypecode(typechars: Iterable[str | ArrayLike], typeset: str | Container[str] = "GDFgdf", default: str = "d") -> str: ...

#
@overload
def real[T](val: _HasReal[T]) -> T: ...
@overload
def real[RealT: _ToReal](val: _ArrayLike[RealT]) -> NDArray[RealT]: ...
@overload
def real(val: ArrayLike) -> NDArray[Any]: ...

#
@overload
def imag[T](val: _HasImag[T]) -> T: ...
@overload
def imag[RealT: _ToReal](val: _ArrayLike[RealT]) -> NDArray[RealT]: ...
@overload
def imag(val: ArrayLike) -> NDArray[Any]: ...

#
@overload
def iscomplex(x: _ScalarLike_co) -> np.bool: ...
@overload
def iscomplex(x: NDArray[Any] | _NestedSequence[ArrayLike]) -> NDArray[np.bool]: ...
@overload
def iscomplex(x: ArrayLike) -> np.bool | NDArray[np.bool]: ...

#
@overload
def isreal(x: _ScalarLike_co) -> np.bool: ...
@overload
def isreal(x: NDArray[Any] | _NestedSequence[ArrayLike]) -> NDArray[np.bool]: ...
@overload
def isreal(x: ArrayLike) -> np.bool | NDArray[np.bool]: ...

#
def iscomplexobj(x: _HasDType[Any] | ArrayLike) -> bool: ...
def isrealobj(x: _HasDType[Any] | ArrayLike) -> bool: ...

#
@overload
def nan_to_num[ScalarT: np.generic](
    x: ScalarT,
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> ScalarT: ...
@overload
def nan_to_num[ScalarT: np.generic](
    x: NDArray[ScalarT] | _NestedSequence[_ArrayLike[ScalarT]],
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> NDArray[ScalarT]: ...
@overload
def nan_to_num[ScalarT: np.generic](
    x: _SupportsArray[np.dtype[ScalarT]],
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> ScalarT | NDArray[ScalarT]: ...
@overload
def nan_to_num(
    x: _NestedSequence[ArrayLike],
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> NDArray[Incomplete]: ...
@overload
def nan_to_num(
    x: ArrayLike,
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> Incomplete: ...

# NOTE: The [overload-overlap] mypy error is a false positive
@overload
def real_if_close(a: _ArrayLike[np.complex64], tol: float = 100) -> NDArray[np.float32 | np.complex64]: ...
@overload
def real_if_close(a: _ArrayLike[np.complex128], tol: float = 100) -> NDArray[np.float64 | np.complex128]: ...
@overload
def real_if_close(a: _ArrayLike[np.clongdouble], tol: float = 100) -> NDArray[np.longdouble | np.clongdouble]: ...
@overload
def real_if_close[RealT: _ToReal](a: _ArrayLike[RealT], tol: float = 100) -> NDArray[RealT]: ...
@overload
def real_if_close(a: ArrayLike, tol: float = 100) -> NDArray[Any]: ...

#
@overload
def typename(char: L["S1"]) -> L["character"]: ...
@overload
def typename(char: L["?"]) -> L["bool"]: ...
@overload
def typename(char: L["b"]) -> L["signed char"]: ...
@overload
def typename(char: L["B"]) -> L["unsigned char"]: ...
@overload
def typename(char: L["h"]) -> L["short"]: ...
@overload
def typename(char: L["H"]) -> L["unsigned short"]: ...
@overload
def typename(char: L["i"]) -> L["integer"]: ...
@overload
def typename(char: L["I"]) -> L["unsigned integer"]: ...
@overload
def typename(char: L["l"]) -> L["long integer"]: ...
@overload
def typename(char: L["L"]) -> L["unsigned long integer"]: ...
@overload
def typename(char: L["q"]) -> L["long long integer"]: ...
@overload
def typename(char: L["Q"]) -> L["unsigned long long integer"]: ...
@overload
def typename(char: L["f"]) -> L["single precision"]: ...
@overload
def typename(char: L["d"]) -> L["double precision"]: ...
@overload
def typename(char: L["g"]) -> L["long precision"]: ...
@overload
def typename(char: L["F"]) -> L["complex single precision"]: ...
@overload
def typename(char: L["D"]) -> L["complex double precision"]: ...
@overload
def typename(char: L["G"]) -> L["complex long double precision"]: ...
@overload
def typename(char: L["S"]) -> L["string"]: ...
@overload
def typename(char: L["U"]) -> L["unicode"]: ...
@overload
def typename(char: L["V"]) -> L["void"]: ...
@overload
def typename(char: L["O"]) -> L["object"]: ...

# NOTE: The [overload-overlap] mypy errors are false positives
@overload
def common_type() -> type[np.float16]: ...
@overload
def common_type(a0: _HasDType[np.float16], /, *ai: _HasDType[np.float16]) -> type[np.float16]: ...
@overload
def common_type(a0: _HasDType[np.float32], /, *ai: _HasDType[_FloatMax32]) -> type[np.float32]: ...
@overload
def common_type(
    a0: _HasDType[np.float64 | np.integer],
    /,
    *ai: _HasDType[_RealMax64],
) -> type[np.float64]: ...
@overload
def common_type(
    a0: _HasDType[np.longdouble],
    /,
    *ai: _HasDType[_Real],
) -> type[np.longdouble]: ...
@overload
def common_type(
    a0: _HasDType[np.complex64],
    /,
    *ai: _HasDType[_InexactMax32],
) -> type[np.complex64]: ...
@overload
def common_type(
    a0: _HasDType[np.complex128],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[np.clongdouble],
    /,
    *ai: _HasDType[np.number],
) -> type[np.clongdouble]: ...
@overload
def common_type(
    a0: _HasDType[_FloatMax32],
    array1: _HasDType[np.float32],
    /,
    *ai: _HasDType[_FloatMax32],
) -> type[np.float32]: ...
@overload
def common_type(
    a0: _HasDType[_RealMax64],
    array1: _HasDType[np.float64 | np.integer],
    /,
    *ai: _HasDType[_RealMax64],
) -> type[np.float64]: ...
@overload
def common_type(
    a0: _HasDType[_Real],
    array1: _HasDType[np.longdouble],
    /,
    *ai: _HasDType[_Real],
) -> type[np.longdouble]: ...
@overload
def common_type(
    a0: _HasDType[_InexactMax32],
    array1: _HasDType[np.complex64],
    /,
    *ai: _HasDType[_InexactMax32],
) -> type[np.complex64]: ...
@overload
def common_type(
    a0: _HasDType[np.float64],
    array1: _HasDType[_ComplexMax128],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[_ComplexMax128],
    array1: _HasDType[np.float64],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[_NumberMax64],
    array1: _HasDType[np.complex128],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[_ComplexMax128],
    array1: _HasDType[np.complex128 | np.integer],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[np.complex128 | np.integer],
    array1: _HasDType[_ComplexMax128],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[_Real],
    /,
    *ai: _HasDType[_Real],
) -> type[np.floating]: ...
@overload
def common_type(
    a0: _HasDType[np.number],
    array1: _HasDType[np.clongdouble],
    /,
    *ai: _HasDType[np.number],
) -> type[np.clongdouble]: ...
@overload
def common_type(
    a0: _HasDType[np.longdouble],
    array1: _HasDType[np.complexfloating],
    /,
    *ai: _HasDType[np.number],
) -> type[np.clongdouble]: ...
@overload
def common_type(
    a0: _HasDType[np.complexfloating],
    array1: _HasDType[np.longdouble],
    /,
    *ai: _HasDType[np.number],
) -> type[np.clongdouble]: ...
@overload
def common_type(
    a0: _HasDType[np.complexfloating],
    array1: _HasDType[np.number],
    /,
    *ai: _HasDType[np.number],
) -> type[np.complexfloating]: ...
@overload
def common_type(
    a0: _HasDType[np.number],
    array1: _HasDType[np.complexfloating],
    /,
    *ai: _HasDType[np.number],
) -> type[np.complexfloating]: ...
@overload
def common_type(
    a0: _HasDType[np.number],
    array1: _HasDType[np.number],
    /,
    *ai: _HasDType[np.number],
) -> type[Any]: ...
