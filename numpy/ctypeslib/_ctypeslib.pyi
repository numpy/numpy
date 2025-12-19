import ctypes as ct
from _typeshed import StrOrBytesPath
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar, Literal as L, overload

import numpy as np
from numpy._core._internal import _ctypes
from numpy._core.multiarray import flagsobj
from numpy._typing import (
    DTypeLike,
    NDArray,
    _AnyShape,
    _ArrayLike,
    _BoolCodes,
    _DTypeLike,
    _Float32Codes,
    _Float64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _IntCCodes,
    _IntPCodes,
    _LongCodes,
    _LongDoubleCodes,
    _LongLongCodes,
    _ShapeLike,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _UIntCCodes,
    _UIntPCodes,
    _ULongCodes,
    _ULongLongCodes,
    _VoidDTypeLike,
)

__all__ = ["load_library", "ndpointer", "c_intp", "as_ctypes", "as_array", "as_ctypes_type"]

type _FlagsKind = L[
    "C_CONTIGUOUS", "CONTIGUOUS", "C",
    "F_CONTIGUOUS", "FORTRAN", "F",
    "ALIGNED", "A",
    "WRITEABLE", "W",
    "OWNDATA", "O",
    "WRITEBACKIFCOPY", "X",
]

# TODO: Add a shape type parameter
class _ndptr[OptionalDTypeT: np.dtype | None](ct.c_void_p):
    # In practice these 4 classvars are defined in the dynamic class
    # returned by `ndpointer`
    _dtype_: OptionalDTypeT = ...
    _shape_: ClassVar[_AnyShape | None] = ...
    _ndim_: ClassVar[int | None] = ...
    _flags_: ClassVar[list[_FlagsKind] | None] = ...

    @overload  # type: ignore[override]
    @classmethod
    def from_param(cls: type[_ndptr[None]], obj: np.ndarray) -> _ctypes[Any]: ...
    @overload
    @classmethod
    def from_param[DTypeT: np.dtype](cls: type[_ndptr[DTypeT]], obj: np.ndarray[Any, DTypeT]) -> _ctypes[Any]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

class _concrete_ndptr[DTypeT: np.dtype](_ndptr[DTypeT]):
    _dtype_: DTypeT = ...
    _shape_: ClassVar[_AnyShape] = ...  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def contents(self) -> np.ndarray[_AnyShape, DTypeT]: ...

def load_library(libname: StrOrBytesPath, loader_path: StrOrBytesPath) -> ct.CDLL: ...

c_intp = ct.c_int64  # most platforms are 64-bit nowadays

#
@overload
def ndpointer(
    dtype: None = None,
    ndim: int | None = None,
    shape: _ShapeLike | None = None,
    flags: _FlagsKind | Iterable[_FlagsKind] | int | flagsobj | None = None,
) -> type[_ndptr[None]]: ...
@overload
def ndpointer[ScalarT: np.generic](
    dtype: _DTypeLike[ScalarT],
    ndim: int | None = None,
    *,
    shape: _ShapeLike,
    flags: _FlagsKind | Iterable[_FlagsKind] | int | flagsobj | None = None,
) -> type[_concrete_ndptr[np.dtype[ScalarT]]]: ...
@overload
def ndpointer(
    dtype: DTypeLike | None,
    ndim: int | None = None,
    *,
    shape: _ShapeLike,
    flags: _FlagsKind | Iterable[_FlagsKind] | int | flagsobj | None = None,
) -> type[_concrete_ndptr[np.dtype]]: ...
@overload
def ndpointer[ScalarT: np.generic](
    dtype: _DTypeLike[ScalarT],
    ndim: int | None = None,
    shape: None = None,
    flags: _FlagsKind | Iterable[_FlagsKind] | int | flagsobj | None = None,
) -> type[_ndptr[np.dtype[ScalarT]]]: ...
@overload
def ndpointer(
    dtype: DTypeLike | None,
    ndim: int | None = None,
    shape: None = None,
    flags: _FlagsKind | Iterable[_FlagsKind] | int | flagsobj | None = None,
) -> type[_ndptr[np.dtype]]: ...

#
@overload  # bool
def as_ctypes_type(dtype: _BoolCodes | _DTypeLike[np.bool] | type[ct.c_bool]) -> type[ct.c_bool]: ...
@overload  # int8
def as_ctypes_type(dtype: _Int8Codes | _DTypeLike[np.int8] | type[ct.c_int8]) -> type[ct.c_int8]: ...
@overload  # int16
def as_ctypes_type(dtype: _Int16Codes | _DTypeLike[np.int16] | type[ct.c_int16]) -> type[ct.c_int16]: ...
@overload  # int32
def as_ctypes_type(dtype: _Int32Codes | _DTypeLike[np.int32] | type[ct.c_int32]) -> type[ct.c_int32]: ...
@overload  # int64
def as_ctypes_type(dtype: _Int64Codes | _DTypeLike[np.int64] | type[ct.c_int64]) -> type[ct.c_int64]: ...
@overload  # intc
def as_ctypes_type(dtype: _IntCCodes | type[ct.c_int]) -> type[ct.c_int]: ...
@overload  # long
def as_ctypes_type(dtype: _LongCodes | type[ct.c_long]) -> type[ct.c_long]: ...
@overload  # longlong
def as_ctypes_type(dtype: _LongLongCodes | type[ct.c_longlong]) -> type[ct.c_longlong]: ...
@overload  # intp
def as_ctypes_type(dtype: _IntPCodes | type[ct.c_ssize_t] | type[int]) -> type[ct.c_ssize_t]: ...
@overload  # uint8
def as_ctypes_type(dtype: _UInt8Codes | _DTypeLike[np.uint8] | type[ct.c_uint8]) -> type[ct.c_uint8]: ...
@overload  # uint16
def as_ctypes_type(dtype: _UInt16Codes | _DTypeLike[np.uint16] | type[ct.c_uint16]) -> type[ct.c_uint16]: ...
@overload  # uint32
def as_ctypes_type(dtype: _UInt32Codes | _DTypeLike[np.uint32] | type[ct.c_uint32]) -> type[ct.c_uint32]: ...
@overload  # uint64
def as_ctypes_type(dtype: _UInt64Codes | _DTypeLike[np.uint64] | type[ct.c_uint64]) -> type[ct.c_uint64]: ...
@overload  # uintc
def as_ctypes_type(dtype: _UIntCCodes | type[ct.c_uint]) -> type[ct.c_uint]: ...
@overload  # ulong
def as_ctypes_type(dtype: _ULongCodes | type[ct.c_ulong]) -> type[ct.c_ulong]: ...
@overload  # ulonglong
def as_ctypes_type(dtype: _ULongLongCodes | type[ct.c_ulonglong]) -> type[ct.c_ulonglong]: ...
@overload  # uintp
def as_ctypes_type(dtype: _UIntPCodes | type[ct.c_size_t]) -> type[ct.c_size_t]: ...
@overload  # float32
def as_ctypes_type(dtype: _Float32Codes | _DTypeLike[np.float32] | type[ct.c_float]) -> type[ct.c_float]: ...
@overload  # float64
def as_ctypes_type(dtype: _Float64Codes | _DTypeLike[np.float64] | type[float | ct.c_double]) -> type[ct.c_double]: ...
@overload  # longdouble
def as_ctypes_type(dtype: _LongDoubleCodes | _DTypeLike[np.longdouble] | type[ct.c_longdouble]) -> type[ct.c_longdouble]: ...
@overload  # void
def as_ctypes_type(dtype: _VoidDTypeLike) -> type[Any]: ...  # `ct.Union` or `ct.Structure`
@overload  # fallback
def as_ctypes_type(dtype: str) -> type[Any]: ...

#
@overload
def as_array(obj: ct._PointerLike, shape: Sequence[int]) -> NDArray[Any]: ...
@overload
def as_array[ScalarT: np.generic](obj: _ArrayLike[ScalarT], shape: _ShapeLike | None = None) -> NDArray[ScalarT]: ...
@overload
def as_array(obj: object, shape: _ShapeLike | None = None) -> NDArray[Any]: ...

#
@overload
def as_ctypes(obj: np.bool) -> ct.c_bool: ...
@overload
def as_ctypes(obj: np.int8) -> ct.c_int8: ...
@overload
def as_ctypes(obj: np.int16) -> ct.c_int16: ...
@overload
def as_ctypes(obj: np.int32) -> ct.c_int32: ...
@overload
def as_ctypes(obj: np.int64) -> ct.c_int64: ...
@overload
def as_ctypes(obj: np.uint8) -> ct.c_uint8: ...
@overload
def as_ctypes(obj: np.uint16) -> ct.c_uint16: ...
@overload
def as_ctypes(obj: np.uint32) -> ct.c_uint32: ...
@overload
def as_ctypes(obj: np.uint64) -> ct.c_uint64: ...
@overload
def as_ctypes(obj: np.float32) -> ct.c_float: ...
@overload
def as_ctypes(obj: np.float64) -> ct.c_double: ...
@overload
def as_ctypes(obj: np.longdouble) -> ct.c_longdouble: ...
@overload
def as_ctypes(obj: np.void) -> Any: ...  # `ct.Union` or `ct.Structure`
@overload
def as_ctypes(obj: NDArray[np.bool]) -> ct.Array[ct.c_bool]: ...
@overload
def as_ctypes(obj: NDArray[np.int8]) -> ct.Array[ct.c_int8]: ...
@overload
def as_ctypes(obj: NDArray[np.int16]) -> ct.Array[ct.c_int16]: ...
@overload
def as_ctypes(obj: NDArray[np.int32]) -> ct.Array[ct.c_int32]: ...
@overload
def as_ctypes(obj: NDArray[np.int64]) -> ct.Array[ct.c_int64]: ...
@overload
def as_ctypes(obj: NDArray[np.uint8]) -> ct.Array[ct.c_uint8]: ...
@overload
def as_ctypes(obj: NDArray[np.uint16]) -> ct.Array[ct.c_uint16]: ...
@overload
def as_ctypes(obj: NDArray[np.uint32]) -> ct.Array[ct.c_uint32]: ...
@overload
def as_ctypes(obj: NDArray[np.uint64]) -> ct.Array[ct.c_uint64]: ...
@overload
def as_ctypes(obj: NDArray[np.float32]) -> ct.Array[ct.c_float]: ...
@overload
def as_ctypes(obj: NDArray[np.float64]) -> ct.Array[ct.c_double]: ...
@overload
def as_ctypes(obj: NDArray[np.longdouble]) -> ct.Array[ct.c_longdouble]: ...
@overload
def as_ctypes(obj: NDArray[np.void]) -> ct.Array[Any]: ...  # `ct.Union` or `ct.Structure`
