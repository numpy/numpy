# NOTE: Numpy's mypy plugin is used for importing the correct
# platform-specific `ctypes._SimpleCData[int]` sub-type
import ctypes
from _typeshed import StrOrBytesPath
from collections.abc import Iterable, Sequence
from ctypes import c_int64 as _c_intp
from typing import Any, ClassVar, Literal as L, overload

import numpy as np
from numpy import (
    byte,
    double,
    intc,
    long,
    longdouble,
    longlong,
    short,
    single,
    ubyte,
    uintc,
    ulong,
    ulonglong,
    ushort,
    void,
)
from numpy._core._internal import _ctypes
from numpy._core.multiarray import flagsobj
from numpy._typing import (
    DTypeLike,
    NDArray,
    _AnyShape,
    _ArrayLike,
    _BoolCodes,
    _ByteCodes,
    _DoubleCodes,
    _DTypeLike,
    _IntCCodes,
    _LongCodes,
    _LongDoubleCodes,
    _LongLongCodes,
    _ShapeLike,
    _ShortCodes,
    _SingleCodes,
    _UByteCodes,
    _UIntCCodes,
    _ULongCodes,
    _ULongLongCodes,
    _UShortCodes,
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
class _ndptr[OptionalDTypeT: np.dtype | None](ctypes.c_void_p):
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

def load_library(libname: StrOrBytesPath, loader_path: StrOrBytesPath) -> ctypes.CDLL: ...

c_intp = _c_intp

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

@overload
def as_ctypes_type(dtype: _BoolCodes | _DTypeLike[np.bool] | type[ctypes.c_bool]) -> type[ctypes.c_bool]: ...
@overload
def as_ctypes_type(dtype: _ByteCodes | _DTypeLike[byte] | type[ctypes.c_byte]) -> type[ctypes.c_byte]: ...
@overload
def as_ctypes_type(dtype: _ShortCodes | _DTypeLike[short] | type[ctypes.c_short]) -> type[ctypes.c_short]: ...
@overload
def as_ctypes_type(dtype: _IntCCodes | _DTypeLike[intc] | type[ctypes.c_int]) -> type[ctypes.c_int]: ...
@overload
def as_ctypes_type(dtype: _LongCodes | _DTypeLike[long] | type[ctypes.c_long]) -> type[ctypes.c_long]: ...
@overload
def as_ctypes_type(dtype: type[int]) -> type[c_intp]: ...
@overload
def as_ctypes_type(dtype: _LongLongCodes | _DTypeLike[longlong] | type[ctypes.c_longlong]) -> type[ctypes.c_longlong]: ...
@overload
def as_ctypes_type(dtype: _UByteCodes | _DTypeLike[ubyte] | type[ctypes.c_ubyte]) -> type[ctypes.c_ubyte]: ...
@overload
def as_ctypes_type(dtype: _UShortCodes | _DTypeLike[ushort] | type[ctypes.c_ushort]) -> type[ctypes.c_ushort]: ...
@overload
def as_ctypes_type(dtype: _UIntCCodes | _DTypeLike[uintc] | type[ctypes.c_uint]) -> type[ctypes.c_uint]: ...
@overload
def as_ctypes_type(dtype: _ULongCodes | _DTypeLike[ulong] | type[ctypes.c_ulong]) -> type[ctypes.c_ulong]: ...
@overload
def as_ctypes_type(dtype: _ULongLongCodes | _DTypeLike[ulonglong] | type[ctypes.c_ulonglong]) -> type[ctypes.c_ulonglong]: ...
@overload
def as_ctypes_type(dtype: _SingleCodes | _DTypeLike[single] | type[ctypes.c_float]) -> type[ctypes.c_float]: ...
@overload
def as_ctypes_type(dtype: _DoubleCodes | _DTypeLike[double] | type[float | ctypes.c_double]) -> type[ctypes.c_double]: ...
@overload
def as_ctypes_type(dtype: _LongDoubleCodes | _DTypeLike[longdouble] | type[ctypes.c_longdouble]) -> type[ctypes.c_longdouble]: ...
@overload
def as_ctypes_type(dtype: _VoidDTypeLike) -> type[Any]: ...  # `ctypes.Union` or `ctypes.Structure`
@overload
def as_ctypes_type(dtype: str) -> type[Any]: ...

@overload
def as_array(obj: ctypes._PointerLike, shape: Sequence[int]) -> NDArray[Any]: ...
@overload
def as_array[ScalarT: np.generic](obj: _ArrayLike[ScalarT], shape: _ShapeLike | None = None) -> NDArray[ScalarT]: ...
@overload
def as_array(obj: object, shape: _ShapeLike | None = None) -> NDArray[Any]: ...

@overload
def as_ctypes(obj: np.bool) -> ctypes.c_bool: ...
@overload
def as_ctypes(obj: byte) -> ctypes.c_byte: ...
@overload
def as_ctypes(obj: short) -> ctypes.c_short: ...
@overload
def as_ctypes(obj: intc) -> ctypes.c_int: ...
@overload
def as_ctypes(obj: long) -> ctypes.c_long: ...
@overload
def as_ctypes(obj: longlong) -> ctypes.c_longlong: ...  # type: ignore[overload-cannot-match]
@overload
def as_ctypes(obj: ubyte) -> ctypes.c_ubyte: ...
@overload
def as_ctypes(obj: ushort) -> ctypes.c_ushort: ...
@overload
def as_ctypes(obj: uintc) -> ctypes.c_uint: ...
@overload
def as_ctypes(obj: ulong) -> ctypes.c_ulong: ...
@overload
def as_ctypes(obj: ulonglong) -> ctypes.c_ulonglong: ...  # type: ignore[overload-cannot-match]
@overload
def as_ctypes(obj: single) -> ctypes.c_float: ...
@overload
def as_ctypes(obj: double) -> ctypes.c_double: ...
@overload
def as_ctypes(obj: longdouble) -> ctypes.c_longdouble: ...
@overload
def as_ctypes(obj: void) -> Any: ...  # `ctypes.Union` or `ctypes.Structure`
@overload
def as_ctypes(obj: NDArray[np.bool]) -> ctypes.Array[ctypes.c_bool]: ...
@overload
def as_ctypes(obj: NDArray[byte]) -> ctypes.Array[ctypes.c_byte]: ...
@overload
def as_ctypes(obj: NDArray[short]) -> ctypes.Array[ctypes.c_short]: ...
@overload
def as_ctypes(obj: NDArray[intc]) -> ctypes.Array[ctypes.c_int]: ...
@overload
def as_ctypes(obj: NDArray[long]) -> ctypes.Array[ctypes.c_long]: ...
@overload
def as_ctypes(obj: NDArray[longlong]) -> ctypes.Array[ctypes.c_longlong]: ...  # type: ignore[overload-cannot-match]
@overload
def as_ctypes(obj: NDArray[ubyte]) -> ctypes.Array[ctypes.c_ubyte]: ...
@overload
def as_ctypes(obj: NDArray[ushort]) -> ctypes.Array[ctypes.c_ushort]: ...
@overload
def as_ctypes(obj: NDArray[uintc]) -> ctypes.Array[ctypes.c_uint]: ...
@overload
def as_ctypes(obj: NDArray[ulong]) -> ctypes.Array[ctypes.c_ulong]: ...
@overload
def as_ctypes(obj: NDArray[ulonglong]) -> ctypes.Array[ctypes.c_ulonglong]: ...  # type: ignore[overload-cannot-match]
@overload
def as_ctypes(obj: NDArray[single]) -> ctypes.Array[ctypes.c_float]: ...
@overload
def as_ctypes(obj: NDArray[double]) -> ctypes.Array[ctypes.c_double]: ...
@overload
def as_ctypes(obj: NDArray[longdouble]) -> ctypes.Array[ctypes.c_longdouble]: ...
@overload
def as_ctypes(obj: NDArray[void]) -> ctypes.Array[Any]: ...  # `ctypes.Union` or `ctypes.Structure`
