from _typeshed import StrOrBytesPath, SupportsWrite
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    Protocol,
    Self,
    overload,
    override,
    type_check_only,
)
from typing_extensions import TypeVar

import numpy as np
from numpy import _OrderKACF, _SupportsFileMethods
from numpy._typing import DTypeLike, _AnyShape, _DTypeLike, _Shape

__all__ = ["memmap"]

_ShapeT_co = TypeVar("_ShapeT_co", bound=_Shape, default=_AnyShape, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype[Any], default=np.dtype[Any], covariant=True)

type _Mode = Literal["r", "c", "r+", "w+"]
type _ToMode = Literal[_Mode, "readonly", "copyonwrite", "readwrite", "write"]

@type_check_only
class _SupportsFileMethodsRW(SupportsWrite[bytes], _SupportsFileMethods, Protocol): ...

###

class memmap(np.ndarray[_ShapeT_co, _DTypeT_co]):
    __module__: Literal["numpy"] = "numpy"
    __array_priority__: ClassVar[float] = 100.0  # pyright: ignore[reportIncompatibleMethodOverride]

    filename: Final[str | None]
    offset: Final[int]
    mode: Final[_Mode]

    @overload
    def __new__[ScalarT: np.generic](
        cls,
        filename: StrOrBytesPath | _SupportsFileMethodsRW,
        dtype: _DTypeT_co,
        mode: _ToMode = "r+",
        offset: int = 0,
        shape: int | tuple[int, ...] | None = None,
        order: _OrderKACF = "C",
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        filename: StrOrBytesPath | _SupportsFileMethodsRW,
        dtype: type[np.uint8] = ...,
        mode: _ToMode = "r+",
        offset: int = 0,
        shape: int | tuple[int, ...] | None = None,
        order: _OrderKACF = "C",
    ) -> memmap[_AnyShape, np.dtype[np.uint8]]: ...
    @overload
    def __new__[ScalarT: np.generic](
        cls,
        filename: StrOrBytesPath | _SupportsFileMethodsRW,
        dtype: _DTypeLike[ScalarT],
        mode: _ToMode = "r+",
        offset: int = 0,
        shape: int | tuple[int, ...] | None = None,
        order: _OrderKACF = "C",
    ) -> memmap[_AnyShape, np.dtype[ScalarT]]: ...
    @overload
    def __new__(
        cls,
        filename: StrOrBytesPath | _SupportsFileMethodsRW,
        dtype: DTypeLike,
        mode: _ToMode = "r+",
        offset: int = 0,
        shape: int | tuple[int, ...] | None = None,
        order: _OrderKACF = "C",
    ) -> memmap: ...

    #
    @override
    def __array_finalize__(self, obj: object, /) -> None: ...
    @override
    def __array_wrap__(  # type: ignore[override]
        self,
        /,
        array: memmap[_ShapeT_co, _DTypeT_co],
        context: tuple[np.ufunc, tuple[Any, ...], int] | None = None,
        return_scalar: bool = False,
    ) -> Any: ...

    #
    def flush(self) -> None: ...
