# pyright: reportIncompatibleMethodOverride=false

from collections.abc import Generator
from types import EllipsisType
from typing import Any, Final, overload
from typing_extensions import TypeVar

import numpy as np
from numpy._typing import _AnyShape, _Shape

__all__ = ["Arrayterator"]

# Type parameter default syntax (PEP 696) requires Python 3.13+
_ShapeT_co = TypeVar("_ShapeT_co", bound=_Shape, default=_AnyShape, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype, default=np.dtype, covariant=True)

type _AnyIndex = EllipsisType | int | slice | tuple[EllipsisType | int | slice, ...]

# NOTE: In reality `Arrayterator` does not actually inherit from `ndarray`,
# but its ``__getattr__` method does wrap around the former and thus has
# access to all its methods

class Arrayterator(np.ndarray[_ShapeT_co, _DTypeT_co]):
    var: np.ndarray[_ShapeT_co, _DTypeT_co]  # type: ignore[assignment]
    buf_size: Final[int | None]
    start: Final[list[int]]
    stop: Final[list[int]]
    step: Final[list[int]]

    @property  # type: ignore[misc]
    def shape(self) -> _ShapeT_co: ...
    @property
    def flat[ScalarT: np.generic](self: Arrayterator[Any, np.dtype[ScalarT]]) -> Generator[ScalarT]: ...  # type: ignore[override]

    #
    def __init__(self, /, var: np.ndarray[_ShapeT_co, _DTypeT_co], buf_size: int | None = None) -> None: ...
    def __getitem__(self, index: _AnyIndex, /) -> Arrayterator[_AnyShape, _DTypeT_co]: ...  # type: ignore[override]
    def __iter__(self) -> Generator[np.ndarray[_AnyShape, _DTypeT_co]]: ...

    #
    @overload
    def __array__(self, /, dtype: None = None, copy: bool | None = None) -> np.ndarray[_ShapeT_co, _DTypeT_co]: ...
    @overload
    def __array__[DTypeT: np.dtype](self, /, dtype: DTypeT, copy: bool | None = None) -> np.ndarray[_ShapeT_co, DTypeT]: ...
