from typing import Any, Final, Literal as L, overload

from numpy import complexfloating, floating, generic, integer
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ShapeLike,
)

__all__ = ["fftfreq", "fftshift", "ifftshift", "rfftfreq"]

###

integer_types: Final[tuple[type[int], type[integer]]] = ...

###

@overload
def fftshift[ScalarT: generic](x: _ArrayLike[ScalarT], axes: _ShapeLike | None = None) -> NDArray[ScalarT]: ...
@overload
def fftshift(x: ArrayLike, axes: _ShapeLike | None = None) -> NDArray[Any]: ...

#
@overload
def ifftshift[ScalarT: generic](x: _ArrayLike[ScalarT], axes: _ShapeLike | None = None) -> NDArray[ScalarT]: ...
@overload
def ifftshift(x: ArrayLike, axes: _ShapeLike | None = None) -> NDArray[Any]: ...

#
@overload
def fftfreq(n: int | integer, d: _ArrayLikeFloat_co = 1.0, device: L["cpu"] | None = None) -> NDArray[floating]: ...
@overload
def fftfreq(n: int | integer, d: _ArrayLikeComplex_co = 1.0, device: L["cpu"] | None = None) -> NDArray[complexfloating]: ...

#
@overload
def rfftfreq(n: int | integer, d: _ArrayLikeFloat_co = 1.0, device: L["cpu"] | None = None) -> NDArray[floating]: ...
@overload
def rfftfreq(n: int | integer, d: _ArrayLikeComplex_co = 1.0, device: L["cpu"] | None = None) -> NDArray[complexfloating]: ...
