from collections.abc import Iterable
from typing import Any, Never, overload

import numpy as np
from numpy._typing import ArrayLike, NDArray, _AnyShape, _ArrayLike, _Shape, _ShapeLike

__all__ = ["broadcast_to", "broadcast_arrays", "broadcast_shapes"]

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]
type _ArrayMax2D[ScalarT: np.generic] = np.ndarray[tuple[int] | tuple[int, int], np.dtype[ScalarT]]

# workaround for mypy and pyright not following the typing spec for overloads
type _ShapeNoD = tuple[Never, Never, Never, Never]

class DummyArray:
    __array_interface__: dict[str, Any]
    base: NDArray[Any] | None
    def __init__(
        self,
        interface: dict[str, Any],
        base: NDArray[Any] | None = None,
    ) -> None: ...

@overload
def as_strided[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    shape: Iterable[int] | None = None,
    strides: Iterable[int] | None = None,
    subok: bool = False,
    writeable: bool = True,
    *,
    check_bounds: bool | None = None
) -> NDArray[ScalarT]: ...
@overload
def as_strided(
    x: ArrayLike,
    shape: Iterable[int] | None = None,
    strides: Iterable[int] | None = None,
    subok: bool = False,
    writeable: bool = True,
    *,
    check_bounds: bool | None = None
) -> NDArray[Any]: ...

@overload
def sliding_window_view[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    window_shape: int | Iterable[int],
    axis: int | tuple[int, ...] | None = None,
    *,
    subok: bool = False,
    writeable: bool = False,
) -> NDArray[ScalarT]: ...
@overload
def sliding_window_view(
    x: ArrayLike,
    window_shape: int | Iterable[int],
    axis: int | tuple[int, ...] | None = None,
    *,
    subok: bool = False,
    writeable: bool = False,
) -> NDArray[Any]: ...

#
@overload  # known dtype, 1d shape
def broadcast_to[ScalarT: np.generic](
    array: _ArrayLike[ScalarT],
    shape: int,
    subok: bool = False,
) -> np.ndarray[tuple[int], np.dtype[ScalarT]]: ...
@overload  # known dtype, known shape
def broadcast_to[ScalarT: np.generic, ShapeT: tuple[int, ...]](
    array: _ArrayLike[ScalarT],
    shape: ShapeT,
    subok: bool = False,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # known dtype, unknown shape
def broadcast_to[ScalarT: np.generic](
    array: _ArrayLike[ScalarT],
    shape: Iterable[int],
    subok: bool = False,
) -> NDArray[ScalarT]: ...
@overload  # unknown dtype, 1d shape
def broadcast_to(
    array: ArrayLike,
    shape: int,
    subok: bool = False,
) -> np.ndarray[tuple[int], np.dtype[Any]]: ...
@overload  # unknown dtype, known shape
def broadcast_to[ShapeT: tuple[int, ...]](
    array: ArrayLike,
    shape: ShapeT,
    subok: bool = False,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # unknown dtype, unknown shape
def broadcast_to(
    array: ArrayLike,
    shape: Iterable[int],
    subok: bool = False,
) -> NDArray[Any]: ...

#
def broadcast_shapes(*args: _ShapeLike) -> _AnyShape: ...

#
@overload  # ()
def broadcast_arrays(*, subok: bool = False) -> tuple[()]: ...
@overload  # Nd T
def broadcast_arrays[ShapeT: _Shape, DTypeT: np.dtype](
    a0: np.ndarray[ShapeT, DTypeT],
    /,
    *,
    subok: bool = False,
) -> tuple[np.ndarray[ShapeT, DTypeT]]: ...
@overload  # ?d
def broadcast_arrays(
    a0: ArrayLike,
    /,
    *,
    subok: bool = False,
) -> tuple[NDArray[Any]]: ...
@overload  # ?d T, ?d T  (workaround)
def broadcast_arrays[DTypeT0: np.dtype, DTypeT1: np.dtype](
    a0: np.ndarray[_ShapeNoD, DTypeT0],
    a1: np.ndarray[_AnyShape, DTypeT1],
    /,
    *,
    subok: bool = False,
) -> tuple[np.ndarray[_AnyShape, DTypeT0], np.ndarray[_AnyShape, DTypeT1]]: ...
@overload  # ?d T, ?d T  (workaround)
def broadcast_arrays[DTypeT0: np.dtype, DTypeT1: np.dtype](
    a0: np.ndarray[_AnyShape, DTypeT0],
    a1: np.ndarray[_ShapeNoD, DTypeT1],
    /,
    *,
    subok: bool = False,
) -> tuple[np.ndarray[_AnyShape, DTypeT0], np.ndarray[_AnyShape, DTypeT1]]: ...
@overload  # 1d T, 1d T
def broadcast_arrays[ScalarT0: np.generic, ScalarT1: np.generic](
    a0: _Array1D[ScalarT0],
    a1: _Array1D[ScalarT1],
    /,
    *,
    subok: bool = False,
) -> tuple[_Array1D[ScalarT0], _Array1D[ScalarT1]]: ...
@overload  # 1d T, 2d T
def broadcast_arrays[ScalarT0: np.generic, ScalarT1: np.generic](
    a0: _Array1D[ScalarT0],
    a1: _Array2D[ScalarT1],
    /,
    *,
    subok: bool = False,
) -> tuple[_Array2D[ScalarT0], _Array2D[ScalarT1]]: ...
@overload  # 2d T, <=2d T
def broadcast_arrays[ScalarT0: np.generic, ScalarT1: np.generic](
    a0: _Array2D[ScalarT0],
    a1: _ArrayMax2D[ScalarT1],
    /,
    *,
    subok: bool = False,
) -> tuple[_Array2D[ScalarT0], _Array2D[ScalarT1]]: ...
@overload  # <=2d T, 3d T
def broadcast_arrays[ScalarT0: np.generic, ScalarT1: np.generic](
    a0: _ArrayMax2D[ScalarT0],
    a1: _Array3D[ScalarT1],
    /,
    *,
    subok: bool = False,
) -> tuple[_Array3D[ScalarT0], _Array3D[ScalarT1]]: ...
@overload  # 3d T, <=3d T
def broadcast_arrays[ScalarT0: np.generic, ScalarT1: np.generic](
    a0: _Array3D[ScalarT0],
    a1: np.ndarray[tuple[int] | tuple[int, int] | tuple[int, int, int], np.dtype[ScalarT1]],
    /,
    *,
    subok: bool = False,
) -> tuple[_Array3D[ScalarT0], _Array3D[ScalarT1]]: ...
@overload  # Nd T, 0d T
def broadcast_arrays[ShapeT: _Shape, DTypeT: np.dtype, ScalarT: np.generic](
    a0: np.ndarray[ShapeT, DTypeT],
    a1: ScalarT,
    /,
    *,
    subok: bool = False,
) -> tuple[np.ndarray[ShapeT, DTypeT], np.ndarray[ShapeT, np.dtype[ScalarT]]]: ...
@overload  # 0d T, Nd T
def broadcast_arrays[ScalarT: np.generic, ShapeT: _Shape, DTypeT: np.dtype](
    a0: ScalarT,
    a1: np.ndarray[ShapeT, DTypeT],
    /,
    *,
    subok: bool = False,
) -> tuple[np.ndarray[ShapeT, np.dtype[ScalarT]], np.ndarray[ShapeT, DTypeT]]: ...
@overload  # Nd T, 0d _
def broadcast_arrays[ShapeT: _Shape, DTypeT: np.dtype](
    a0: np.ndarray[ShapeT, DTypeT],
    a1: complex,
    /,
    *,
    subok: bool = False,
) -> tuple[np.ndarray[ShapeT, DTypeT], np.ndarray[ShapeT, np.dtype[Any]]]: ...
@overload  # 0d _, Nd T
def broadcast_arrays[ShapeT: _Shape, DTypeT: np.dtype](
    a0: complex,
    a1: np.ndarray[ShapeT, DTypeT],
    /,
    *,
    subok: bool = False,
) -> tuple[np.ndarray[ShapeT, np.dtype[Any]], np.ndarray[ShapeT, DTypeT]]: ...
@overload  # ?d T, ?d T
def broadcast_arrays[ScalarT0: np.generic, ScalarT1: np.generic](
    a0: _ArrayLike[ScalarT0],
    a1: _ArrayLike[ScalarT1],
    /,
    *,
    subok: bool = False,
) -> tuple[NDArray[ScalarT0], NDArray[ScalarT1]]: ...
@overload  # ?d, ?d
def broadcast_arrays(
    a0: ArrayLike,
    a1: ArrayLike,
    /,
    *,
    subok: bool = False,
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload  # ?d T, *?d T
def broadcast_arrays[ScalarT: np.generic](
    a0: _ArrayLike[ScalarT],
    a1: _ArrayLike[ScalarT],
    /,
    *ai: _ArrayLike[ScalarT],
    subok: bool = False,
) -> tuple[NDArray[ScalarT], ...]: ...
@overload  # ?d, *?d
def broadcast_arrays(
    a0: ArrayLike,
    a1: ArrayLike,
    /,
    *ai: ArrayLike,
    subok: bool = False,
) -> tuple[NDArray[Any], ...]: ...

# used internally by `lib._function_base_impl._parse_input_dimensions`
def _broadcast_shape(*args: ArrayLike) -> _AnyShape: ...
