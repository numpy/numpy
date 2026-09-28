from collections.abc import Iterable, Sequence
from typing import Any, Never, SupportsIndex, overload

import numpy as np
from numpy._typing import ArrayLike, NDArray, _AnyShape, _ArrayLike, _Shape, _ShapeLike

__all__ = ["broadcast_to", "broadcast_arrays", "broadcast_shapes"]

type _0D = tuple[()]
type _1D = tuple[int]
type _2D = tuple[int, int]
type _3D = tuple[int, int, int]
type _4D = tuple[int, int, int, int]

type _Array1D[ScalarT: np.generic] = np.ndarray[_1D, np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[_2D, np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[_3D, np.dtype[ScalarT]]
type _ArrayMax2D[ScalarT: np.generic] = np.ndarray[_1D | _2D, np.dtype[ScalarT]]

type _ToShape1D = int | np.integer | _1D
type _ToShape2D = _ToShape1D | _2D

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

#
@overload  # Nd T, None
def as_strided[ShapeT: _Shape, DTypeT: np.dtype](
    x: np.ndarray[ShapeT, DTypeT],
    shape: None = None,
    strides: Iterable[int] | None = None,
    subok: bool = False,
    writeable: bool = True,
    *,
    check_bounds: bool | None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # ?d T, Nd
def as_strided[ScalarT: np.generic, ShapeT: _Shape](
    x: _ArrayLike[ScalarT],
    shape: ShapeT,
    strides: Iterable[int] | None = None,
    subok: bool = False,
    writeable: bool = True,
    *,
    check_bounds: bool | None = None,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # ?d T, ?d
def as_strided[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    shape: Iterable[int] | None = None,
    strides: Iterable[int] | None = None,
    subok: bool = False,
    writeable: bool = True,
    *,
    check_bounds: bool | None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d, Nd
def as_strided[ShapeT: _Shape](
    x: ArrayLike,
    shape: ShapeT,
    strides: Iterable[int] | None = None,
    subok: bool = False,
    writeable: bool = True,
    *,
    check_bounds: bool | None = None,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # ?d, ?d  (fallback)
def as_strided(
    x: ArrayLike,
    shape: Iterable[int] | None = None,
    strides: Iterable[int] | None = None,
    subok: bool = False,
    writeable: bool = True,
    *,
    check_bounds: bool | None = None,
) -> NDArray[Any]: ...

#
@overload  # ?d T, ?d  (workaround)
def sliding_window_view[DTypeT: np.dtype](
    x: np.ndarray[_ShapeNoD, DTypeT],
    window_shape: int | Iterable[int],
    axis: int | tuple[int, ...] | None = None,
    *,
    subok: bool = False,
    writeable: bool = False,
) -> np.ndarray[_AnyShape, DTypeT]: ...
@overload  # 1d T, 1d
def sliding_window_view[DTypeT: np.dtype](
    x: np.ndarray[tuple[int], DTypeT],
    window_shape: int | tuple[int],
    axis: int | tuple[int] | None = None,
    *,
    subok: bool = False,
    writeable: bool = False,
) -> np.ndarray[tuple[int, int], DTypeT]: ...
@overload  # 2d T, 1d, axis=<given>
def sliding_window_view[DTypeT: np.dtype](
    x: np.ndarray[tuple[int, int], DTypeT],
    window_shape: int | tuple[int],
    axis: int | tuple[int],
    *,
    subok: bool = False,
    writeable: bool = False,
) -> np.ndarray[tuple[int, int, int], DTypeT]: ...
@overload  # 2d T, 2d
def sliding_window_view[DTypeT: np.dtype](
    x: np.ndarray[tuple[int, int], DTypeT],
    window_shape: tuple[int, int],
    axis: tuple[int, int] | None = None,
    *,
    subok: bool = False,
    writeable: bool = False,
) -> np.ndarray[tuple[int, int, int, int], DTypeT]: ...
@overload  # 3d T, 1d, axis=<given>
def sliding_window_view[DTypeT: np.dtype](
    x: np.ndarray[tuple[int, int, int], DTypeT],
    window_shape: int | tuple[int],
    axis: int | tuple[int],
    *,
    subok: bool = False,
    writeable: bool = False,
) -> np.ndarray[tuple[int, int, int, int], DTypeT]: ...
@overload  # ?d T, ?d
def sliding_window_view[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    window_shape: int | Iterable[int],
    axis: int | tuple[int, ...] | None = None,
    *,
    subok: bool = False,
    writeable: bool = False,
) -> NDArray[ScalarT]: ...
@overload  # ?d, ?d  (fallback)
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
def broadcast_to[ScalarT: np.generic, ShapeT: (_0D, _1D, _2D, _3D, _4D)](
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
def broadcast_to[ShapeT: (_0D, _1D, _2D, _3D, _4D)](
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
@overload  # ()
def broadcast_shapes() -> tuple[()]: ...
@overload  # 1d
def broadcast_shapes(a0: int | np.integer, /) -> tuple[int]: ...
@overload  # Nd
def broadcast_shapes[ShapeT: _Shape](a0: ShapeT, /) -> ShapeT: ...
@overload  # ?d
def broadcast_shapes(a0: Sequence[SupportsIndex], /) -> _AnyShape: ...
@overload  # ?d, ?d  (workaround)
def broadcast_shapes(a0: tuple[int, int, int, int], a1: _ShapeLike, /) -> _AnyShape: ...
@overload  # ?d, ?d  (workaround)
def broadcast_shapes(a0: _ShapeLike, a1: tuple[int, int, int, int], /) -> _AnyShape: ...
@overload  # 1d, 1d
def broadcast_shapes(a0: _ToShape1D, a1: _ToShape1D, /) -> tuple[int]: ...
@overload  # 1d, 2d
def broadcast_shapes(a0: _ToShape1D, a1: tuple[int, int], /) -> tuple[int, int]: ...
@overload  # 2d, <=2d
def broadcast_shapes(a0: tuple[int, int], a1: _ToShape2D, /) -> tuple[int, int]: ...
@overload  # <=2d, 3d
def broadcast_shapes(a0: _ToShape2D, a1: tuple[int, int, int], /) -> tuple[int, int, int]: ...
@overload  # 3d, <=3d
def broadcast_shapes(a0: tuple[int, int, int], a1: _ToShape2D | tuple[int, int, int], /) -> tuple[int, int, int]: ...
@overload  # ?d, *?d  (fallback)
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
