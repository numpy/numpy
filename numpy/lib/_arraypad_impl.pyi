from collections.abc import Sequence
from typing import Any, Literal as L, Protocol, overload, type_check_only

import numpy as np
from numpy._typing import ArrayLike, NDArray, _ArrayLike, _ArrayLikeInt, _Shape

__all__ = ["pad"]

@type_check_only
class _ModeFunc(Protocol):
    def __call__(
        self,
        vector: NDArray[Any],
        iaxis_pad_width: tuple[int, int],
        iaxis: int,
        kwargs: dict[str, Any],
        /,
    ) -> None: ...

# grouped by the keyword argument they accept
type _ModeStatLength = L["maximum", "mean", "median", "minimum"]
type _ModeReflectType = L["reflect", "symmetric"]
type _ModeNoKwargs = L["edge", "wrap", "empty"] | _ModeFunc
type _Mode = L["constant", "linear_ramp"] | _ModeStatLength | _ModeReflectType | _ModeNoKwargs

type _PadWidth = (
    _ArrayLikeInt
    | dict[int, int]
    | dict[int, tuple[int, int]]
    | dict[int, int | tuple[int, int]]
)

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

###

# the keyword arguments are only checked against `mode` for non-list input
@overload  # Nd, mode="constant"
def pad[ShapeT: _Shape, DTypeT: np.dtype](
    array: np.ndarray[ShapeT, DTypeT],
    pad_width: _PadWidth,
    mode: L["constant"] = "constant",
    *,
    constant_values: ArrayLike = 0,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd, mode="linear_ramp"
def pad[ShapeT: _Shape, DTypeT: np.dtype](
    array: np.ndarray[ShapeT, DTypeT],
    pad_width: _PadWidth,
    mode: L["linear_ramp"],
    *,
    end_values: ArrayLike = 0,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd, mode="maximum" | "mean" | "median" | "minimum"
def pad[ShapeT: _Shape, DTypeT: np.dtype](
    array: np.ndarray[ShapeT, DTypeT],
    pad_width: _PadWidth,
    mode: _ModeStatLength,
    *,
    stat_length: _ArrayLikeInt | None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd, mode="reflect"
def pad[ShapeT: _Shape, DTypeT: np.dtype](
    array: np.ndarray[ShapeT, DTypeT],
    pad_width: _PadWidth,
    mode: _ModeReflectType,
    *,
    reflect_type: L["odd", "even"] = "even",
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd, mode=<other>
def pad[ShapeT: _Shape, DTypeT: np.dtype](
    array: np.ndarray[ShapeT, DTypeT],
    pad_width: _PadWidth,
    mode: _ModeNoKwargs,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # 1d bool
def pad(
    array: list[bool],
    pad_width: _PadWidth,
    mode: _Mode = "constant",
    *,
    stat_length: _ArrayLikeInt | None = None,
    constant_values: ArrayLike = 0,
    end_values: ArrayLike = 0,
    reflect_type: L["odd", "even"] = "even",
) -> _Array1D[np.bool]: ...
@overload  # 1d int
def pad(
    array: list[int],
    pad_width: _PadWidth,
    mode: _Mode = "constant",
    *,
    stat_length: _ArrayLikeInt | None = None,
    constant_values: ArrayLike = 0,
    end_values: ArrayLike = 0,
    reflect_type: L["odd", "even"] = "even",
) -> _Array1D[np.int_]: ...
@overload  # 1d float
def pad(
    array: list[float],
    pad_width: _PadWidth,
    mode: _Mode = "constant",
    *,
    stat_length: _ArrayLikeInt | None = None,
    constant_values: ArrayLike = 0,
    end_values: ArrayLike = 0,
    reflect_type: L["odd", "even"] = "even",
) -> _Array1D[np.float64]: ...
@overload  # 1d complex
def pad(
    array: list[complex],
    pad_width: _PadWidth,
    mode: _Mode = "constant",
    *,
    stat_length: _ArrayLikeInt | None = None,
    constant_values: ArrayLike = 0,
    end_values: ArrayLike = 0,
    reflect_type: L["odd", "even"] = "even",
) -> _Array1D[np.complex128]: ...
@overload  # 2d bool
def pad(
    array: Sequence[list[bool]],
    pad_width: _PadWidth,
    mode: _Mode = "constant",
    *,
    stat_length: _ArrayLikeInt | None = None,
    constant_values: ArrayLike = 0,
    end_values: ArrayLike = 0,
    reflect_type: L["odd", "even"] = "even",
) -> _Array2D[np.bool]: ...
@overload  # 2d int
def pad(
    array: Sequence[list[int]],
    pad_width: _PadWidth,
    mode: _Mode = "constant",
    *,
    stat_length: _ArrayLikeInt | None = None,
    constant_values: ArrayLike = 0,
    end_values: ArrayLike = 0,
    reflect_type: L["odd", "even"] = "even",
) -> _Array2D[np.int_]: ...
@overload  # 2d float
def pad(
    array: Sequence[list[float]],
    pad_width: _PadWidth,
    mode: _Mode = "constant",
    *,
    stat_length: _ArrayLikeInt | None = None,
    constant_values: ArrayLike = 0,
    end_values: ArrayLike = 0,
    reflect_type: L["odd", "even"] = "even",
) -> _Array2D[np.float64]: ...
@overload  # 2d complex
def pad(
    array: Sequence[list[complex]],
    pad_width: _PadWidth,
    mode: _Mode = "constant",
    *,
    stat_length: _ArrayLikeInt | None = None,
    constant_values: ArrayLike = 0,
    end_values: ArrayLike = 0,
    reflect_type: L["odd", "even"] = "even",
) -> _Array2D[np.complex128]: ...
@overload  # Nd T, mode="constant"
def pad[ScalarT: np.generic](
    array: _ArrayLike[ScalarT],
    pad_width: _PadWidth,
    mode: L["constant"] = "constant",
    *,
    constant_values: ArrayLike = 0,
) -> NDArray[ScalarT]: ...
@overload  # Nd T, mode="linear_ramp"
def pad[ScalarT: np.generic](
    array: _ArrayLike[ScalarT],
    pad_width: _PadWidth,
    mode: L["linear_ramp"],
    *,
    end_values: ArrayLike = 0,
) -> NDArray[ScalarT]: ...
@overload  # Nd T, mode="maximum" | "mean" | "median" | "minimum"
def pad[ScalarT: np.generic](
    array: _ArrayLike[ScalarT],
    pad_width: _PadWidth,
    mode: _ModeStatLength,
    *,
    stat_length: _ArrayLikeInt | None = None,
) -> NDArray[ScalarT]: ...
@overload  # Nd T, mode="reflect"
def pad[ScalarT: np.generic](
    array: _ArrayLike[ScalarT],
    pad_width: _PadWidth,
    mode: _ModeReflectType,
    *,
    reflect_type: L["odd", "even"] = "even",
) -> NDArray[ScalarT]: ...
@overload  # Nd T, mode=<other>
def pad[ScalarT: np.generic](
    array: _ArrayLike[ScalarT],
    pad_width: _PadWidth,
    mode: _ModeNoKwargs,
) -> NDArray[ScalarT]: ...
@overload  # fallback, mode="constant"
def pad(
    array: ArrayLike,
    pad_width: _PadWidth,
    mode: L["constant"] = "constant",
    *,
    constant_values: ArrayLike = 0,
) -> NDArray[Any]: ...
@overload  # fallback, mode="linear_ramp"
def pad(
    array: ArrayLike,
    pad_width: _PadWidth,
    mode: L["linear_ramp"],
    *,
    end_values: ArrayLike = 0,
) -> NDArray[Any]: ...
@overload  # fallback, mode="maximum" | "mean" | "median" | "minimum"
def pad(
    array: ArrayLike,
    pad_width: _PadWidth,
    mode: _ModeStatLength,
    *,
    stat_length: _ArrayLikeInt | None = None,
) -> NDArray[Any]: ...
@overload  # fallback, mode="reflect"
def pad(
    array: ArrayLike,
    pad_width: _PadWidth,
    mode: _ModeReflectType,
    *,
    reflect_type: L["odd", "even"] = "even",
) -> NDArray[Any]: ...
@overload  # fallback, mode=<other>
def pad(array: ArrayLike, pad_width: _PadWidth, mode: _ModeNoKwargs) -> NDArray[Any]: ...
