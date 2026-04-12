from collections.abc import Sequence
from typing import Literal as L, overload

import numpy as np
from numpy import complex128, float64
from numpy._typing import _ArrayLike, _ArrayLikeNumber_co, _Shape
from numpy._typing._array_like import _DualArrayLike
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "hfft",
    "ihfft",
    "rfftn",
    "irfftn",
    "rfft2",
    "irfft2",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
]

type _NormKind = L["backward", "ortho", "forward"] | None

###

@overload  # Nd complexfloating
def fft[ShapeT: _Shape, DTypeT: np.dtype[np.complexfloating]](
    a: np.ndarray[ShapeT, DTypeT],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd float64 | +integer
def fft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float64 | np.integer | np.bool]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd float32 | float16
def fft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float32 | np.float16]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex64]]: ...
@overload  # Nd longdouble
def fft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.longdouble]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.clongdouble]]: ...
@overload  # 1d +complex
def fft(
    a: Sequence[complex],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]: ...
@overload  # 2d +complex
def fft(
    a: Sequence[Sequence[complex]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...
@overload  # ?d complexfloating
def fft[ScalarT: np.complexfloating](
    a: _ArrayLike[ScalarT],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d +complex
def fft(
    a: _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], complex],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def fft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def fft[ArrayT: NDArray[np.complexfloating]](
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

#
def ifft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: NDArray[complex128] | None = None,
) -> NDArray[complex128]: ...

def rfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: NDArray[complex128] | None = None,
) -> NDArray[complex128]: ...

def irfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: NDArray[float64] | None = None,
) -> NDArray[float64]: ...

# Input array must be compatible with `conjugate`
def hfft(
    a: _ArrayLikeNumber_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: NDArray[float64] | None = None,
) -> NDArray[float64]: ...

def ihfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: NDArray[complex128] | None = None,
) -> NDArray[complex128]: ...

def fftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: NDArray[complex128] | None = None,
) -> NDArray[complex128]: ...

def ifftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: NDArray[complex128] | None = None,
) -> NDArray[complex128]: ...

def rfftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: NDArray[complex128] | None = None,
) -> NDArray[complex128]: ...

def irfftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: NDArray[float64] | None = None,
) -> NDArray[float64]: ...

def fft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: NDArray[complex128] | None = None,
) -> NDArray[complex128]: ...

def ifft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: NDArray[complex128] | None = None,
) -> NDArray[complex128]: ...

def rfft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: NDArray[complex128] | None = None,
) -> NDArray[complex128]: ...

def irfft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: NDArray[float64] | None = None,
) -> NDArray[float64]: ...
