from collections.abc import Sequence
from typing import Literal as L, overload

import numpy as np
from numpy._typing import (
    NDArray,
    _ArrayLike,
    _ArrayLikeFloat_co,
    _ArrayLikeNumber_co,
    _Shape,
)
from numpy._typing._array_like import _DualArrayLike

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

# keep in sync with `ifft`
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
    a: _ArrayLikeNumber_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def fft[ArrayT: NDArray[np.complexfloating]](
    a: _ArrayLikeNumber_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `fft`
@overload  # Nd complexfloating
def ifft[ShapeT: _Shape, DTypeT: np.dtype[np.complexfloating]](
    a: np.ndarray[ShapeT, DTypeT],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd float64 | +integer
def ifft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float64 | np.integer | np.bool]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd float32 | float16
def ifft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float32 | np.float16]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex64]]: ...
@overload  # Nd longdouble
def ifft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.longdouble]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.clongdouble]]: ...
@overload  # 1d +complex
def ifft(
    a: Sequence[complex],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]: ...
@overload  # 2d +complex
def ifft(
    a: Sequence[Sequence[complex]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...
@overload  # ?d complexfloating
def ifft[ScalarT: np.complexfloating](
    a: _ArrayLike[ScalarT],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d +complex
def ifft(
    a: _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], complex],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def ifft(
    a: _ArrayLikeNumber_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def ifft[ArrayT: NDArray[np.complexfloating]](
    a: _ArrayLikeNumber_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `ihfft`
@overload  # Nd float64 | +integer
def rfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float64 | np.integer | np.bool]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd float32 | float16
def rfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float32 | np.float16]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex64]]: ...
@overload  # Nd longdouble
def rfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.longdouble]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.clongdouble]]: ...
@overload  # 1d +float
def rfft(
    a: Sequence[float],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]: ...
@overload  # 2d +float
def rfft(
    a: Sequence[Sequence[float]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...
@overload  # ?d +float
def rfft(
    a: _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], float],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def rfft(
    a: _ArrayLikeFloat_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def rfft[ArrayT: NDArray[np.complexfloating]](
    a: _ArrayLikeFloat_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `hfft`
@overload  # Nd floating
def irfft[ShapeT: _Shape, DTypeT: np.dtype[np.floating]](
    a: np.ndarray[ShapeT, DTypeT],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd complex128 | +integer
def irfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.complex128 | np.integer | np.bool]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd complex64
def irfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.complex64]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float32]]: ...
@overload  # Nd clongdouble
def irfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.clongdouble]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.longdouble]]: ...
@overload  # 1d +complex
def irfft(
    a: Sequence[complex],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...
@overload  # 2d +complex
def irfft(
    a: Sequence[Sequence[complex]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]: ...
@overload  # ?d floating
def irfft[ScalarT: np.floating](
    a: _ArrayLike[ScalarT],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d +complex | complex128 | +integer
def irfft(
    a: _DualArrayLike[np.dtype[np.complex128 | np.integer | np.bool], complex],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.float64]: ...
@overload  # fallback
def irfft(
    a: _ArrayLikeNumber_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.floating]: ...
@overload  # out: <given>
def irfft[ArrayT: NDArray[np.floating]](
    a: _ArrayLikeNumber_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `irfft` above
@overload  # Nd floating
def hfft[ShapeT: _Shape, DTypeT: np.dtype[np.floating]](
    a: np.ndarray[ShapeT, DTypeT],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd complex128 | +integer
def hfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.complex128 | np.integer | np.bool]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd complex64
def hfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.complex64]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float32]]: ...
@overload  # Nd clongdouble
def hfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.clongdouble]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.longdouble]]: ...
@overload  # 1d +complex
def hfft(
    a: Sequence[complex],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...
@overload  # 2d +complex
def hfft(
    a: Sequence[Sequence[complex]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]: ...
@overload  # ?d floating
def hfft[ScalarT: np.floating](
    a: _ArrayLike[ScalarT],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d +complex | complex128 | +integer
def hfft(
    a: _DualArrayLike[np.dtype[np.complex128 | np.integer | np.bool], complex],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.float64]: ...
@overload  # fallback
def hfft(
    a: _ArrayLikeNumber_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.floating]: ...
@overload  # out: <given>
def hfft[ArrayT: NDArray[np.floating]](
    a: _ArrayLikeNumber_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `rfft`
@overload  # Nd float64 | +integer
def ihfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float64 | np.integer | np.bool]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd float32 | float16
def ihfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float32 | np.float16]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex64]]: ...
@overload  # Nd longdouble
def ihfft[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.longdouble]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.clongdouble]]: ...
@overload  # 1d +float
def ihfft(
    a: Sequence[float],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]: ...
@overload  # 2d +float
def ihfft(
    a: Sequence[Sequence[float]],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...
@overload  # ?d +float
def ihfft(
    a: _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], float],
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def ihfft(
    a: _ArrayLikeFloat_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def ihfft[ArrayT: NDArray[np.complexfloating]](
    a: _ArrayLikeFloat_co,
    n: int | None = None,
    axis: int = -1,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `ifftn`
@overload  # Nd complexfloating
def fftn[ShapeT: _Shape, DTypeT: np.dtype[np.complexfloating]](
    a: np.ndarray[ShapeT, DTypeT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd float64 | +integer
def fftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float64 | np.integer | np.bool]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd float32 | float16
def fftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float32 | np.float16]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex64]]: ...
@overload  # Nd longdouble
def fftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.longdouble]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.clongdouble]]: ...
@overload  # 1d +complex
def fftn(
    a: Sequence[complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]: ...
@overload  # 2d +complex
def fftn(
    a: Sequence[Sequence[complex]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...
@overload  # ?d complexfloating
def fftn[ScalarT: np.complexfloating](
    a: _ArrayLike[ScalarT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d +complex
def fftn(
    a: _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def fftn(
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def fftn[ArrayT: NDArray[np.complexfloating]](
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `fftn`
@overload  # Nd complexfloating
def ifftn[ShapeT: _Shape, DTypeT: np.dtype[np.complexfloating]](
    a: np.ndarray[ShapeT, DTypeT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd float64 | +integer
def ifftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float64 | np.integer | np.bool]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd float32 | float16
def ifftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float32 | np.float16]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex64]]: ...
@overload  # Nd longdouble
def ifftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.longdouble]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.clongdouble]]: ...
@overload  # 1d +complex
def ifftn(
    a: Sequence[complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]: ...
@overload  # 2d +complex
def ifftn(
    a: Sequence[Sequence[complex]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...
@overload  # ?d complexfloating
def ifftn[ScalarT: np.complexfloating](
    a: _ArrayLike[ScalarT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d +complex
def ifftn(
    a: _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def ifftn(
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def ifftn[ArrayT: NDArray[np.complexfloating]](
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

#
@overload  # Nd float64 | +integer
def rfftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float64 | np.integer | np.bool]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd float32 | float16
def rfftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float32 | np.float16]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex64]]: ...
@overload  # Nd longdouble
def rfftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.longdouble]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.clongdouble]]: ...
@overload  # 1d +float
def rfftn(
    a: Sequence[float],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]: ...
@overload  # 2d +float
def rfftn(
    a: Sequence[Sequence[float]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...
@overload  # ?d +float
def rfftn(
    a: _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], float],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def rfftn(
    a: _ArrayLikeFloat_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def rfftn[ArrayT: NDArray[np.complexfloating]](
    a: _ArrayLikeFloat_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

#
@overload  # Nd floating
def irfftn[ShapeT: _Shape, DTypeT: np.dtype[np.floating]](
    a: np.ndarray[ShapeT, DTypeT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd complex128 | +integer
def irfftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.complex128 | np.integer | np.bool]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd complex64
def irfftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.complex64]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float32]]: ...
@overload  # Nd clongdouble
def irfftn[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.clongdouble]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.longdouble]]: ...
@overload  # 1d +complex
def irfftn(
    a: Sequence[complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...
@overload  # 2d +complex
def irfftn(
    a: Sequence[Sequence[complex]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]: ...
@overload  # ?d floating
def irfftn[ScalarT: np.floating](
    a: _ArrayLike[ScalarT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d +complex | complex128 | +integer
def irfftn(
    a: _DualArrayLike[np.dtype[np.complex128 | np.integer | np.bool], complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.float64]: ...
@overload  # fallback
def irfftn(
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.floating]: ...
@overload  # out: <given>
def irfftn[ArrayT: NDArray[np.floating]](
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `ifft2`
@overload  # Nd complexfloating
def fft2[ShapeT: _Shape, DTypeT: np.dtype[np.complexfloating]](
    a: np.ndarray[ShapeT, DTypeT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd float64 | +integer
def fft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float64 | np.integer | np.bool]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd float32 | float16
def fft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float32 | np.float16]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex64]]: ...
@overload  # Nd longdouble
def fft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.longdouble]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.clongdouble]]: ...
@overload  # 1d +complex
def fft2(
    a: Sequence[complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]: ...
@overload  # 2d +complex
def fft2(
    a: Sequence[Sequence[complex]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...
@overload  # ?d complexfloating
def fft2[ScalarT: np.complexfloating](
    a: _ArrayLike[ScalarT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d +complex
def fft2(
    a: _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def fft2(
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def fft2[ArrayT: NDArray[np.complexfloating]](
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `fft2`
@overload  # Nd complexfloating
def ifft2[ShapeT: _Shape, DTypeT: np.dtype[np.complexfloating]](
    a: np.ndarray[ShapeT, DTypeT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd float64 | +integer
def ifft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float64 | np.integer | np.bool]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd float32 | float16
def ifft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float32 | np.float16]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex64]]: ...
@overload  # Nd longdouble
def ifft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.longdouble]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.clongdouble]]: ...
@overload  # 1d +complex
def ifft2(
    a: Sequence[complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]: ...
@overload  # 2d +complex
def ifft2(
    a: Sequence[Sequence[complex]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...
@overload  # ?d complexfloating
def ifft2[ScalarT: np.complexfloating](
    a: _ArrayLike[ScalarT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d +complex
def ifft2(
    a: _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def ifft2(
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def ifft2[ArrayT: NDArray[np.complexfloating]](
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

#
@overload  # Nd float64 | +integer
def rfft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float64 | np.integer | np.bool]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd float32 | float16
def rfft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.float32 | np.float16]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.complex64]]: ...
@overload  # Nd longdouble
def rfft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.longdouble]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.clongdouble]]: ...
@overload  # 1d +float
def rfft2(
    a: Sequence[float],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]: ...
@overload  # 2d +float
def rfft2(
    a: Sequence[Sequence[float]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...
@overload  # ?d +float
def rfft2(
    a: _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], float],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def rfft2(
    a: _ArrayLikeFloat_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.complexfloating]: ...
@overload  # out: <given>
def rfft2[ArrayT: NDArray[np.complexfloating]](
    a: _ArrayLikeFloat_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

#
@overload  # Nd floating
def irfft2[ShapeT: _Shape, DTypeT: np.dtype[np.floating]](
    a: np.ndarray[ShapeT, DTypeT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd complex128 | +integer
def irfft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.complex128 | np.integer | np.bool]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd complex64
def irfft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.complex64]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float32]]: ...
@overload  # Nd clongdouble
def irfft2[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.clongdouble]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.longdouble]]: ...
@overload  # 1d +complex
def irfft2(
    a: Sequence[complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...
@overload  # 2d +complex
def irfft2(
    a: Sequence[Sequence[complex]],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]: ...
@overload  # ?d floating
def irfft2[ScalarT: np.floating](
    a: _ArrayLike[ScalarT],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d +complex | complex128 | +integer
def irfft2(
    a: _DualArrayLike[np.dtype[np.complex128 | np.integer | np.bool], complex],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.float64]: ...
@overload  # fallback
def irfft2(
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    out: None = None,
) -> NDArray[np.floating]: ...
@overload  # out: <given>
def irfft2[ArrayT: NDArray[np.floating]](
    a: _ArrayLikeNumber_co,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    norm: _NormKind = None,
    *,
    out: ArrayT,
) -> ArrayT: ...
