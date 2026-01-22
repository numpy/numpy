from typing import Any, Final, Literal as L, overload

import numpy as np
from numpy._typing import ArrayLike, NDArray, _ArrayLike, _Shape, _ShapeLike

__all__ = ["fftfreq", "fftshift", "ifftshift", "rfftfreq"]

###

type _Device = L["cpu"]

type _IntLike = int | np.integer

type _AsFloat64 = np.float64 | np.float32 | np.float16 | np.integer | np.bool
type _AsComplex128 = np.complex128 | np.complex64
type _Inexact80 = np.longdouble | np.clongdouble

type _Array[ShapeT: _Shape, ScalarT: np.generic] = np.ndarray[ShapeT, np.dtype[ScalarT]]
type _1D = tuple[int]

###

integer_types: Final[tuple[type[int], type[np.integer]]] = ...

@overload
def fftshift[ScalarT: np.generic](x: _ArrayLike[ScalarT], axes: _ShapeLike | None = None) -> NDArray[ScalarT]: ...
@overload
def fftshift(x: ArrayLike, axes: _ShapeLike | None = None) -> NDArray[Any]: ...

#
@overload
def ifftshift[ScalarT: np.generic](x: _ArrayLike[ScalarT], axes: _ShapeLike | None = None) -> NDArray[ScalarT]: ...
@overload
def ifftshift(x: ArrayLike, axes: _ShapeLike | None = None) -> NDArray[Any]: ...

# keep in sync with `rfftfreq` below
@overload  # 0d +f64  (default)
def fftfreq(
    n: _IntLike,
    d: _AsFloat64 | float = 1.0,
    device: _Device | None = None,
) -> _Array[_1D, np.float64]: ...
@overload  # 0d c64 | c128
def fftfreq(
    n: _IntLike,
    d: _AsComplex128,
    device: _Device | None = None,
) -> _Array[_1D, np.complex128]: ...
@overload  # 0d +complex
def fftfreq(
    n: _IntLike,
    d: complex,
    device: _Device | None = None,
) -> _Array[_1D, np.complex128 | Any]: ...
@overload  # 0d T: f80 | c160
def fftfreq[ScalarT: _Inexact80](
    n: _IntLike,
    d: ScalarT,
    device: _Device | None = None,
) -> _Array[_1D, ScalarT]: ...
@overload  # nd +f64
def fftfreq[ShapeT: _Shape](
    n: _IntLike,
    d: _Array[ShapeT, _AsFloat64],
    device: _Device | None = None,
) -> _Array[ShapeT, np.float64]: ...
@overload  # nd c64 | c128
def fftfreq[ShapeT: _Shape](
    n: _IntLike,
    d: _Array[ShapeT, _AsComplex128],
    device: _Device | None = None,
) -> _Array[ShapeT, np.complex128]: ...
@overload  # nd T: f80 | c160
def fftfreq[ShapeT: _Shape, LongDoubleT: _Inexact80](
    n: _IntLike,
    d: _Array[ShapeT, LongDoubleT],
    device: _Device | None = None,
) -> _Array[ShapeT, LongDoubleT]: ...
@overload  # nd +complex (fallback)
def fftfreq[ShapeT: _Shape](
    n: _IntLike,
    d: _Array[ShapeT, np.number | np.bool],
    device: _Device | None = None,
) -> _Array[ShapeT, Any]: ...

# keep in sync with `fftfreq` above
@overload  # 0d +f64  (default)
def rfftfreq(
    n: _IntLike,
    d: _AsFloat64 | float = 1.0,
    device: _Device | None = None,
) -> _Array[_1D, np.float64]: ...
@overload  # 0d c64 | c128
def rfftfreq(
    n: _IntLike,
    d: _AsComplex128,
    device: _Device | None = None,
) -> _Array[_1D, np.complex128]: ...
@overload  # 0d +complex
def rfftfreq(
    n: _IntLike,
    d: complex,
    device: _Device | None = None,
) -> _Array[_1D, np.complex128 | Any]: ...
@overload  # 0d T: f80 | c160
def rfftfreq[LongDoubleT: _Inexact80](
    n: _IntLike,
    d: LongDoubleT,
    device: _Device | None = None,
) -> _Array[_1D, LongDoubleT]: ...
@overload  # nd +f64
def rfftfreq[ShapeT: _Shape](
    n: _IntLike,
    d: _Array[ShapeT, _AsFloat64],
    device: _Device | None = None,
) -> _Array[ShapeT, np.float64]: ...
@overload  # nd c64 | c128
def rfftfreq[ShapeT: _Shape](
    n: _IntLike,
    d: _Array[ShapeT, _AsComplex128],
    device: _Device | None = None,
) -> _Array[ShapeT, np.complex128]: ...
@overload  # nd T: f80 | c160
def rfftfreq[ShapeT: _Shape, LongDoubleT: _Inexact80](
    n: _IntLike,
    d: _Array[ShapeT, LongDoubleT],
    device: _Device | None = None,
) -> _Array[ShapeT, LongDoubleT]: ...
@overload  # nd +complex (fallback)
def rfftfreq[ShapeT: _Shape](
    n: _IntLike,
    d: _Array[ShapeT, np.number | np.bool],
    device: _Device | None = None,
) -> _Array[ShapeT, Any]: ...
