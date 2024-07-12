from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    Final,
    Literal,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt

from ._polytypes import (
    _AnyComplexScalar,
    _AnyComplexSeries1D,
    _AnyComplexSeriesND,
    _AnyIntSeries1D,
    _AnyRealSeries1D,
    _AnyRealSeriesND,
    _AnyIntArg,
    _AnyComplexSeries1D,
    _AnyObjectSeries1D,
    _AnyRealScalar,
    _AnyScalar,
    _AnySeries1D,
    _AnySeriesND,
    _AnyRealScalar,
    _Array2,
    _CoefArrayND,
    _CoefArray1D,
    _ComplexArray1D,
    _ComplexArrayND,
    _FloatArray1D,
    _FloatArrayND,
    _FuncBinOp,
    _FuncValND,
    _FuncVanderND,
    _IntArrayND,
    _ObjectArray1D,
    _ObjectArrayND,
    _SimpleSequence,
    _SupportsLenAndGetItem,
    _Tuple2,
)

___all__ = [
    "as_series",
    "format_float"
    "getdomain",
    "mapdomain",
    "mapparms",
    "trimcoef",
    "trimseq",
]

_AnyLineF: TypeAlias = Callable[[_AnyScalar, _AnyScalar], _CoefArrayND]
_AnyMulF: TypeAlias = Callable[[npt.ArrayLike, npt.ArrayLike], _CoefArrayND]
_AnyVanderF: TypeAlias = Callable[[npt.ArrayLike, SupportsIndex], _CoefArrayND]

@overload
def as_series(
    alist: _IntArrayND | _FloatArrayND,
    trim: bool = ...,
) -> list[_FloatArray1D]: ...
@overload
def as_series(
    alist: _ComplexArrayND,
    trim: bool = ...,
) -> list[_ComplexArray1D]: ...
@overload
def as_series(
    alist: _ObjectArrayND,
    trim: bool = ...,
) -> list[_ObjectArray1D]: ...
@overload
def as_series(  # type: ignore[overload-overlap]
    alist: Iterable[_FloatArrayND | _IntArrayND],
    trim: bool = ...,
) -> list[_FloatArray1D]: ...
@overload
def as_series(
    alist: Iterable[_ComplexArrayND],
    trim: bool = ...,
) -> list[_ComplexArray1D]: ...
@overload
def as_series(
    alist: Iterable[_ObjectArrayND],
    trim: bool = ...,
) -> list[_ObjectArray1D]: ...
@overload
def as_series(  # type: ignore[overload-overlap]
    alist: Iterable[_AnyRealSeries1D | float],
    trim: bool = ...,
) -> list[_FloatArray1D]: ...
@overload
def as_series(
    alist: Iterable[_AnyComplexSeries1D | complex],
    trim: bool = ...,
) -> list[_ComplexArray1D]: ...
@overload
def as_series(
    alist: Iterable[_AnyObjectSeries1D | object],
    trim: bool = ...,
) -> list[_ObjectArray1D]: ...

_T_seq = TypeVar("_T_seq", bound=_CoefArrayND | _SimpleSequence[_AnyScalar])
def trimseq(seq: _T_seq) -> _T_seq: ...

@overload
def trimcoef(  # type: ignore[overload-overlap]
    c: _IntArrayND | _FloatArrayND,
    tol: _AnyRealScalar = ...,
) -> _FloatArray1D: ...
@overload
def trimcoef(
    c: _ComplexArrayND,
    tol: _AnyRealScalar = ...,
) -> _ComplexArray1D: ...
@overload
def trimcoef(
    c: _ObjectArrayND,
    tol: _AnyRealScalar = ...,
) -> _ObjectArray1D: ...
@overload
def trimcoef(  # type: ignore[overload-overlap]
    c: _AnyRealSeries1D | float,
    tol: _AnyRealScalar = ...,
) -> _FloatArray1D: ...
@overload
def trimcoef(
    c: _AnyComplexSeries1D | complex,
    tol: _AnyRealScalar = ...,
) -> _ComplexArray1D: ...
@overload
def trimcoef(
    c: _AnyObjectSeries1D | object,
    tol: _AnyRealScalar = ...,
) -> _ObjectArray1D: ...

@overload
def getdomain(  # type: ignore[overload-overlap]
    x: _FloatArrayND | _IntArrayND,
) -> _Array2[np.float64]: ...
@overload
def getdomain(
    x: _ComplexArrayND,
) -> _Array2[np.complex128]: ...
@overload
def getdomain(
    x: _ObjectArrayND,
) -> _Array2[np.object_]: ...
@overload
def getdomain(  # type: ignore[overload-overlap]
    x: _AnyRealSeries1D | float,
) -> _Array2[np.float64]: ...
@overload
def getdomain(
    x: _AnyComplexSeries1D | complex,
) -> _Array2[np.complex128]: ...
@overload
def getdomain(
    x: _AnyObjectSeries1D | object,
) -> _Array2[np.object_]: ...

@overload
def mapparms(  # type: ignore[overload-overlap]
    old: npt.NDArray[np.floating[Any] | np.integer[Any]],
    new: npt.NDArray[np.floating[Any] | np.integer[Any]],
) -> _Tuple2[np.floating[Any]]: ...
@overload
def mapparms(
    old: npt.NDArray[np.number[Any]],
    new: npt.NDArray[np.number[Any]],
) -> _Tuple2[np.complexfloating[Any, Any]]: ...
@overload
def mapparms(
    old: npt.NDArray[np.object_ | np.number[Any]],
    new: npt.NDArray[np.object_ | np.number[Any]],
) -> _Tuple2[object]: ...
@overload
def mapparms(  # type: ignore[overload-overlap]
    old: _SupportsLenAndGetItem[float],
    new: _SupportsLenAndGetItem[float],
) -> _Tuple2[float]: ...
@overload
def mapparms(
    old: _SupportsLenAndGetItem[complex],
    new: _SupportsLenAndGetItem[complex],
) -> _Tuple2[complex]: ...
@overload
def mapparms(
    old: _AnyRealSeries1D,
    new: _AnyRealSeries1D,
) -> _Tuple2[np.floating[Any]]: ...
@overload
def mapparms(
    old: _AnyComplexSeries1D,
    new: _AnyComplexSeries1D,
) -> _Tuple2[np.complexfloating[Any, Any]]: ...
@overload
def mapparms(
    old: _AnySeries1D,
    new: _AnySeries1D,
) -> _Tuple2[object]: ...

@overload
def mapdomain(  # type: ignore[overload-overlap]
    x: _AnyRealScalar,
    old: _AnyRealSeries1D,
    new: _AnyRealSeries1D,
) -> np.floating[Any]: ...
@overload
def mapdomain(
    x: _AnyComplexScalar,
    old: _AnyComplexSeries1D,
    new: _AnyComplexSeries1D,
) -> np.complexfloating[Any, Any]: ...
@overload
def mapdomain(
    x: _AnyScalar,
    old: _AnySeries1D,
    new: _AnySeries1D,
) -> object: ...
@overload
def mapdomain(  # type: ignore[overload-overlap]
    x: npt.NDArray[np.floating[Any] | np.integer[Any]],
    old: npt.NDArray[np.floating[Any] | np.integer[Any]],
    new: npt.NDArray[np.floating[Any] | np.integer[Any]],
) -> _FloatArray1D: ...
@overload
def mapdomain(
    x: npt.NDArray[np.number[Any]],
    old: npt.NDArray[np.number[Any]],
    new: npt.NDArray[np.number[Any]],
) -> _ComplexArray1D: ...
@overload
def mapdomain(
    x: npt.NDArray[np.object_ | np.number[Any]],
    old: npt.NDArray[np.object_ | np.number[Any]],
    new: npt.NDArray[np.object_ | np.number[Any]],
) -> _ObjectArray1D: ...
@overload
def mapdomain(  # type: ignore[overload-overlap]
    x: _AnyRealSeries1D,
    old: _AnyRealSeries1D,
    new: _AnyRealSeries1D,
) -> _FloatArray1D: ...
@overload
def mapdomain(
    x: _AnyComplexSeries1D,
    old: _AnyComplexSeries1D,
    new: _AnyComplexSeries1D,
) -> _ComplexArray1D: ...
@overload
def mapdomain(
    x: _AnySeries1D,
    old:_AnySeries1D,
    new: _AnySeries1D,
) -> _ObjectArray1D: ...
@overload
def mapdomain(
    x: object,
    old: _AnySeries1D,
    new: _AnySeries1D,
) -> object: ...

def _nth_slice(
    i: SupportsIndex,
    ndim: SupportsIndex,
) -> tuple[None | slice, ...]: ...

_vander_nd: _FuncVanderND[Literal["_vander_nd"]]
_vander_nd_flat: _FuncVanderND[Literal["_vander_nd_flat"]]

# keep in sync with `._polytypes._FuncFromRoots`
@overload
def _fromroots(  # type: ignore[overload-overlap]
    line_f: _AnyLineF,
    mul_f: _AnyMulF,
    roots: _AnyRealSeries1D,
) -> _FloatArray1D: ...
@overload
def _fromroots(
    line_f: _AnyLineF,
    mul_f: _AnyMulF,
    roots: _AnyComplexSeries1D,
) -> _ComplexArray1D: ...
@overload
def _fromroots(
    line_f: _AnyLineF,
    mul_f: _AnyMulF,
    roots: _AnyObjectSeries1D,
) -> _ObjectArray1D: ...
@overload
def _fromroots(
    line_f: _AnyLineF,
    mul_f: _AnyMulF,
    roots: _AnySeries1D,
) -> _CoefArray1D: ...

_valnd: _FuncValND[Literal["_valnd"]]
_gridnd: _FuncValND[Literal["_gridnd"]]

# keep in sync with `_polytypes._FuncBinOp`
@overload
def _div(  # type: ignore[overload-overlap]
    mul_f: _AnyMulF,
    c1: _AnyRealSeries1D,
    c2: _AnyRealSeries1D,
) -> _Tuple2[_FloatArray1D]: ...
@overload
def _div(
    mul_f: _AnyMulF,
    c1: _AnyComplexSeries1D,
    c2: _AnyComplexSeries1D,
) -> _Tuple2[_ComplexArray1D]: ...
@overload
def _div(
    mul_f: _AnyMulF,
    c1: _AnyObjectSeries1D,
    c2: _AnyObjectSeries1D,
) -> _Tuple2[_ObjectArray1D]: ...
@overload
def _div(
    mul_f: _AnyMulF,
    c1: _AnySeries1D,
    c2: _AnySeries1D,
) -> _Tuple2[_CoefArray1D]: ...

_add: Final[_FuncBinOp]
_sub: Final[_FuncBinOp]

# keep in sync with `_polytypes._FuncPow`
@overload
def _pow(  # type: ignore[overload-overlap]
    mul_f: _AnyMulF,
    c: _AnyRealSeries1D,
    pow: _AnyIntArg,
    maxpower: None | _AnyIntArg = ...,
) -> _FloatArray1D: ...
@overload
def _pow(
    mul_f: _AnyMulF,
    c: _AnyComplexSeries1D,
    pow: _AnyIntArg,
    maxpower: None | _AnyIntArg = ...,
) -> _ComplexArray1D: ...
@overload
def _pow(
    mul_f: _AnyMulF,
    c: _AnyObjectSeries1D,
    pow: _AnyIntArg,
    maxpower: None | _AnyIntArg = ...,
) -> _ObjectArray1D: ...
@overload
def _pow(
    mul_f: _AnyMulF,
    c: _AnySeries1D,
    pow: _AnyIntArg,
    maxpower: None | _AnyIntArg = ...,
) -> _CoefArray1D: ...

# keep in sync with `_polytypes._FuncFit`
@overload
def _fit(  # type: ignore[overload-overlap]
    vander_f: _AnyVanderF,
    x: _AnyRealSeries1D,
    y: _AnyRealSeriesND,
    deg: _AnyIntSeries1D,
    domain: None | _AnyRealSeries1D = ...,
    rcond: None | _AnyRealScalar = ...,
    full: Literal[False] = ...,
    w: None | _AnyRealSeries1D = ...,
) -> _FloatArrayND: ...
@overload
def _fit(
    vander_f: _AnyVanderF,
    x: _AnyComplexSeries1D,
    y: _AnyComplexSeriesND,
    deg: _AnyIntSeries1D,
    domain: None | _AnyComplexSeries1D = ...,
    rcond: None | _AnyRealScalar = ...,
    full: Literal[False] = ...,
    w: None | _AnyComplexSeries1D = ...,
) -> _ComplexArrayND: ...
@overload
def _fit(
    vander_f: _AnyVanderF,
    x: _AnySeries1D,
    y: _AnySeriesND,
    deg: _AnyIntSeries1D,
    domain: None | _AnySeries1D = ...,
    rcond: None | _AnyRealScalar = ...,
    full: Literal[False] = ...,
    w: None | _AnySeries1D = ...,
) -> _CoefArrayND: ...
@overload
def _fit(
    vander_f: _AnyVanderF,
    x: _AnySeries1D,
    y: _AnySeries1D,
    deg: _AnyIntSeries1D,
    domain: None | _AnySeries1D,
    rcond: None | _AnyRealScalar ,
    full: Literal[True],
    /,
    w: None | _AnySeries1D = ...,
) -> tuple[_CoefArray1D, Sequence[np.inexact[Any] | np.int32]]: ...
@overload
def _fit(
    vander_f: _AnyVanderF,
    x: _AnySeries1D,
    y: _AnySeries1D,
    deg: _AnyIntSeries1D,
    domain: None | _AnySeries1D = ...,
    rcond: None | _AnyRealScalar = ...,
    *,
    full: Literal[True],
    w: None | _AnySeries1D = ...,
) -> tuple[_CoefArray1D, Sequence[np.inexact[Any] | np.int32]]: ...

def _as_int(x: SupportsIndex, desc: str) -> int: ...
def format_float(x: _AnyRealScalar, parens: bool = ...) -> str: ...
