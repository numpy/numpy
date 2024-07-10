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
from numpy._typing import _ArrayLikeInt_co, _FloatLike_co

from ._polytypes import (
    _AnyComplexSeriesND,
    _AnyFloatSeriesND,
    _AnyInt,
    _AnyScalar,
    _AnyComplexSeries1D,
    _AnyFloatSeries1D,
    _AnyNumberSeries1D,
    _AnyObjectScalar,
    _AnyObjectSeries1D,
    _AnySeries1D,
    _AnySeriesND,
    _Array1D,
    _AnyFloatScalar,
    _CoefArrayND,
    _CoefArray1D,
    _AnyFloatScalar,
    _FuncBinOp,
    _FuncValND,
    _FuncVanderND,
    _Interval,
    _AnyNumberScalar,
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
    alist: npt.NDArray[np.integer[Any]],
    trim: bool = ...,
) -> list[_Array1D[np.floating[Any]]]: ...
@overload
def as_series(
    alist: npt.NDArray[np.floating[Any]],
    trim: bool = ...,
) -> list[_Array1D[np.floating[Any]]]: ...
@overload
def as_series(
    alist: npt.NDArray[np.complexfloating[Any, Any]],
    trim: bool = ...,
) -> list[_Array1D[np.complexfloating[Any, Any]]]: ...
@overload
def as_series(
    alist: npt.NDArray[np.object_],
    trim: bool = ...,
) -> list[_Array1D[np.object_]]: ...
@overload
def as_series(  # type: ignore[overload-overlap]
    alist: Iterable[npt.NDArray[np.integer[Any]]],
    trim: bool = ...,
) -> list[_Array1D[np.floating[Any]]]: ...
@overload
def as_series(
    alist: Iterable[npt.NDArray[np.floating[Any]]],
    trim: bool = ...,
) -> list[_Array1D[np.floating[Any]]]: ...
@overload
def as_series(
    alist: Iterable[npt.NDArray[np.complexfloating[Any, Any]]],
    trim: bool = ...,
) -> list[_Array1D[np.complexfloating[Any, Any]]]: ...
@overload
def as_series(
    alist: Iterable[npt.NDArray[np.object_]],
    trim: bool = ...,
) -> list[_Array1D[np.object_]]: ...
@overload
def as_series(  # type: ignore[overload-overlap]
    alist: Iterable[_AnyFloatSeries1D | float],
    trim: bool = ...,
) -> list[_Array1D[np.floating[Any]]]: ...
@overload
def as_series(
    alist: Iterable[_AnyComplexSeries1D | complex],
    trim: bool = ...,
) -> list[_Array1D[np.complexfloating[Any, Any]]]: ...
@overload
def as_series(
    alist: Iterable[_AnyObjectSeries1D | object],
    trim: bool = ...,
) -> list[_Array1D[np.object_]]: ...

_T_seq = TypeVar("_T_seq", bound=_CoefArrayND | _SimpleSequence[_AnyScalar])
def trimseq(seq: _T_seq) -> _T_seq: ...

@overload
def trimcoef(  # type: ignore[overload-overlap]
    c: npt.NDArray[np.integer[Any]] | npt.NDArray[np.floating[Any]],
    tol: _AnyFloatScalar = ...,
) -> _Array1D[np.floating[Any]]: ...
@overload
def trimcoef(
    c: npt.NDArray[np.complexfloating[Any, Any]],
    tol: _AnyFloatScalar = ...,
) -> _Array1D[np.complexfloating[Any, Any]]: ...
@overload
def trimcoef(
    c: npt.NDArray[np.object_],
    tol: _AnyFloatScalar = ...,
) -> _Array1D[np.object_]: ...
@overload
def trimcoef(  # type: ignore[overload-overlap]
    c: _AnyFloatSeries1D | float,
    tol: _AnyFloatScalar = ...,
) -> _Array1D[np.floating[Any]]: ...
@overload
def trimcoef(
    c: _AnyComplexSeries1D | complex,
    tol: _AnyFloatScalar = ...,
) -> _Array1D[np.complexfloating[Any, Any]]: ...
@overload
def trimcoef(
    c: _AnyObjectSeries1D | object,
    tol: _AnyFloatScalar = ...,
) -> _Array1D[np.object_]: ...

@overload
def getdomain(  # type: ignore[overload-overlap]
    x: npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]],
) -> _Interval[np.float64]: ...
@overload
def getdomain(
    x: npt.NDArray[np.complexfloating[Any, Any]],
) -> _Interval[np.complex128]: ...
@overload
def getdomain(
    x: npt.NDArray[np.object_],
) -> _Interval[np.object_]: ...
@overload
def getdomain(  # type: ignore[overload-overlap]
    x: _AnyFloatSeries1D | float,
) -> _Interval[np.float64]: ...
@overload
def getdomain(
    x: _AnyComplexSeries1D | complex,
) -> _Interval[np.complex128]: ...
@overload
def getdomain(
    x: _AnyObjectSeries1D | object,
) -> _Interval[np.object_]: ...

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
    old: _AnyFloatSeries1D,
    new: _AnyFloatSeries1D,
) -> _Tuple2[np.floating[Any]]: ...
@overload
def mapparms(
    old: _AnyNumberSeries1D,
    new: _AnyNumberSeries1D,
) -> _Tuple2[np.complexfloating[Any, Any]]: ...
@overload
def mapparms(
    old: _AnySeries1D,
    new: _AnySeries1D,
) -> _Tuple2[object]: ...

@overload
def mapdomain(  # type: ignore[overload-overlap]
    x: _AnyFloatScalar,
    old: _AnyFloatSeries1D,
    new: _AnyFloatSeries1D,
) -> np.floating[Any]: ...
@overload
def mapdomain(
    x: _AnyNumberScalar,
    old: _AnyComplexSeries1D,
    new: _AnyComplexSeries1D,
) -> np.complexfloating[Any, Any]: ...
@overload
def mapdomain(
    x: _AnyObjectScalar | _AnyNumberScalar,
    old: _AnyObjectSeries1D | _AnyComplexSeries1D,
    new: _AnyObjectSeries1D | _AnyComplexSeries1D,
) -> object: ...
@overload
def mapdomain(  # type: ignore[overload-overlap]
    x: npt.NDArray[np.floating[Any] | np.integer[Any]],
    old: npt.NDArray[np.floating[Any] | np.integer[Any]],
    new: npt.NDArray[np.floating[Any] | np.integer[Any]],
) -> _Array1D[np.floating[Any]]: ...
@overload
def mapdomain(
    x: npt.NDArray[np.number[Any]],
    old: npt.NDArray[np.number[Any]],
    new: npt.NDArray[np.number[Any]],
) -> _Array1D[np.complexfloating[Any, Any]]: ...
@overload
def mapdomain(
    x: npt.NDArray[np.object_ | np.number[Any]],
    old: npt.NDArray[np.object_ | np.number[Any]],
    new: npt.NDArray[np.object_ | np.number[Any]],
) -> _Array1D[np.object_]: ...
@overload
def mapdomain(
    x: _AnyFloatSeries1D,
    old: _AnyFloatSeries1D,
    new: _AnyFloatSeries1D,
) -> _Array1D[np.floating[Any]]: ...
@overload
def mapdomain(
    x: _AnyNumberSeries1D,
    old: _AnyNumberSeries1D,
    new: _AnyNumberSeries1D,
) -> _Array1D[np.complexfloating[Any, Any]]: ...
@overload
def mapdomain(
    x: _AnySeries1D,
    old:_AnySeries1D,
    new: _AnySeries1D,
) -> _Array1D[np.object_]: ...
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
    roots: _AnyFloatSeries1D,
) -> _Array1D[np.floating[Any]]: ...
@overload
def _fromroots(
    line_f: _AnyLineF,
    mul_f: _AnyMulF,
    roots: _AnyComplexSeries1D,
) -> _Array1D[np.complexfloating[Any, Any]]: ...
@overload
def _fromroots(
    line_f: _AnyLineF,
    mul_f: _AnyMulF,
    roots: _AnyObjectSeries1D,
) -> _Array1D[np.object_]: ...
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
    c1: _AnyFloatSeries1D,
    c2: _AnyFloatSeries1D,
) -> _Tuple2[_Array1D[np.floating[Any]]]: ...
@overload
def _div(
    mul_f: _AnyMulF,
    c1: _AnyComplexSeries1D,
    c2: _AnyComplexSeries1D,
) -> _Tuple2[_Array1D[np.complexfloating[Any, Any]]]: ...
@overload
def _div(
    mul_f: _AnyMulF,
    c1: _AnyObjectSeries1D,
    c2: _AnyObjectSeries1D,
) -> _Tuple2[_Array1D[np.object_]]: ...
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
    c: _AnyFloatSeries1D,
    pow: _AnyInt,
    maxpower: None | _AnyInt = ...,
) -> _Array1D[np.floating[Any]]: ...
@overload
def _pow(
    mul_f: _AnyMulF,
    c: _AnyComplexSeries1D,
    pow: _AnyInt,
    maxpower: None | _AnyInt = ...,
) -> _Array1D[np.complexfloating[Any, Any]]: ...
@overload
def _pow(
    mul_f: _AnyMulF,
    c: _AnyObjectSeries1D,
    pow: _AnyInt,
    maxpower: None | _AnyInt = ...,
) -> _Array1D[np.object_]: ...
@overload
def _pow(
    mul_f: _AnyMulF,
    c: _AnySeries1D,
    pow: _AnyInt,
    maxpower: None | _AnyInt = ...,
) -> _CoefArray1D: ...

# keep in sync with `_polytypes._FuncFit`
@overload
def _fit(  # type: ignore[overload-overlap]
    vander_f: _AnyVanderF,
    x: _AnyFloatSeries1D,
    y: _AnyFloatSeriesND,
    deg: _ArrayLikeInt_co,
    domain: None | _AnyFloatSeries1D = ...,
    rcond: None | float = ...,
    full: Literal[False] = ...,
    w: None | _AnyFloatSeries1D = ...,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def _fit(
    vander_f: _AnyVanderF,
    x: _AnyComplexSeries1D,
    y: _AnyComplexSeriesND,
    deg: _ArrayLikeInt_co,
    domain: None | _AnyComplexSeries1D = ...,
    rcond: None | float = ...,
    full: Literal[False] = ...,
    w: None | _AnyComplexSeries1D = ...,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
@overload
def _fit(
    vander_f: _AnyVanderF,
    x: _AnySeries1D,
    y: _AnySeriesND,
    deg: _ArrayLikeInt_co,
    domain: None | _AnySeries1D = ...,
    rcond: None | float = ...,
    full: Literal[False] = ...,
    w: None | _AnySeries1D = ...,
) -> _CoefArrayND: ...
@overload
def _fit(
    vander_f: _AnyVanderF,
    x: _AnySeries1D,
    y: _AnySeries1D,
    deg: _ArrayLikeInt_co,
    domain: None | _AnySeries1D,
    rcond: None | float ,
    full: Literal[True],
    /,
    w: None | _AnySeries1D = ...,
) -> tuple[_CoefArray1D, Sequence[np.inexact[Any] | np.int32]]: ...
@overload
def _fit(
    vander_f: _AnyVanderF,
    x: _AnySeries1D,
    y: _AnySeries1D,
    deg: _ArrayLikeInt_co,
    domain: None | _AnySeries1D = ...,
    rcond: None | float = ...,
    *,
    full: Literal[True],
    w: None | _AnySeries1D = ...,
) -> tuple[_CoefArray1D, Sequence[np.inexact[Any] | np.int32]]: ...

def _as_int(x: SupportsIndex, desc: str) -> int: ...
def format_float(x: _FloatLike_co, parens: bool = ...) -> str: ...
