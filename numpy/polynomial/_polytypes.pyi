from collections.abc import Sequence
from typing import (
    Any,
    NoReturn,
    Protocol,
    Self,
    SupportsIndex,
    SupportsInt,
    overload,
    type_check_only,
)

import numpy as np
import numpy.typing as npt
from numpy._typing import (
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _ComplexLike_co,
    _FloatLike_co,
    _IntLike_co,
    _NestedSequence,
    _NumberLike_co,
    _SupportsArray,
)

@type_check_only
class _CanArray[ArrayT: np.ndarray](Protocol):
    def __array__(self, /) -> ArrayT: ...

# compatible with e.g. int, float, complex, Decimal, Fraction, and ABCPolyBase
@type_check_only
class _SupportsCoefOps[T](Protocol):
    def __eq__(self, x: object, /) -> bool: ...
    def __ne__(self, x: object, /) -> bool: ...
    def __neg__(self, /) -> Self: ...
    def __pos__(self, /) -> Self: ...
    def __add__(self, x: T, /) -> Self: ...
    def __sub__(self, x: T, /) -> Self: ...
    def __mul__(self, x: T, /) -> Self: ...
    def __pow__(self, x: T, /) -> Self | float: ...
    def __radd__(self, x: T, /) -> Self: ...
    def __rsub__(self, x: T, /) -> Self: ...
    def __rmul__(self, x: T, /) -> Self: ...

type _ToCoef1D = _SupportsCoefOps[Any] | Sequence[_SupportsCoefOps[Any]] | npt.NDArray[np.number | np.bool | np.object_]
type _ToCoefND = _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]]

type _PolyScalar = np.bool | np.number | np.object_

type _Series[ScalarT: _PolyScalar] = np.ndarray[tuple[int], np.dtype[ScalarT]]

type _FloatSeries = _Series[np.floating]
type _ComplexSeries = _Series[np.complexfloating]
type _ObjectSeries = _Series[np.object_]
type _CoefSeries = _Series[np.inexact | np.object_]

type _FloatArray = npt.NDArray[np.floating]
type _ComplexArray = npt.NDArray[np.complexfloating]
type _ObjectArray = npt.NDArray[np.object_]
type _CoefArray = npt.NDArray[np.inexact | np.object_]

type _Tuple2[_T] = tuple[_T, _T]

type _AnyInt = SupportsInt | SupportsIndex

type _CoefObjectLike_co = np.object_ | _SupportsCoefOps[Any]
type _CoefLike_co = _NumberLike_co | _CoefObjectLike_co

# The term "series" is used here to refer to 1-d arrays of numeric scalars.
type _SeriesLikeBool_co = _SupportsArray[np.dtype[np.bool]] | Sequence[bool | np.bool]
type _SeriesLikeInt_co = _SupportsArray[np.dtype[np.integer | np.bool]] | Sequence[_IntLike_co]
type _SeriesLikeFloat_co = _SupportsArray[np.dtype[np.floating | np.integer | np.bool]] | Sequence[_FloatLike_co]
type _SeriesLikeComplex_co = _SupportsArray[np.dtype[np.number | np.bool]] | Sequence[_ComplexLike_co]
type _SeriesLikeObject_co = _SupportsArray[np.dtype[np.object_]] | Sequence[_CoefObjectLike_co]
type _SeriesLikeCoef_co = _SupportsArray[np.dtype[_PolyScalar]] | Sequence[_CoefLike_co]

type _ArrayLikeCoefObject_co = _CoefObjectLike_co | _SeriesLikeObject_co | _NestedSequence[_SeriesLikeObject_co]
type _ArrayLikeCoef_co = npt.NDArray[_PolyScalar] | _ArrayLikeNumber_co | _ArrayLikeCoefObject_co

type _Line[ScalarT: _PolyScalar] = np.ndarray[tuple[int], np.dtype[ScalarT]]

@type_check_only
class _FuncLine(Protocol):
    @overload
    def __call__[ScalarT: _PolyScalar](self, /, off: ScalarT, scl: ScalarT) -> _Line[ScalarT]: ...
    @overload
    def __call__(self, /, off: int, scl: int) -> _Line[np.int_]: ...
    @overload
    def __call__(self, /, off: float, scl: float) -> _Line[np.float64]: ...
    @overload
    def __call__(self, /, off: complex, scl: complex) -> _Line[np.complex128]: ...
    @overload
    def __call__(self, /, off: _SupportsCoefOps[Any], scl: _SupportsCoefOps[Any]) -> _Line[np.object_]: ...

@type_check_only
class _FuncFromRoots(Protocol):
    @overload
    def __call__(self, /, roots: _SeriesLikeFloat_co) -> _FloatSeries: ...
    @overload
    def __call__(self, /, roots: _SeriesLikeComplex_co) -> _ComplexSeries: ...
    @overload
    def __call__(self, /, roots: _SeriesLikeCoef_co) -> _ObjectSeries: ...

@type_check_only
class _FuncBinOp(Protocol):
    @overload
    def __call__(self, /, c1: _SeriesLikeBool_co, c2: _SeriesLikeBool_co) -> NoReturn: ...
    @overload
    def __call__(self, /, c1: _SeriesLikeFloat_co, c2: _SeriesLikeFloat_co) -> _FloatSeries: ...
    @overload
    def __call__(self, /, c1: _SeriesLikeComplex_co, c2: _SeriesLikeComplex_co) -> _ComplexSeries: ...
    @overload
    def __call__(self, /, c1: _SeriesLikeCoef_co, c2: _SeriesLikeCoef_co) -> _ObjectSeries: ...

@type_check_only
class _FuncPoly2Ortho(Protocol):
    @overload
    def __call__(self, /, pol: _SeriesLikeFloat_co) -> _FloatSeries: ...
    @overload
    def __call__(self, /, pol: _SeriesLikeComplex_co) -> _ComplexSeries: ...
    @overload
    def __call__(self, /, pol: _SeriesLikeCoef_co) -> _ObjectSeries: ...

@type_check_only
class _FuncPow(Protocol):
    @overload
    def __call__(self, /, c: _SeriesLikeFloat_co, pow: _IntLike_co, maxpower: _IntLike_co | None = ...) -> _FloatSeries: ...
    @overload
    def __call__(self, /, c: _SeriesLikeComplex_co, pow: _IntLike_co, maxpower: _IntLike_co | None = ...) -> _ComplexSeries: ...
    @overload
    def __call__(self, /, c: _SeriesLikeCoef_co, pow: _IntLike_co, maxpower: _IntLike_co | None = ...) -> _ObjectSeries: ...
