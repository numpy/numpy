import decimal
import fractions
import numbers
import sys
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    SupportsIndex,
    SupportsInt,
    TypeAlias,
    TypeVar,
    final,
    overload,
)

import numpy as np
import numpy.typing as npt
from numpy._typing import (
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _NestedSequence,
)

if sys.version_info >= (3, 11):
    from typing import LiteralString
elif TYPE_CHECKING:
    from typing_extensions import LiteralString
else:
    LiteralString: TypeAlias = str

_T = TypeVar("_T", bound=object)
_Tuple2: TypeAlias = tuple[_T, _T]

_V = TypeVar("_V")
_V_co = TypeVar("_V_co", covariant=True)
_Self = TypeVar("_Self", bound=object)

class _SupportsLenAndGetItem(Protocol[_V_co]):
    def __len__(self, /) -> int: ...
    def __getitem__(self, i: int, /) -> _V_co: ...

class _SimpleSequence(Protocol[_V_co]):
    def __len__(self, /) -> int: ...
    @overload
    def __getitem__(self, i: int, /) -> _V_co: ...
    @overload
    def __getitem__(self: _Self, ii: slice, /) -> _Self: ...

_SCT = TypeVar("_SCT", bound=np.number[Any] | np.object_)
_SCT_co = TypeVar("_SCT_co", bound=np.number[Any] | np.object_, covariant=True)

class _SupportsArray(Protocol[_SCT_co]):
    def __array__(self ,) -> npt.NDArray[_SCT_co]: ...

_Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[_SCT]]
_Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_SCT]]

_IntArray1D: TypeAlias = _Array1D[np.integer[Any]]
_IntArrayND: TypeAlias = npt.NDArray[np.integer[Any]]
_FloatArray1D: TypeAlias = _Array1D[np.floating[Any]]
_FloatArrayND: TypeAlias = npt.NDArray[np.floating[Any]]
_ComplexArray1D: TypeAlias = _Array1D[np.complexfloating[Any, Any]]
_ComplexArrayND: TypeAlias = npt.NDArray[np.complexfloating[Any, Any]]
_ObjectArray1D: TypeAlias = _Array1D[np.object_]
_ObjectArrayND: TypeAlias = npt.NDArray[np.object_]

_Array1: TypeAlias = np.ndarray[tuple[Literal[1]], np.dtype[_SCT]]
_Array2: TypeAlias = np.ndarray[tuple[Literal[2]], np.dtype[_SCT]]
_Line: TypeAlias = np.ndarray[tuple[Literal[1, 2]], np.dtype[_SCT]]

_CoefArray1D: TypeAlias = _Array1D[np.inexact[Any] | np.object_]
_CoefArrayND: TypeAlias = npt.NDArray[np.inexact[Any] | np.object_]

_AnyIntArg: TypeAlias = SupportsInt | SupportsIndex

_AnyIntScalar: TypeAlias = int | np.integer[Any]
_AnyRealScalar: TypeAlias = float | np.floating[Any] | np.integer[Any]
_AnyComplexScalar: TypeAlias = complex | np.number[Any]
_AnyObjectScalar: TypeAlias = (
    np.object_
    | fractions.Fraction
    | decimal.Decimal
    | numbers.Complex
)
_AnyScalar: TypeAlias = _AnyComplexScalar | _AnyObjectScalar

_AnyIntSeries1D: TypeAlias = (
    _SupportsArray[np.integer[Any]]
    | _SupportsLenAndGetItem[_AnyIntScalar]
)
_AnyRealSeries1D: TypeAlias = (
    _SupportsArray[np.integer[Any] | np.floating[Any]]
    | _SupportsLenAndGetItem[_AnyRealScalar]
)
_AnyComplexSeries1D: TypeAlias = (
    _SupportsArray[np.number[Any]]
    | _SupportsLenAndGetItem[_AnyComplexScalar]
)
_AnyObjectSeries1D: TypeAlias = (
    _SupportsArray[np.object_]
    | _SupportsLenAndGetItem[_AnyObjectScalar]
)
_AnySeries1D: TypeAlias = (
    _SupportsArray[np.number[Any] | np.object_]
    | _SupportsLenAndGetItem[object]
)

_AnyIntSeriesND: TypeAlias = (
    int
    | _SupportsArray[np.integer[Any]]
    | _NestedSequence[int]
    | _NestedSequence[_SupportsArray[np.integer[Any]]]
)

_AnyRealSeriesND: TypeAlias = (
    float
    | _SupportsArray[np.integer[Any] | np.floating[Any]]
    | _NestedSequence[float]
    | _NestedSequence[_SupportsArray[np.integer[Any] | np.floating[Any]]]
)
_AnyComplexSeriesND: TypeAlias = (
    complex
    | _SupportsArray[np.number[Any]]
    | _NestedSequence[complex]
    | _NestedSequence[_SupportsArray[np.number[Any]]]
)
_AnyObjectSeriesND: TypeAlias = (
    _AnyObjectScalar
    | _SupportsArray[np.object_]
    | _NestedSequence[_AnyObjectScalar]
    | _NestedSequence[_SupportsArray[np.object_]]
)
_AnySeriesND: TypeAlias = (
    _AnyScalar
    | _SupportsArray[np.number[Any] | np.object_]
    | _NestedSequence[object]
    | _NestedSequence[_SupportsArray[np.number[Any] | np.object_]]
)

_Name_co = TypeVar("_Name_co", bound=LiteralString, covariant=True)

class _Named(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

@final
class _FuncLine(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(self, /, off: _SCT, scl: _SCT) -> _Line[_SCT]: ...  # type: ignore[overload-overlap]
    @overload
    def __call__(self, /, off: int, scl: int) -> _Line[np.int_] : ...  # type: ignore[overload-overlap]
    @overload
    def __call__(self, /, off: float, scl: float) -> _Line[np.float64]: ...
    @overload
    def __call__(
        self, /,
        off: complex,
        scl: complex,
    ) -> _Line[np.complex128]: ...
    @overload
    def __call__(self, /, off: object, scl: object) -> _Line[np.object_]: ...

@final
class _FuncFromRoots(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(self, /, roots: _AnyRealSeries1D) -> _FloatArray1D: ...  # type: ignore[overload-overlap]
    @overload
    def __call__(self, /, roots: _AnyComplexSeries1D) -> _ComplexArray1D: ...
    @overload
    def __call__(self, /, roots: _AnySeries1D) -> _ObjectArray1D: ...

@final
class _FuncBinOp(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c1: _AnyRealSeries1D,
        c2: _AnyRealSeries1D,
    ) -> _FloatArray1D: ...
    @overload
    def __call__(
        self, /,
        c1: _AnyComplexSeries1D,
        c2: _AnyComplexSeries1D,
    ) -> _ComplexArray1D: ...
    @overload
    def __call__(
        self, /,
        c1: _AnySeries1D,
        c2: _AnySeries1D,
    ) -> _ObjectArray1D: ...

@final
class _FuncUnOp(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(self, /, c: _AnyRealSeries1D) -> _FloatArray1D: ...  # type: ignore[overload-overlap]
    @overload
    def __call__(self, /, c: _AnyComplexSeries1D) -> _ComplexArray1D: ...
    @overload
    def __call__(self, /, c: _AnySeries1D) -> _ObjectArray1D: ...

@final
class _FuncPoly2Ortho(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(self, /, pol: _AnyRealSeries1D) -> _FloatArray1D: ...  # type: ignore[overload-overlap]
    @overload
    def __call__(self, /, pol: _AnyComplexSeries1D) -> _ComplexArray1D: ...
    @overload
    def __call__(self, /, pol: _AnySeries1D) -> _ObjectArray1D: ...

@final
class _FuncPow(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyRealSeries1D,
        pow: _AnyIntArg,
        maxpower: None | _AnyIntArg = ...,
    ) -> _FloatArray1D: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeries1D,
        pow: _AnyIntArg,
        maxpower: None | _AnyIntArg = ...,
    ) -> _ComplexArray1D: ...
    @overload
    def __call__(
        self, /,
        c: _AnySeries1D,
        pow: _AnyIntArg,
        maxpower: None | _AnyIntArg = ...,
    ) -> _ObjectArray1D: ...

@final
class _FuncDer(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyRealSeriesND,
        m: SupportsIndex = ...,
        scl: _AnyComplexScalar = ...,
        axis: SupportsIndex = ...,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeriesND,
        m: SupportsIndex = ...,
        scl: _AnyComplexScalar = ...,
        axis: SupportsIndex = ...,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        c: _AnyObjectSeriesND,
        m: SupportsIndex = ...,
        scl: _AnyComplexScalar = ...,
        axis: SupportsIndex = ...,
    ) -> _ObjectArrayND: ...

@final
class _FuncInteg(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyRealSeriesND,
        m: SupportsIndex = ...,
        k: _AnyComplexScalar | _SupportsLenAndGetItem[_AnyComplexScalar] = ...,
        lbnd: _AnyComplexScalar = ...,
        scl: _AnyComplexScalar = ...,
        axis: SupportsIndex = ...,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeriesND,
        m: SupportsIndex = ...,
        k: _AnyComplexScalar | _SupportsLenAndGetItem[_AnyComplexScalar] = ...,
        lbnd: _AnyComplexScalar = ...,
        scl: _AnyComplexScalar = ...,
        axis: SupportsIndex = ...,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        c: _AnyObjectSeriesND,
        m: SupportsIndex = ...,
        k: _AnyComplexScalar | _SupportsLenAndGetItem[_AnyScalar] = ...,
        lbnd: _AnyComplexScalar = ...,
        scl: _AnyComplexScalar = ...,
        axis: SupportsIndex = ...,
    ) -> _ObjectArrayND: ...

_AnyRealRoots: TypeAlias = (
    _Array1D[np.floating[Any] | np.integer[Any]]
    | Sequence[_AnyRealScalar]
)
_AnyComplexRoots: TypeAlias = (
    _Array1D[np.number[Any]]
    | Sequence[_AnyComplexScalar]
)
_AnyObjectRoots: TypeAlias = _ObjectArray1D | Sequence[_AnyObjectScalar]
_AnyRoots: TypeAlias = _ObjectArray1D | Sequence[_AnyScalar]

_AnyRealPoints: TypeAlias = (
    npt.NDArray[np.floating[Any] | np.integer[Any]]
    | tuple[_AnyRealSeriesND, ...]
    | list[_AnyRealSeriesND]
)
_AnyComplexPoints: TypeAlias = (
    npt.NDArray[np.number[Any]]
    | tuple[_AnyComplexSeriesND, ...]
    | list[_AnyComplexSeriesND]
)
_AnyObjectPoints: TypeAlias = (
    _ObjectArrayND
    | tuple[_AnyObjectSeriesND, ...]
    | list[_AnyObjectSeriesND]
)
_AnyPoints: TypeAlias = (
    npt.NDArray[np.number[Any] | np.object_]
    | tuple[_AnySeriesND, ...]
    | list[_AnySeriesND]
)

@final
class _FuncValFromRoots(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealScalar,
        r: _AnyRealScalar,
        tensor: bool = ...,
    ) -> np.floating[Any]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexScalar,
        r: _AnyComplexScalar,
        tensor: bool = ...,
    ) -> np.complexfloating[Any, Any]: ...
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyScalar,
        r: _AnyScalar,
        tensor: bool = ...,
    ) -> object: ...
    @overload
    def __call__(
        self, /,
        x: _AnyRealScalar | _AnyRealPoints,
        r: _AnyRealSeriesND,
        tensor: bool = ...,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexScalar | _AnyComplexPoints,
        r: _AnyComplexSeriesND,
        tensor: bool = ...,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyScalar | _AnyPoints,
        r: _AnySeriesND,
        tensor: bool = ...,
    ) -> _ObjectArrayND: ...

@final
class _FuncVal(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealScalar,
        c: _AnyRealRoots,
        tensor: bool = ...,
    ) -> np.floating[Any]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexScalar,
        c: _AnyComplexRoots,
        tensor: bool = ...,
    ) -> np.complexfloating[Any, Any]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyScalar,
        c: _AnyObjectRoots,
        tensor: bool = ...,
    ) -> object: ...
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealPoints,
        c: _AnyRealSeriesND,
        tensor: bool = ...,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexPoints,
        c: _AnyComplexSeriesND,
        tensor: bool = ...,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyPoints,
        c: _AnySeriesND,
        tensor: bool = ...,
    ) -> _ObjectArrayND: ...

@final
class _FuncVal2D(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealScalar,
        y: _AnyRealScalar,
        c: _AnyRealRoots,
    ) -> np.floating[Any]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexScalar,
        y: _AnyComplexScalar,
        c: _AnyComplexRoots,
    ) -> np.complexfloating[Any, Any]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyScalar,
        y: _AnyScalar,
        c: _AnyRoots,
    ) -> object: ...
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealPoints,
        y: _AnyRealPoints,
        c: _AnyRealSeriesND,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexPoints,
        y: _AnyComplexPoints,
        c: _AnyComplexSeriesND,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyPoints,
        y: _AnyPoints,
        c: _AnySeriesND,
    ) -> _ObjectArrayND: ...

@final
class _FuncVal3D(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealScalar,
        y: _AnyRealScalar,
        z: _AnyRealScalar,
        c: _AnyRealRoots
    ) -> np.floating[Any]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexScalar,
        y: _AnyComplexScalar,
        z: _AnyComplexScalar,
        c: _AnyComplexRoots,
    ) -> np.complexfloating[Any, Any]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyScalar,
        y: _AnyScalar,
        z: _AnyScalar,
        c: _AnyRoots,
    ) -> object: ...
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealPoints,
        y: _AnyRealPoints,
        z: _AnyRealPoints,
        c: _AnyRealSeriesND,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexPoints,
        y: _AnyComplexPoints,
        z: _AnyComplexPoints,
        c: _AnyComplexSeriesND,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyPoints,
        y: _AnyPoints,
        z: _AnyPoints,
        c: _AnySeriesND,
    ) -> _ObjectArrayND: ...

_AnyValF: TypeAlias = Callable[
    [npt.ArrayLike, npt.ArrayLike, bool],
    _CoefArrayND,
]

@final
class _FuncValND(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self,
        val_f: _AnyValF,
        c: _AnyRealRoots,
        /,
        *args: _AnyRealScalar,
    ) -> np.floating[Any]: ...
    @overload
    def __call__(
        self,
        val_f: _AnyValF,
        c: _AnyComplexRoots,
        /,
        *args: _AnyComplexScalar,
    ) -> np.complexfloating[Any, Any]: ...
    @overload
    def __call__(
        self,
        val_f: _AnyValF,
        c: _AnyObjectRoots,
        /,
        *args: _AnyObjectScalar,
    ) -> object: ...
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self,
        val_f: _AnyValF,
        c: _AnyRealSeriesND,
        /,
        *args: _AnyRealPoints,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self,
        val_f: _AnyValF,
        c: _AnyComplexSeriesND,
        /,
        *args: _AnyComplexPoints,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self,
        val_f: _AnyValF,
        c: _AnySeriesND,
        /,
        *args: _AnyObjectPoints,
    ) -> _ObjectArrayND: ...

@final
class _FuncVander(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealSeriesND,
        deg: SupportsIndex,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexSeriesND,
        deg: SupportsIndex,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnySeriesND,
        deg: SupportsIndex,
    ) -> _ObjectArrayND: ...
    @overload
    def __call__(
        self, /,
        x: npt.ArrayLike,
        deg: SupportsIndex,
    ) -> _CoefArrayND: ...

_AnyDegrees: TypeAlias = _SupportsLenAndGetItem[SupportsIndex]

@final
class _FuncVander2D(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealSeriesND,
        y: _AnyRealSeriesND,
        deg: _AnyDegrees,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexSeriesND,
        y: _AnyComplexSeriesND,
        deg: _AnyDegrees,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnySeriesND,
        y: _AnySeriesND,
        deg: _AnyDegrees,
    ) -> _ObjectArrayND: ...
    @overload
    def __call__(
        self, /,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        deg: _AnyDegrees,
    ) -> _CoefArrayND: ...

@final
class _FuncVander3D(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealSeriesND,
        y: _AnyRealSeriesND,
        z: _AnyRealSeriesND,
        deg: _AnyDegrees,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexSeriesND,
        y: _AnyComplexSeriesND,
        z: _AnyComplexSeriesND,
        deg: _AnyDegrees,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnySeriesND,
        y: _AnySeriesND,
        z: _AnySeriesND,
        deg: _AnyDegrees,
    ) -> _ObjectArrayND: ...
    @overload
    def __call__(
        self, /,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        z: npt.ArrayLike,
        deg: _AnyDegrees,
    ) -> _CoefArrayND: ...

# keep in sync with the broadest overload of `._FuncVander`
_AnyFuncVander: TypeAlias = Callable[
    [npt.ArrayLike, SupportsIndex],
    _CoefArrayND,
]

@final
class _FuncVanderND(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        vander_fs: _SupportsLenAndGetItem[_AnyFuncVander],
        points: _SupportsLenAndGetItem[_ArrayLikeFloat_co],
        degrees: _SupportsLenAndGetItem[SupportsIndex],
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        vander_fs: _SupportsLenAndGetItem[_AnyFuncVander],
        points: _SupportsLenAndGetItem[_ArrayLikeComplex_co],
        degrees: _SupportsLenAndGetItem[SupportsIndex],
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        vander_fs: _SupportsLenAndGetItem[_AnyFuncVander],
        points: _SupportsLenAndGetItem[
            _ArrayLikeObject_co | _ArrayLikeComplex_co,
        ],
        degrees: _SupportsLenAndGetItem[SupportsIndex],
    ) -> _ObjectArrayND: ...
    @overload
    def __call__(
        self, /,
        vander_fs: _SupportsLenAndGetItem[_AnyFuncVander],
        points: _SupportsLenAndGetItem[npt.ArrayLike],
        degrees: _SupportsLenAndGetItem[SupportsIndex],
    ) -> _CoefArrayND: ...

@final
class _FuncFit(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyRealSeries1D,
        y: _AnyRealSeriesND,
        deg: int | _AnyIntSeries1D,
        rcond: None | float = ...,
        full: Literal[False] = ...,
        w: None | _AnyRealSeries1D = ...,
    ) -> _FloatArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexSeries1D,
        y: _AnyComplexSeriesND,
        deg: int | _AnyIntSeries1D,
        rcond: None | float = ...,
        full: Literal[False] = ...,
        w: None | _AnyComplexSeriesND = ...,
    ) -> _ComplexArrayND: ...
    @overload
    def __call__(
        self, /,
        x: _AnySeries1D,
        y: _AnySeriesND,
        deg: int | _AnyIntSeries1D,
        rcond: None | float = ...,
        full: Literal[False] = ...,
        w: None | _AnySeries1D = ...,
    ) -> _ObjectArrayND: ...
    @overload
    def __call__(
        self,
        x: _AnySeries1D,
        y: _AnySeriesND,
        deg: int | _AnyIntSeries1D,
        rcond: None | float,
        full: Literal[True],
        /,
        w: None | _AnySeries1D = ...,
    ) -> tuple[_CoefArrayND, Sequence[np.inexact[Any] | np.int32]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnySeries1D,
        y: _AnySeriesND,
        deg: int | _AnyIntSeries1D,
        rcond: None | float = ...,
        *,
        full: Literal[True],
        w: None | _AnySeries1D = ...,
    ) -> tuple[_CoefArrayND, Sequence[np.inexact[Any] | np.int32]]: ...

@final
class _FuncRoots(_Named[_Name_co], Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyRealSeries1D,
    ) -> _Array1D[np.float64]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeries1D,
    ) -> _Array1D[np.complex128]: ...
    @overload
    def __call__(self, /, c: _AnySeries1D) -> _ObjectArray1D: ...

@final
class _FuncCompanion(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyRealSeries1D,
    ) -> _Array2D[np.float64]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeries1D,
    ) -> _Array2D[np.complex128]: ...
    @overload
    def __call__(self, /, c: _AnySeries1D) -> _Array2D[np.object_]: ...

@final
class _FuncGauss(_Named[_Name_co], Protocol[_Name_co]):
    def __call__(
        self, /,
        deg: SupportsIndex,
    ) -> _Tuple2[_Array1D[np.float64]]: ...

@final
class _FuncWeight(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyRealSeriesND,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeriesND,
    ) -> npt.NDArray[np.complex128]: ...
    @overload
    def __call__(self, /, c: _AnySeriesND) -> _ObjectArrayND: ...

_N_pts = TypeVar("_N_pts", bound=int)

@final
class _FuncPts(_Named[_Name_co], Protocol[_Name_co]):
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        npts: _N_pts,
    ) -> np.ndarray[tuple[_N_pts], np.dtype[np.float64]]: ...
    @overload
    def __call__(self, /, npts: _AnyIntArg) -> _Array1D[np.float64]: ...
