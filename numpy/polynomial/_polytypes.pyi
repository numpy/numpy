import decimal
import fractions
import numbers
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    Literal,
    Protocol,
    SupportsComplex,
    SupportsFloat,
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
    _SupportsArray,
)

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

_SCT = TypeVar("_SCT", bound=np.generic)
_Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[_SCT]]
_Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_SCT]]

_CoefScalarType: TypeAlias = np.number[Any] | np.object_
_CoefArray1D: TypeAlias = _Array1D[_CoefScalarType]
_CoefArrayND: TypeAlias = npt.NDArray[_CoefScalarType]

class _SupportsBool(Protocol):
    def __bool__(self, /) -> bool: ...

_AnyFloatScalar: TypeAlias = float | np.floating[Any] | np.integer[Any]
_AnyComplexScalar: TypeAlias = complex | np.complexfloating[Any, Any]
_AnyNumberScalar: TypeAlias = complex | np.number[Any]
_AnyObjectScalar: TypeAlias = (
    fractions.Fraction
    | decimal.Decimal
    | numbers.Complex
    | np.object_
)
_AnyScalar: TypeAlias = _AnyNumberScalar | _AnyObjectScalar
_AnyInt: TypeAlias = SupportsInt | SupportsIndex

_AnyFloatSeries1D: TypeAlias = (
    _SupportsArray[np.dtype[np.floating[Any] | np.integer[Any]]]
    | _SupportsLenAndGetItem[float | np.floating[Any] | np.integer[Any]]
)
_AnyComplexSeries1D: TypeAlias = (
    npt.NDArray[np.complexfloating[Any, Any]]
    | _SupportsArray[np.dtype[np.complexfloating[Any, Any]]]
    | _SupportsLenAndGetItem[_AnyComplexScalar]
)
_AnyNumberSeries1D: TypeAlias = (
    npt.NDArray[np.number[Any]]
    | _SupportsArray[np.dtype[np.number[Any]]]
    | _SupportsLenAndGetItem[_AnyNumberScalar]
)
_AnyObjectSeries1D: TypeAlias = (
    npt.NDArray[np.object_]
    | _SupportsLenAndGetItem[_AnyObjectScalar]
)
_AnySeries1D: TypeAlias = (
    npt.NDArray[_CoefScalarType]
    | _SupportsLenAndGetItem[_AnyScalar | object]
)

_AnyFloatSeriesND: TypeAlias = (
    _AnyFloatScalar
    | _SupportsArray[np.dtype[np.floating[Any] | np.integer[Any]]]
    | _NestedSequence[float | np.floating[Any] | np.integer[Any]]
)
_AnyComplexSeriesND: TypeAlias = (
    _AnyComplexScalar
    | _SupportsArray[np.dtype[np.number[Any]]]
    | _NestedSequence[complex | np.number[Any]]
)
_AnyObjectSeriesND: TypeAlias = (
    _AnyObjectScalar
    | _SupportsArray[np.dtype[np.object_]]
    | _NestedSequence[_AnyObjectScalar]
)
_AnySeriesND: TypeAlias = (
    _AnyScalar
    | _SupportsArray[np.dtype[_CoefScalarType]]
    | _NestedSequence[SupportsComplex | SupportsFloat]
)

_SCT_domain = TypeVar("_SCT_domain", np.float64, np.complex128, np.object_)
_Interval: TypeAlias = np.ndarray[tuple[Literal[2]], np.dtype[_SCT_domain]]

_T = TypeVar("_T", bound=object)
_Tuple2: TypeAlias = tuple[_T, _T]

_SCT_number = TypeVar("_SCT_number", bound=_CoefScalarType)
_Array1: TypeAlias = np.ndarray[tuple[Literal[1]], np.dtype[_SCT]]
_Array2: TypeAlias = np.ndarray[tuple[Literal[2]], np.dtype[_SCT]]
_Line: TypeAlias = _Array1[_SCT_number] | _Array2[_SCT_number]

_Name_co = TypeVar("_Name_co", bound=str, covariant=True)

@final
class _FuncLine(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        off: _SCT_number,
        scl: _SCT_number,
    ) -> _Line[_SCT_number]: ...
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        off: int,
        scl: int,
    ) -> _Line[np.int_] : ...
    @overload
    def __call__(
        self, /,
        off: float,
        scl: float,
    ) -> _Line[np.float64]: ...
    @overload
    def __call__(
        self, /,
        off: complex,
        scl: complex,
    ) -> _Line[np.complex128]: ...
    @overload
    def __call__(
        self, /,
        off: _AnyObjectScalar,
        scl: _AnyObjectScalar,
    ) -> _Line[np.object_]: ...

@final
class _FuncFromRoots(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        roots: _AnyFloatSeries1D,
    ) -> _Array1D[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        roots: _AnyComplexSeries1D,
    ) -> _Array1D[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        roots: _AnyObjectSeries1D,
    ) -> _Array1D[np.object_]: ...
    @overload
    def __call__(self, /, roots: _AnySeries1D) -> _CoefArray1D: ...

@final
class _FuncBinOp(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c1: _AnyFloatSeries1D,
        c2: _AnyFloatSeries1D,
    ) -> _Array1D[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        c1: _AnyComplexSeries1D,
        c2: _AnyComplexSeries1D,
    ) -> _Array1D[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        c1: _AnyObjectSeries1D,
        c2: _AnyObjectSeries1D,
    ) -> _Array1D[np.object_]: ...
    @overload
    def __call__(
        self, /,
        c1: _AnySeries1D,
        c2: _AnySeries1D,
    ) -> _CoefArray1D: ...

@final
class _FuncUnOp(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyFloatSeries1D,
    ) -> _Array1D[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeries1D,
    ) -> _Array1D[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(self, /, c: _AnyObjectSeries1D) -> _Array1D[np.object_]: ...
    @overload
    def __call__(self, /, c: _AnySeries1D) -> _CoefArray1D: ...

@final
class _FuncPoly2Ortho(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        pol: _AnyFloatSeries1D,
    ) -> _Array1D[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        pol: _AnyComplexSeries1D,
    ) -> _Array1D[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(self, /, pol: _AnyObjectSeries1D) -> _Array1D[np.object_]: ...
    @overload
    def __call__(self, /, pol: _AnySeries1D) -> _CoefArray1D: ...

@final
class _FuncPow(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyFloatSeries1D,
        pow: _AnyInt,
        maxpower: None | _AnyInt = ...,
    ) -> _Array1D[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeries1D,
        pow: _AnyInt,
        maxpower: None | _AnyInt = ...,
    ) -> _Array1D[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyObjectSeries1D,
        pow: _AnyInt,
        maxpower: None | _AnyInt = ...,
    ) -> _Array1D[np.object_]: ...
    @overload
    def __call__(
        self, /,
        c: _AnySeries1D,
        pow: _AnyInt,
        maxpower: None | _AnyInt = ...,
    ) -> _CoefArray1D: ...


@final
class _FuncDer(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyFloatSeriesND,
        m: SupportsIndex = ...,
        scl: _AnyNumberScalar = ...,
        axis: SupportsIndex = ...,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeriesND,
        m: SupportsIndex = ...,
        scl: _AnyNumberScalar = ...,
        axis: SupportsIndex = ...,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyObjectSeriesND,
        m: SupportsIndex = ...,
        scl: _AnyNumberScalar = ...,
        axis: SupportsIndex = ...,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self, /,
        c: _AnySeriesND,
        m: SupportsIndex = ...,
        scl: _AnyNumberScalar = ...,
        axis: SupportsIndex = ...,
    ) -> _CoefArrayND: ...


@final
class _FuncInteg(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyFloatSeriesND,
        m: SupportsIndex = ...,
        k: _AnyNumberScalar | _SupportsLenAndGetItem[_AnyNumberScalar] = ...,
        lbnd: _AnyNumberScalar = ...,
        scl: _AnyNumberScalar = ...,
        axis: SupportsIndex = ...,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeriesND,
        m: SupportsIndex = ...,
        k: _AnyNumberScalar | _SupportsLenAndGetItem[_AnyNumberScalar] = ...,
        lbnd: _AnyNumberScalar = ...,
        scl: _AnyNumberScalar = ...,
        axis: SupportsIndex = ...,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyObjectSeriesND,
        m: SupportsIndex = ...,
        k: _AnyNumberScalar | _SupportsLenAndGetItem[_AnyNumberScalar] = ...,
        lbnd: _AnyNumberScalar = ...,
        scl: _AnyNumberScalar = ...,
        axis: SupportsIndex = ...,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self, /,
        c: _AnySeriesND,
        m: SupportsIndex = ...,
        k: _AnyNumberScalar | _SupportsLenAndGetItem[_AnyNumberScalar] = ...,
        lbnd: _AnyNumberScalar = ...,
        scl: _AnyNumberScalar = ...,
        axis: SupportsIndex = ...,
    ) -> _CoefArrayND: ...


_AnyFloatRoots: TypeAlias = (
    _Array1D[np.floating[Any] | np.integer[Any]]
    | Sequence[_AnyFloatScalar]
)
_AnyComplexRoots: TypeAlias = (
    _Array1D[np.number[Any]]
    | Sequence[_AnyComplexScalar]
)
_AnyObjectRoots: TypeAlias = (
    _Array1D[np.object_]
    | Sequence[_AnyObjectScalar]
)

_AnyFloatPoints: TypeAlias = (
    npt.NDArray[np.floating[Any] | np.integer[Any]]
    | tuple[_AnyFloatSeriesND, ...]
    | list[_AnyFloatSeriesND]
)
_AnyComplexPoints: TypeAlias = (
    npt.NDArray[np.complexfloating[Any, Any]]
    | tuple[_AnyComplexSeriesND, ...]
    | list[_AnyComplexSeriesND]
)
_AnyObjectPoints: TypeAlias = (
    npt.NDArray[np.object_]
    | tuple[_AnyObjectSeriesND, ...]
    | list[_AnyObjectSeriesND]
)
_AnyPoints: TypeAlias = (
    _CoefArrayND
    | tuple[_AnySeriesND, ...]
    | list[_AnySeriesND]
)

@final
class _FuncValFromRoots(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyFloatScalar,
        r: _AnyFloatScalar,
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
        x: _AnyObjectScalar,
        r: _AnyObjectScalar,
        tensor: bool = ...,
    ) -> object: ...
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyFloatScalar | _AnyFloatPoints,
        r: _AnyFloatSeriesND,
        tensor: bool = ...,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexScalar | _AnyComplexPoints,
        r: _AnyComplexSeriesND,
        tensor: bool = ...,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyScalar | _AnyObjectPoints | _AnyComplexPoints,
        r: _AnyObjectSeriesND,
        tensor: bool = ...,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyScalar | _AnyPoints,
        r: _AnySeriesND,
        tensor: bool = ...,
    ) -> _CoefArrayND: ...

@final
class _FuncVal(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyFloatScalar,
        c: _AnyFloatRoots,
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
        x: _AnyObjectScalar,
        c: _AnyObjectRoots,
        tensor: bool = ...,
    ) -> object: ...
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyFloatPoints,
        c: _AnyFloatSeriesND,
        tensor: bool = ...,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexPoints,
        c: _AnyComplexSeriesND,
        tensor: bool = ...,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyObjectPoints,
        c: _AnyObjectSeries1D | _AnyComplexSeriesND,
        tensor: bool = ...,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyPoints,
        c: _AnySeriesND,
        tensor: bool = ...,
    ) -> _CoefArrayND: ...

@final
class _FuncVal2D(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyFloatScalar,
        y: _AnyFloatScalar,
        c: _AnyFloatRoots,
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
        x: _AnyObjectScalar,
        y: _AnyObjectScalar,
        c: _AnyObjectRoots,
    ) -> object: ...
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyFloatPoints,
        y: _AnyFloatPoints,
        c: _AnyFloatSeriesND,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexPoints,
        y: _AnyComplexPoints,
        c: _AnyComplexSeriesND,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyObjectPoints,
        y: _AnyObjectPoints,
        c: _AnyObjectSeries1D | _AnyComplexSeriesND,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyPoints,
        y: _AnyPoints,
        c: _AnySeriesND,
    ) -> _CoefArrayND: ...

@final
class _FuncVal3D(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyFloatScalar,
        y: _AnyFloatScalar,
        z: _AnyFloatScalar,
        c: _AnyFloatRoots
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
        x: _AnyObjectScalar,
        y: _AnyObjectScalar,
        z: _AnyObjectScalar,
        c: _AnyObjectRoots,
    ) -> object: ...
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyFloatPoints,
        y: _AnyFloatPoints,
        z: _AnyFloatPoints,
        c: _AnyFloatSeriesND,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexPoints,
        y: _AnyComplexPoints,
        z: _AnyComplexPoints,
        c: _AnyComplexSeriesND,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyObjectPoints,
        y: _AnyObjectPoints,
        z: _AnyObjectPoints,
        c: _AnyObjectSeries1D | _AnyComplexSeriesND,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyPoints,
        y: _AnyPoints,
        z: _AnyPoints,
        c: _AnySeriesND,
    ) -> _CoefArrayND: ...

_AnyValF: TypeAlias = Callable[
    [npt.ArrayLike, npt.ArrayLike, bool],
    _CoefArrayND,
]

@final
class _FuncValND(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self,
        val_f: _AnyValF,
        c: _AnyFloatRoots,
        /,
        *args: _AnyFloatScalar,
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
        c: _AnyFloatSeriesND,
        /,
        *args: _AnyFloatPoints,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self,
        val_f: _AnyValF,
        c: _AnyComplexSeriesND,
        /,
        *args: _AnyComplexPoints,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self,
        val_f: _AnyValF,
        c: _AnyObjectSeries1D | _AnyComplexSeriesND,
        /,
        *args: _AnyObjectPoints,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self,
        val_f: _AnyValF,
        c: _AnySeriesND,
        /,
        *args: _AnyPoints,
    ) -> _CoefArrayND: ...

@final
class _FuncVander(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _ArrayLikeFloat_co,
        deg: SupportsIndex,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _ArrayLikeComplex_co,
        deg: SupportsIndex,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _ArrayLikeObject_co,
        deg: SupportsIndex,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self, /,
        x: npt.ArrayLike,
        deg: SupportsIndex,
    ) -> _CoefArrayND: ...

_AnyDegrees: TypeAlias = _SupportsLenAndGetItem[SupportsIndex]

@final
class _FuncVander2D(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co,
        deg: _AnyDegrees,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _ArrayLikeComplex_co,
        y: _ArrayLikeComplex_co,
        deg: _AnyDegrees,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _ArrayLikeObject_co,
        y: _ArrayLikeObject_co,
        deg: _AnyDegrees,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self, /,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        deg: _AnyDegrees,
    ) -> _CoefArrayND: ...

@final
class _FuncVander3D(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co,
        z: _ArrayLikeFloat_co,
        deg: _AnyDegrees,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _ArrayLikeComplex_co,
        z: _ArrayLikeComplex_co,
        deg: _AnyDegrees,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _ArrayLikeObject_co,
        y: _ArrayLikeObject_co,
        z: _ArrayLikeObject_co,
        deg: _AnyDegrees,
    ) -> npt.NDArray[np.object_]: ...
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
class _FuncVanderND(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        vander_fs: _SupportsLenAndGetItem[_AnyFuncVander],
        points: _SupportsLenAndGetItem[_ArrayLikeFloat_co],
        degrees: _SupportsLenAndGetItem[SupportsIndex],
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        vander_fs: _SupportsLenAndGetItem[_AnyFuncVander],
        points: _SupportsLenAndGetItem[_ArrayLikeComplex_co],
        degrees: _SupportsLenAndGetItem[SupportsIndex],
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        vander_fs: _SupportsLenAndGetItem[_AnyFuncVander],
        points: _SupportsLenAndGetItem[_ArrayLikeObject_co],
        degrees: _SupportsLenAndGetItem[SupportsIndex],
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self, /,
        vander_fs: _SupportsLenAndGetItem[_AnyFuncVander],
        points: _SupportsLenAndGetItem[npt.ArrayLike],
        degrees: _SupportsLenAndGetItem[SupportsIndex],
    ) -> _CoefArrayND: ...

@final
class _FuncFit(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        x: _AnyFloatSeries1D,
        y: _AnyFloatSeriesND,
        deg: _ArrayLikeInt_co,
        rcond: None | float = ...,
        full: Literal[False] = ...,
        w: None | _AnyFloatSeries1D = ...,
    ) -> npt.NDArray[np.floating[Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnyComplexSeries1D,
        y: _AnyComplexSeriesND,
        deg: _ArrayLikeInt_co,
        rcond: None | float = ...,
        full: Literal[False] = ...,
        w: None | _AnyComplexSeriesND = ...,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
    @overload
    def __call__(
        self, /,
        x: _AnySeries1D,
        y: _AnySeriesND,
        deg: _ArrayLikeInt_co,
        rcond: None | float = ...,
        full: Literal[False] = ...,
        w: None | _AnySeries1D = ...,
    ) -> _CoefArrayND: ...

    @overload
    def __call__(
        self,
        x: _AnySeries1D,
        y: _AnySeriesND,
        deg: _ArrayLikeInt_co,
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
        deg: _ArrayLikeInt_co,
        rcond: None | float = ...,
        *,
        full: Literal[True],
        w: None | _AnySeries1D = ...,
    ) -> tuple[_CoefArrayND, Sequence[np.inexact[Any] | np.int32]]: ...

@final
class _FuncRoots(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyFloatSeries1D,
    ) -> _Array1D[np.float64]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeries1D,
    ) -> _Array1D[np.complex128]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyObjectSeries1D,
    ) -> _Array1D[np.object_]: ...
    @overload
    def __call__(
        self,  /,
        c: _AnySeries1D,
    ) -> _Array1D[np.float64 | np.complex128 | np.object_]: ...

@final
class _FuncCompanion(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyFloatSeries1D,
    ) -> _Array2D[np.float64]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeries1D,
    ) -> _Array2D[np.complex128]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyObjectSeries1D,
    ) -> _Array2D[np.object_]: ...
    @overload
    def __call__(
        self,  /,
        c: _AnySeries1D,
    ) -> _Array2D[np.float64 | np.complex128 | np.object_]: ...

@final
class _FuncGauss(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    def __call__(self, /, SupportsIndex) -> _Tuple2[_Array1D[np.float64]]: ...

@final
class _FuncWeight(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        c: _AnyFloatSeriesND,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyComplexSeriesND,
    ) -> npt.NDArray[np.complex128]: ...
    @overload
    def __call__(
        self, /,
        c: _AnyObjectSeriesND,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self,  /,
        c: _AnySeriesND,
    ) -> npt.NDArray[np.float64 | np.complex128 | np.object_]: ...

_N_pts = TypeVar("_N_pts", bound=int)

@final
class _FuncPts(Protocol[_Name_co]):
    @property
    def __name__(self, /) -> _Name_co: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, /,
        npts: _N_pts,
    ) -> np.ndarray[tuple[_N_pts], np.dtype[np.float64]]: ...
    @overload
    def __call__(self, /, npts: _AnyInt) -> _Array1D[np.float64]: ...
