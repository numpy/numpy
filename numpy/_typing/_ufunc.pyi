"""A module with private type-check-only `numpy.ufunc` subclasses.

The signatures of the ufuncs are too varied to reasonably type
with a single class. So instead, `ufunc` has been expanded into
four private subclasses, one for each combination of
`~ufunc.nin` and `~ufunc.nout`.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from types import EllipsisType
from typing import (
    Any,
    Literal,
    Never,
    NoReturn,
    Protocol,
    SupportsIndex,
    TypedDict,
    Unpack,
    overload,
    override,
    type_check_only,
)

import numpy as np
from numpy import _CastingKind, _OrderKACF, ufunc

from ._array_like import ArrayLike, NDArray, _ArrayLikeBool_co, _ArrayLikeInt_co
from ._dtype_like import DTypeLike
from ._scalars import _ScalarLike_co
from ._shape import _AnyShape, _Shape, _ShapeLike

type _2PTuple[T] = tuple[T, T, *tuple[T, ...]]
type _3PTuple[T] = tuple[T, T, T, *tuple[T, ...]]
type _4PTuple[T] = tuple[T, T, T, T, *tuple[T, ...]]

type _ObjectArray[ShapeT: _Shape, ItemT] = np.ndarray[ShapeT, np.dtype[np.object_[ItemT]]]

# workaround for mypy and pyright not following the typing spec for overloads
type _ArrayJustND = np.ndarray[tuple[Never, Never, Never, Never], Any]

@type_check_only
class _SupportsArrayUFunc(Protocol):
    def __array_ufunc__(
        self,
        ufunc: ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...

@type_check_only
class _ReduceKwargs(TypedDict, total=False):
    initial: Incomplete  # = <no value>
    where: _ArrayLikeBool_co | None  # = True

# NOTE: `reduce`, `accumulate`, `reduceat` and `outer` raise a ValueError for
# ufuncs that don't accept two input arguments and return one output argument.
# In such cases the respective methods return `NoReturn`

# NOTE: Similarly, `at` won't be defined for ufuncs that return
# multiple outputs; in such cases `at` is typed to return `NoReturn`

# NOTE: If 2 output types are returned then `out` must be a
# 2-tuple of arrays. Otherwise `None` or a plain array are also acceptable

# pyright: reportIncompatibleMethodOverride=false

@type_check_only
class _PyFunc_Kwargs_Nargs2(TypedDict, total=False):
    where: _ArrayLikeBool_co | None
    casting: _CastingKind
    order: _OrderKACF
    dtype: DTypeLike
    subok: bool
    signature: str | tuple[DTypeLike, DTypeLike]

@type_check_only
class _PyFunc_Kwargs_Nargs3(TypedDict, total=False):
    where: _ArrayLikeBool_co | None
    casting: _CastingKind
    order: _OrderKACF
    dtype: DTypeLike
    subok: bool
    signature: str | tuple[DTypeLike, DTypeLike, DTypeLike]

@type_check_only
class _PyFunc_Kwargs_Nargs3P(TypedDict, total=False):
    where: _ArrayLikeBool_co | None
    casting: _CastingKind
    order: _OrderKACF
    dtype: DTypeLike
    subok: bool
    signature: str | _3PTuple[DTypeLike]

@type_check_only
class _PyFunc_Kwargs_Nargs4P(TypedDict, total=False):
    where: _ArrayLikeBool_co | None
    casting: _CastingKind
    order: _OrderKACF
    dtype: DTypeLike
    subok: bool
    signature: str | _4PTuple[DTypeLike]

@type_check_only
class _PyFunc_Nin1_Nout1[ReturnT, IdentT](ufunc):  # type: ignore[misc]
    @property
    @override
    def identity(self) -> IdentT: ...
    @property
    @override
    def nin(self) -> Literal[1]: ...
    @property
    @override
    def nout(self) -> Literal[1]: ...
    @property
    @override
    def nargs(self) -> Literal[2]: ...
    @property
    @override
    def ntypes(self) -> Literal[1]: ...
    @property
    @override
    def signature(self) -> None: ...

    #
    @override
    @overload  # Nd
    def __call__[ShapeT: _Shape](
        self,
        x1: np.ndarray[ShapeT, Any],
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # 0d
    def __call__(
        self,
        x1: _ScalarLike_co,
        /,
        out: None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> ReturnT: ...
    @overload  # 0d, out=...
    def __call__(
        self,
        x1: _ScalarLike_co,
        /,
        out: EllipsisType,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> _ObjectArray[tuple[()], ReturnT]: ...
    @overload  # 1d
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: list[ScalarT],
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # 2d
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: Sequence[list[ScalarT]],
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> _ObjectArray[tuple[int, int], ReturnT]: ...
    @overload  # ?d  (fallback)
    def __call__(
        self,
        x1: ArrayLike,
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> ReturnT | _ObjectArray[_AnyShape, ReturnT]: ...
    @overload  # ?d, out=T
    def __call__[OutT: np.ndarray](
        self,
        x1: ArrayLike,
        /,
        out: OutT | tuple[OutT],
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> OutT: ...
    @overload  # __array_ufunc__
    def __call__(
        self,
        x1: _SupportsArrayUFunc,
        /,
        out: np.ndarray | tuple[np.ndarray] | EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> Any: ...

    #
    @override
    def at(self, a: np.ndarray | _SupportsArrayUFunc, indices: _ArrayLikeInt_co, /) -> None: ...  # type: ignore[override]

    #
    @override
    def accumulate(self, array: Never, /) -> NoReturn: ...  # type: ignore[override]
    @override
    def reduce(self, array: Never, /) -> NoReturn: ...  # type: ignore[override]
    @override
    def reduceat(self, array: Never, /, indices: Never) -> NoReturn: ...  # type: ignore[override]
    @override
    def outer(self, A: Never, B: Never, /) -> NoReturn: ...  # type: ignore[override]

@type_check_only
class _PyFunc_Nin2_Nout1[ReturnT, IdentT](ufunc):  # type: ignore[misc]
    @property
    @override
    def identity(self) -> IdentT: ...
    @property
    @override
    def nin(self) -> Literal[2]: ...
    @property
    @override
    def nout(self) -> Literal[1]: ...
    @property
    @override
    def nargs(self) -> Literal[3]: ...
    @property
    @override
    def ntypes(self) -> Literal[1]: ...
    @property
    @override
    def signature(self) -> None: ...

    #
    @override
    @overload  # Nd, Nd | 0d
    def __call__[ShapeT: _Shape](
        self,
        x1: np.ndarray[ShapeT, Any],
        x2: np.ndarray[ShapeT, Any] | _ScalarLike_co,
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # 0d, Nd
    def __call__[ShapeT: _Shape](
        self,
        x1: _ScalarLike_co,
        x2: np.ndarray[ShapeT, Any],
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # 0d, 0d
    def __call__(
        self,
        x1: _ScalarLike_co,
        x2: _ScalarLike_co,
        /,
        out: None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> ReturnT: ...
    @overload  # 0d, 0d, out=...
    def __call__(
        self,
        x1: _ScalarLike_co,
        x2: _ScalarLike_co,
        /,
        out: EllipsisType,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[tuple[()], ReturnT]: ...
    @overload  # 1d, <=1d
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: list[ScalarT],
        x2: _ScalarLike_co | Sequence[_ScalarLike_co],
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # 0d, 1d
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: _ScalarLike_co,
        x2: list[ScalarT],
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # 2d, <=2d
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: Sequence[list[ScalarT]],
        x2: _ScalarLike_co | Sequence[_ScalarLike_co] | Sequence[Sequence[_ScalarLike_co]],
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[tuple[int, int], ReturnT]: ...
    @overload  # <=1d, 2d
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: _ScalarLike_co | Sequence[_ScalarLike_co],
        x2: Sequence[list[ScalarT]],
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[tuple[int, int], ReturnT]: ...
    @overload  # ?d  (fallback)
    def __call__(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        /,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> ReturnT | _ObjectArray[_AnyShape, ReturnT]: ...
    @overload  # ?d, out=T
    def __call__[OutT: np.ndarray](
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        /,
        out: OutT | tuple[OutT],
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> OutT: ...
    @overload  # __array_ufunc__, ?d
    def __call__(
        self,
        x1: _SupportsArrayUFunc,
        x2: _SupportsArrayUFunc | ArrayLike,
        /,
        out: np.ndarray | tuple[np.ndarray] | EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> Any: ...
    @overload  # ?d, __array_ufunc__
    def __call__(
        self,
        x1: ArrayLike,
        x2: _SupportsArrayUFunc,
        /,
        out: np.ndarray | tuple[np.ndarray] | EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> Any: ...

    #
    @override  # type: ignore[override]
    @overload  # Nd
    def accumulate[ShapeT: _Shape](  # pyrefly: ignore[bad-override]
        self,
        array: np.ndarray[ShapeT, Any],
        /,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # 1d
    def accumulate[ScalarT: _ScalarLike_co](
        self,
        array: list[ScalarT],
        /,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # 2d
    def accumulate[ScalarT: _ScalarLike_co](
        self,
        array: Sequence[list[ScalarT]],
        /,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
    ) -> _ObjectArray[tuple[int, int], ReturnT]: ...
    @overload  # ?d, out=T
    def accumulate[OutT: np.ndarray](
        self,
        array: ArrayLike,
        /,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        *,
        out: OutT,
    ) -> OutT: ...
    @overload  # ?d  (fallback)
    def accumulate(
        self,
        array: ArrayLike,
        /,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
    ) -> NDArray[np.object_[ReturnT]]: ...

    #
    @override  # type: ignore[override]
    @overload  # Nd, keepdims=True
    def reduce[ShapeT: _Shape](  # pyrefly: ignore[bad-override]
        self,
        array: np.ndarray[ShapeT, Any],
        /,
        axis: _ShapeLike | None = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
        *,
        keepdims: Literal[True],
        **kwargs: Unpack[_ReduceKwargs],
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # ?d Any  (workaround)
    def reduce(
        self,
        array: _ArrayJustND,
        /,
        axis: _ShapeLike | None = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
        keepdims: bool = False,
        **kwargs: Unpack[_ReduceKwargs],
    ) -> NDArray[np.object_[ReturnT]] | Any: ...
    @overload  # <=1d
    def reduce[ScalarT: _ScalarLike_co](
        self,
        array: _ScalarLike_co | np.ndarray[tuple[int], Any] | list[ScalarT],
        /,
        axis: SupportsIndex | None = 0,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        **kwargs: Unpack[_ReduceKwargs],
    ) -> ReturnT: ...
    @overload  # <=1d, out=...
    def reduce[ScalarT: _ScalarLike_co](
        self,
        array: _ScalarLike_co | np.ndarray[tuple[int], Any] | list[ScalarT],
        /,
        axis: SupportsIndex | None = 0,
        dtype: DTypeLike | None = None,
        *,
        out: EllipsisType,
        keepdims: Literal[False] = False,
        **kwargs: Unpack[_ReduceKwargs],
    ) -> _ObjectArray[tuple[()], ReturnT]: ...
    @overload  # 2d
    def reduce[ScalarT: _ScalarLike_co](
        self,
        array: np.ndarray[tuple[int, int], Any] | Sequence[list[ScalarT]],
        /,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        **kwargs: Unpack[_ReduceKwargs],
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # 3d
    def reduce(
        self,
        array: np.ndarray[tuple[int, int, int], Any],
        /,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        **kwargs: Unpack[_ReduceKwargs],
    ) -> _ObjectArray[tuple[int, int], ReturnT]: ...
    @overload  # ?d, out=T
    def reduce[OutT: np.ndarray](
        self,
        array: ArrayLike,
        /,
        axis: _ShapeLike | None = 0,
        dtype: DTypeLike | None = None,
        *,
        out: OutT | tuple[OutT],
        keepdims: bool = False,
        **kwargs: Unpack[_ReduceKwargs],
    ) -> OutT: ...
    @overload  # ?d, out=...
    def reduce(
        self,
        array: ArrayLike,
        /,
        axis: _ShapeLike | None = 0,
        dtype: DTypeLike | None = None,
        *,
        out: EllipsisType,
        keepdims: bool = False,
        **kwargs: Unpack[_ReduceKwargs],
    ) -> NDArray[np.object_[ReturnT]]: ...
    @overload  # ?d, keepdims=True
    def reduce(
        self,
        array: ArrayLike,
        /,
        axis: _ShapeLike | None = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
        *,
        keepdims: Literal[True],
        **kwargs: Unpack[_ReduceKwargs],
    ) -> NDArray[np.object_[ReturnT]]: ...
    @overload  # ?d  (fallback)
    def reduce(
        self,
        array: ArrayLike,
        /,
        axis: _ShapeLike | None = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
        keepdims: bool = False,
        **kwargs: Unpack[_ReduceKwargs],
    ) -> NDArray[np.object_[ReturnT]] | Any: ...
    @overload  # __array_ufunc__
    def reduce(
        self,
        array: _SupportsArrayUFunc,
        /,
        axis: _ShapeLike | None = 0,
        dtype: DTypeLike | None = None,
        out: np.ndarray | tuple[np.ndarray] | EllipsisType | None = None,
        keepdims: bool = False,
        **kwargs: Unpack[_ReduceKwargs],
    ) -> Any: ...

    #
    @override  # type: ignore[override]
    @overload  # Nd
    def reduceat[ShapeT: _Shape](  # pyrefly: ignore[bad-override]
        self,
        array: np.ndarray[ShapeT, Any],
        /,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # 1d
    def reduceat[ScalarT: _ScalarLike_co](
        self,
        array: list[ScalarT],
        /,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # 2d
    def reduceat[ScalarT: _ScalarLike_co](
        self,
        array: Sequence[list[ScalarT]],
        /,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
    ) -> _ObjectArray[tuple[int, int], ReturnT]: ...
    @overload  # ?d, out=T
    def reduceat[OutT: np.ndarray](
        self,
        array: ArrayLike,
        /,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        *,
        out: OutT | tuple[OutT],
    ) -> OutT: ...
    @overload  # ?d  (fallback)
    def reduceat(
        self,
        array: ArrayLike,
        /,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: EllipsisType | None = None,
    ) -> NDArray[np.object_[ReturnT]]: ...
    @overload  # __array_ufunc__
    def reduceat(
        self,
        array: _SupportsArrayUFunc,
        /,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = 0,
        dtype: DTypeLike | None = None,
        out: np.ndarray | tuple[np.ndarray] | EllipsisType | None = None,
    ) -> Any: ...

    #
    @override  # type: ignore[override]
    @overload  # Nd, 0d
    def outer[ShapeT: _Shape](  # pyrefly: ignore[bad-override]
        self,
        A: np.ndarray[ShapeT, Any],
        B: _ScalarLike_co,
        /,
        *,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # 0d, Nd
    def outer[ShapeT: _Shape](
        self,
        A: _ScalarLike_co,
        B: np.ndarray[ShapeT, Any],
        /,
        *,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # 0d, 0d
    def outer(
        self,
        A: _ScalarLike_co,
        B: _ScalarLike_co,
        /,
        *,
        out: None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> ReturnT: ...
    @overload  # 0d, 0d, out=...
    def outer(
        self,
        A: _ScalarLike_co,
        B: _ScalarLike_co,
        /,
        *,
        out: EllipsisType,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[tuple[()], ReturnT]: ...
    @overload  # 1d, 1d
    def outer[ScalarT: _ScalarLike_co](
        self,
        A: list[ScalarT],
        B: Sequence[_ScalarLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[tuple[int, int], ReturnT]: ...
    @overload  # 1d, 0d
    def outer[ScalarT: _ScalarLike_co](
        self,
        A: list[ScalarT],
        B: _ScalarLike_co,
        /,
        *,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # 0d, 1d
    def outer[ScalarT: _ScalarLike_co](
        self,
        A: _ScalarLike_co,
        B: list[ScalarT],
        /,
        *,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # ?d  (fallback)
    def outer(
        self,
        A: ArrayLike,
        B: ArrayLike,
        /,
        *,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ObjectArray[_AnyShape, ReturnT] | Any: ...
    @overload  # ?d, out=T
    def outer[OutT: np.ndarray](
        self,
        A: ArrayLike,
        B: ArrayLike,
        /,
        *,
        out: OutT,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> OutT: ...
    @overload  # __array_ufunc__, ?d
    def outer(
        self,
        A: _SupportsArrayUFunc,
        B: _SupportsArrayUFunc | ArrayLike,
        /,
        *,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> Any: ...
    @overload  # 0d, __array_ufunc__
    def outer(
        self,
        A: _ScalarLike_co,
        B: _SupportsArrayUFunc | ArrayLike,
        /,
        *,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> Any: ...

    #
    @override
    def at(  # type: ignore[override]
        self,
        a: np.ndarray | _SupportsArrayUFunc,
        indices: _ArrayLikeInt_co,
        b: ArrayLike,
        /,
    ) -> None: ...

@type_check_only
class _PyFunc_Nin3P_Nout1[ReturnT, IdentT](ufunc):  # type: ignore[misc]
    @property
    @override
    def identity(self) -> IdentT: ...
    @property
    @override
    def nout(self) -> Literal[1]: ...
    @property
    @override
    def ntypes(self) -> Literal[1]: ...
    @property
    @override
    def signature(self) -> None: ...

    #
    @override
    @overload  # Nd, Nd | 0d, ...
    def __call__[ShapeT: _Shape](
        self,
        x1: np.ndarray[ShapeT, Any],
        x2: np.ndarray[ShapeT, Any] | _ScalarLike_co,
        x3: np.ndarray[ShapeT, Any] | _ScalarLike_co,
        /,
        *xs: np.ndarray[ShapeT, Any] | _ScalarLike_co,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # 0d, Nd, Nd | 0d, ...
    def __call__[ShapeT: _Shape](
        self,
        x1: _ScalarLike_co,
        x2: np.ndarray[ShapeT, Any],
        x3: np.ndarray[ShapeT, Any] | _ScalarLike_co,
        /,
        *xs: np.ndarray[ShapeT, Any] | _ScalarLike_co,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # 0d, 0d, Nd, Nd | 0d, ...
    def __call__[ShapeT: _Shape](
        self,
        x1: _ScalarLike_co,
        x2: _ScalarLike_co,
        x3: np.ndarray[ShapeT, Any],
        /,
        *xs: np.ndarray[ShapeT, Any] | _ScalarLike_co,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[ShapeT, ReturnT]: ...
    @overload  # 0d, ...
    def __call__(
        self,
        x1: _ScalarLike_co,
        x2: _ScalarLike_co,
        x3: _ScalarLike_co,
        /,
        *xs: _ScalarLike_co,
        out: None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> ReturnT: ...
    @overload  # 0d, ..., out=...
    def __call__(
        self,
        x1: _ScalarLike_co,
        x2: _ScalarLike_co,
        x3: _ScalarLike_co,
        /,
        *xs: _ScalarLike_co,
        out: EllipsisType,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[tuple[()], ReturnT]: ...
    @overload  # 1d, <=1d, ...
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: list[ScalarT],
        x2: _ScalarLike_co | Sequence[_ScalarLike_co],
        x3: _ScalarLike_co | Sequence[_ScalarLike_co],
        /,
        *xs: _ScalarLike_co | Sequence[_ScalarLike_co],
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # 0d, 1d, <=1d, ...
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: _ScalarLike_co,
        x2: list[ScalarT],
        x3: _ScalarLike_co | Sequence[_ScalarLike_co],
        /,
        *xs: _ScalarLike_co | Sequence[_ScalarLike_co],
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # 0d, 0d, 1d, <=1d, ...
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: _ScalarLike_co,
        x2: _ScalarLike_co,
        x3: list[ScalarT],
        /,
        *xs: _ScalarLike_co | Sequence[_ScalarLike_co],
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[tuple[int], ReturnT]: ...
    @overload  # 2d, <=2d, ...
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: Sequence[list[ScalarT]],
        x2: _ScalarLike_co | Sequence[_ScalarLike_co] | Sequence[Sequence[_ScalarLike_co]],
        x3: _ScalarLike_co | Sequence[_ScalarLike_co] | Sequence[Sequence[_ScalarLike_co]],
        /,
        *xs: _ScalarLike_co | Sequence[_ScalarLike_co] | Sequence[Sequence[_ScalarLike_co]],
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[tuple[int, int], ReturnT]: ...
    @overload  # <=1d, 2d, <=2d, ...
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: _ScalarLike_co | Sequence[_ScalarLike_co],
        x2: Sequence[list[ScalarT]],
        x3: _ScalarLike_co | Sequence[_ScalarLike_co] | Sequence[Sequence[_ScalarLike_co]],
        /,
        *xs: _ScalarLike_co | Sequence[_ScalarLike_co] | Sequence[Sequence[_ScalarLike_co]],
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[tuple[int, int], ReturnT]: ...
    @overload  # <=1d, <=1d, 2d, <=2d, ...
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: _ScalarLike_co | Sequence[_ScalarLike_co],
        x2: _ScalarLike_co | Sequence[_ScalarLike_co],
        x3: Sequence[list[ScalarT]],
        /,
        *xs: _ScalarLike_co | Sequence[_ScalarLike_co] | Sequence[Sequence[_ScalarLike_co]],
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[tuple[int, int], ReturnT]: ...
    @overload  # ?d  (fallback)
    def __call__(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        x3: ArrayLike,
        /,
        *xs: ArrayLike,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ObjectArray[_AnyShape, ReturnT] | Any: ...
    @overload  # ?d, out=T
    def __call__[OutT: np.ndarray](
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        x3: ArrayLike,
        /,
        *xs: ArrayLike,
        out: OutT | tuple[OutT],
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> OutT: ...
    @overload  # __array_ufunc__
    def __call__(
        self,
        x1: _SupportsArrayUFunc | ArrayLike,
        x2: _SupportsArrayUFunc | ArrayLike,
        x3: _SupportsArrayUFunc | ArrayLike,
        /,
        *xs: _SupportsArrayUFunc | ArrayLike,
        out: np.ndarray | tuple[np.ndarray] | EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> Any: ...

    #
    @override
    def accumulate(self, array: Never, /) -> NoReturn: ...  # type: ignore[override]
    @override
    def reduce(self, array: Never, /) -> NoReturn: ...  # type: ignore[override]
    @override
    def reduceat(self, array: Never, /, indices: Never) -> NoReturn: ...  # type: ignore[override]
    @override
    def outer(self, A: Never, B: Never, /) -> NoReturn: ...  # type: ignore[override]
    @override
    def at(self, a: Never, indices: Never, /, *args: Never) -> NoReturn: ...  # type: ignore[override]

@type_check_only
class _PyFunc_Nin1P_Nout2P[ReturnT, IdentT](ufunc):  # type: ignore[misc]
    @property
    @override
    def identity(self) -> IdentT: ...
    @property
    @override
    def ntypes(self) -> Literal[1]: ...
    @property
    @override
    def signature(self) -> None: ...

    #
    @override
    @overload  # Nd, Nd | 0d, ...
    def __call__[ShapeT: _Shape](
        self,
        x1: np.ndarray[ShapeT, Any],
        /,
        *xs: np.ndarray[ShapeT, Any] | _ScalarLike_co,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> _2PTuple[_ObjectArray[ShapeT, ReturnT]]: ...
    @overload  # 0d, ...
    def __call__(
        self,
        x1: _ScalarLike_co,
        /,
        *xs: _ScalarLike_co,
        out: None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> _2PTuple[ReturnT]: ...
    @overload  # 0d, ..., out=...
    def __call__(
        self,
        x1: _ScalarLike_co,
        /,
        *xs: _ScalarLike_co,
        out: EllipsisType,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> _2PTuple[_ObjectArray[tuple[()], ReturnT]]: ...
    @overload  # 1d, <=1d, ...
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: list[ScalarT],
        /,
        *xs: _ScalarLike_co | Sequence[_ScalarLike_co],
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> _2PTuple[_ObjectArray[tuple[int], ReturnT]]: ...
    @overload  # 2d, <=2d, ...
    def __call__[ScalarT: _ScalarLike_co](
        self,
        x1: Sequence[list[ScalarT]],
        /,
        *xs: _ScalarLike_co | Sequence[_ScalarLike_co] | Sequence[Sequence[_ScalarLike_co]],
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> _2PTuple[_ObjectArray[tuple[int, int], ReturnT]]: ...
    @overload  # ?d  (fallback)
    def __call__(
        self,
        x1: ArrayLike,
        /,
        *xs: ArrayLike,
        out: EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> _2PTuple[NDArray[np.object_[ReturnT]] | Any]: ...
    @overload  # ?d, out=T
    def __call__[OutT: np.ndarray](
        self,
        x1: ArrayLike,
        /,
        *xs: ArrayLike,
        out: _2PTuple[OutT],
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> _2PTuple[OutT]: ...
    @overload  # __array_ufunc__
    def __call__(
        self,
        x1: _SupportsArrayUFunc | ArrayLike,
        /,
        *xs: _SupportsArrayUFunc | ArrayLike,
        out: _2PTuple[np.ndarray] | EllipsisType | None = None,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> Any: ...

    #
    @override
    def accumulate(self, array: Never, /) -> NoReturn: ...  # type: ignore[override]
    @override
    def reduce(self, array: Never, /) -> NoReturn: ...  # type: ignore[override]
    @override
    def reduceat(self, array: Never, /, indices: Never) -> NoReturn: ...  # type: ignore[override]
    @override
    def outer(self, A: Never, B: Never, /) -> NoReturn: ...  # type: ignore[override]
    @override
    def at(self, a: Never, indices: Never, /, *args: Never) -> NoReturn: ...  # type: ignore[override]
