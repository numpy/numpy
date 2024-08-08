"""A module with private type-check-only `numpy.ufunc` subclasses.

The signatures of the ufuncs are too varied to reasonably type
with a single class. So instead, `ufunc` has been expanded into
four private subclasses, one for each combination of
`~ufunc.nin` and `~ufunc.nout`.

"""

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    NoReturn,
    TypedDict,
    overload,
    TypeAlias,
    TypeVar,
    Literal,
    SupportsIndex,
    Protocol,
    NoReturn,
    final,
)
if sys.version_info >= (3, 11):
    from typing import Unpack
elif TYPE_CHECKING:
    from typing_extensions import Unpack

import numpy as np
from numpy import ufunc, _CastingKind, _OrderKACF
from numpy.typing import NDArray

from ._shape import _ShapeLike
from ._scalars import _ScalarLike_co
from ._array_like import ArrayLike, _ArrayLikeBool_co, _ArrayLikeInt_co
from ._dtype_like import DTypeLike

_T = TypeVar("_T")
_2Tuple: TypeAlias = tuple[_T, _T]
_3Tuple: TypeAlias = tuple[_T, _T, _T]
_4Tuple: TypeAlias = tuple[_T, _T, _T, _T]

if TYPE_CHECKING or sys.version_info >= (3, 11):
    _2PTuple: TypeAlias = tuple[_T, _T, Unpack[tuple[_T, ...]]]
    _3PTuple: TypeAlias = tuple[_T, _T, _T, Unpack[tuple[_T, ...]]]
    _4PTuple: TypeAlias = tuple[_T, _T, _T, _T, Unpack[tuple[_T, ...]]]
else:
    _2PTuple: TypeAlias = tuple[_T, ...]
    _3PTuple: TypeAlias = tuple[_T, ...]
    _4PTuple: TypeAlias = tuple[_T, ...]

_NTypes = TypeVar("_NTypes", bound=int, covariant=True)
_IDType = TypeVar("_IDType", bound=Any, covariant=True)
_NameType = TypeVar("_NameType", bound=str, covariant=True)
_Signature = TypeVar("_Signature", bound=str, covariant=True)

_Nin = TypeVar("_Nin", bound=int, covariant=True)
_Nout = TypeVar("_Nout", bound=int, covariant=True)
_ArrayT = TypeVar("_ArrayT", bound=NDArray[Any])


class _SupportsArrayUFunc(Protocol):
    def __array_ufunc__(
        self,
        ufunc: ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...


# NOTE: `reduce`, `accumulate`, `reduceat` and `outer` raise a ValueError for
# ufuncs that don't accept two input arguments and return one output argument.
# In such cases the respective methods return `NoReturn`

# NOTE: Similarly, `at` won't be defined for ufuncs that return
# multiple outputs; in such cases `at` is typed to return `NoReturn`

# NOTE: If 2 output types are returned then `out` must be a
# 2-tuple of arrays. Otherwise `None` or a plain array are also acceptable

class _UFunc_Nin1_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType]):  # type: ignore[misc]
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[1]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[2]: ...
    @property
    def signature(self) -> None: ...

    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        out: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _2Tuple[None | str] = ...,
    ) -> Any: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _2Tuple[None | str] = ...,
    ) -> NDArray[Any]: ...
    @overload
    def __call__(
        self,
        __x1: _SupportsArrayUFunc,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _2Tuple[None | str] = ...,
    ) -> Any: ...

    def at(
        self,
        a: _SupportsArrayUFunc,
        indices: _ArrayLikeInt_co,
        /,
    ) -> None: ...

    def reduce(self, *args, **kwargs) -> NoReturn: ...
    def accumulate(self, *args, **kwargs) -> NoReturn: ...
    def reduceat(self, *args, **kwargs) -> NoReturn: ...
    def outer(self, *args, **kwargs) -> NoReturn: ...


class _UFunc_Nin2_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType]):  # type: ignore[misc]
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[3]: ...
    @property
    def signature(self) -> None: ...

    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __x2: _ScalarLike_co,
        out: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> Any: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> NDArray[Any]: ...

    def at(
        self,
        a: NDArray[Any],
        indices: _ArrayLikeInt_co,
        b: ArrayLike,
        /,
    ) -> None: ...

    def reduce(
        self,
        array: ArrayLike,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None | NDArray[Any] = ...,
        keepdims: bool = ...,
        initial: Any = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...

    def accumulate(
        self,
        array: ArrayLike,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None | NDArray[Any] = ...,
    ) -> NDArray[Any]: ...

    def reduceat(
        self,
        array: ArrayLike,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None | NDArray[Any] = ...,
    ) -> NDArray[Any]: ...

    # Expand `**kwargs` into explicit keyword-only arguments
    @overload
    def outer(
        self,
        A: _ScalarLike_co,
        B: _ScalarLike_co,
        /, *,
        out: None = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> Any: ...
    @overload
    def outer(  # type: ignore[misc]
        self,
        A: ArrayLike,
        B: ArrayLike,
        /, *,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> NDArray[Any]: ...

class _UFunc_Nin1_Nout2(ufunc, Generic[_NameType, _NTypes, _IDType]):  # type: ignore[misc]
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[1]: ...
    @property
    def nout(self) -> Literal[2]: ...
    @property
    def nargs(self) -> Literal[3]: ...
    @property
    def signature(self) -> None: ...

    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __out1: None = ...,
        __out2: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> _2Tuple[Any]: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __out1: None | NDArray[Any] = ...,
        __out2: None | NDArray[Any] = ...,
        *,
        out: _2Tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> _2Tuple[NDArray[Any]]: ...
    @overload
    def __call__(
        self,
        __x1: _SupportsArrayUFunc,
        __out1: None | NDArray[Any] = ...,
        __out2: None | NDArray[Any] = ...,
        *,
        out: _2Tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
    ) -> _2Tuple[Any]: ...

    def at(self, *args, **kwargs) -> NoReturn: ...
    def reduce(self, *args, **kwargs) -> NoReturn: ...
    def accumulate(self, *args, **kwargs) -> NoReturn: ...
    def reduceat(self, *args, **kwargs) -> NoReturn: ...
    def outer(self, *args, **kwargs) -> NoReturn: ...

class _UFunc_Nin2_Nout2(ufunc, Generic[_NameType, _NTypes, _IDType]):  # type: ignore[misc]
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[2]: ...
    @property
    def nargs(self) -> Literal[4]: ...
    @property
    def signature(self) -> None: ...

    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __x2: _ScalarLike_co,
        __out1: None = ...,
        __out2: None = ...,
        *,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _4Tuple[None | str] = ...,
    ) -> _2Tuple[Any]: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        __out1: None | NDArray[Any] = ...,
        __out2: None | NDArray[Any] = ...,
        *,
        out: _2Tuple[NDArray[Any]] = ...,
        where: None | _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _4Tuple[None | str] = ...,
    ) -> _2Tuple[NDArray[Any]]: ...

    def at(self, *args, **kwargs) -> NoReturn: ...
    def reduce(self, *args, **kwargs) -> NoReturn: ...
    def accumulate(self, *args, **kwargs) -> NoReturn: ...
    def reduceat(self, *args, **kwargs) -> NoReturn: ...
    def outer(self, *args, **kwargs) -> NoReturn: ...

class _GUFunc_Nin2_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType, _Signature]):  # type: ignore[misc]
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[3]: ...
    @property
    def signature(self) -> _Signature: ...

    # Scalar for 1D array-likes; ndarray otherwise
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: None = ...,
        *,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        axes: list[_2Tuple[SupportsIndex]] = ...,
    ) -> Any: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: NDArray[Any] | tuple[NDArray[Any]],
        *,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: str | _3Tuple[None | str] = ...,
        axes: list[_2Tuple[SupportsIndex]] = ...,
    ) -> NDArray[Any]: ...

    def at(self, *args, **kwargs) -> NoReturn: ...
    def reduce(self, *args, **kwargs) -> NoReturn: ...
    def accumulate(self, *args, **kwargs) -> NoReturn: ...
    def reduceat(self, *args, **kwargs) -> NoReturn: ...
    def outer(self, *args, **kwargs) -> NoReturn: ...

class _PyFunc_Kwargs_Nargs2(TypedDict, total=False):
    where: None | _ArrayLikeBool_co
    casting: _CastingKind
    order: _OrderKACF
    dtype: DTypeLike
    subok: bool
    signature: str | tuple[DTypeLike, DTypeLike]

class _PyFunc_Kwargs_Nargs3(TypedDict, total=False):
    where: None | _ArrayLikeBool_co
    casting: _CastingKind
    order: _OrderKACF
    dtype: DTypeLike
    subok: bool
    signature: str | tuple[DTypeLike, DTypeLike, DTypeLike]

class _PyFunc_Kwargs_Nargs3P(TypedDict, total=False):
    where: None | _ArrayLikeBool_co
    casting: _CastingKind
    order: _OrderKACF
    dtype: DTypeLike
    subok: bool
    signature: str | _3PTuple[DTypeLike]

class _PyFunc_Kwargs_Nargs4P(TypedDict, total=False):
    where: None | _ArrayLikeBool_co
    casting: _CastingKind
    order: _OrderKACF
    dtype: DTypeLike
    subok: bool
    signature: str | _4PTuple[DTypeLike]

@final
class _PyFunc_Nin1_Nout1(ufunc, Generic[_IDType]):  # type: ignore[misc]
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[1]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[2]: ...
    @property
    def ntypes(self) -> Literal[1]: ...
    @property
    def signature(self) -> None: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self,
        x1: _ScalarLike_co,
        /,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> np.object_: ...
    @overload
    def __call__(
        self,
        x1: ArrayLike,
        /,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> NDArray[np.object_]: ...
    @overload
    def __call__(
        self,
        x1: ArrayLike,
        /,
        out: _ArrayT | tuple[_ArrayT],
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> _ArrayT: ...
    @overload
    def __call__(
        self,
        x1: _SupportsArrayUFunc,
        /,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs2],
    ) -> Any: ...

    def reduce(self, *args, **kwargs) -> NoReturn: ...
    def accumulate(self, *args, **kwargs) -> NoReturn: ...
    def reduceat(self, *args, **kwargs) -> NoReturn: ...
    def outer(self, *args, **kwargs) -> NoReturn: ...
    def at(
        self,
        a: _SupportsArrayUFunc,
        indices: _ArrayLikeInt_co,
        /,
    ) -> None: ...


@final
class _PyFunc_Nin2_Nout1(ufunc, Generic[_IDType]):  # type: ignore[misc]
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[3]: ...
    @property
    def ntypes(self) -> Literal[1]: ...
    @property
    def signature(self) -> None: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self,
        x1: _ScalarLike_co,
        x2: _ScalarLike_co,
        /,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> np.object_: ...
    @overload
    def __call__(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        /,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> NDArray[np.object_]: ...
    @overload
    def __call__(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        /,
        out: _ArrayT | tuple[_ArrayT],
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ArrayT: ...
    @overload
    def __call__(
        self,
        x1: _SupportsArrayUFunc,
        x2: _SupportsArrayUFunc | ArrayLike,
        /,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> Any: ...
    @overload
    def __call__(
        self,
        x1: ArrayLike,
        x2: _SupportsArrayUFunc,
        /,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> Any: ...

    def at(
        self,
        a: NDArray[Any],
        indices: _ArrayLikeInt_co,
        b: ArrayLike,
        /,
    ) -> None: ...

    @overload
    def reduce(
        self,
        array: ArrayLike,
        axis: None | _ShapeLike,
        dtype: DTypeLike,
        out: _ArrayT,
        /,
        keepdims: bool = ...,
        initial: _ScalarLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _ArrayT: ...
    @overload
    def reduce(
        self,
        array: ArrayLike,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        *,
        out: _ArrayT | tuple[_ArrayT],
        keepdims: bool = ...,
        initial: _ScalarLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _ArrayT: ...
    @overload
    def reduce(
        self,
        /,
        array: ArrayLike,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        *,
        keepdims: Literal[True],
        initial: _ScalarLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> NDArray[np.object_]: ...
    @overload
    def reduce(
        self,
        /,
        array: ArrayLike,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _ScalarLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> np.object_ | NDArray[np.object_]: ...

    @overload
    def reduceat(
        self,
        array: ArrayLike,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex,
        dtype: DTypeLike,
        out: _ArrayT,
        /,
    ) -> _ArrayT: ...
    @overload
    def reduceat(
        self,
        /,
        array: ArrayLike,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        *,
        out: _ArrayT | tuple[_ArrayT],
    ) -> _ArrayT: ...
    @overload
    def reduceat(
        self,
        /,
        array: ArrayLike,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> NDArray[np.object_]: ...
    @overload
    def reduceat(
        self,
        /,
        array: _SupportsArrayUFunc,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
    ) -> Any: ...

    @overload
    def accumulate(
        self,
        array: ArrayLike,
        axis: SupportsIndex,
        dtype: DTypeLike,
        out: _ArrayT,
        /,
    ) -> _ArrayT: ...
    @overload
    def accumulate(
        self,
        array: ArrayLike,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        *,
        out: _ArrayT | tuple[_ArrayT],
    ) -> _ArrayT: ...
    @overload
    def accumulate(
        self,
        /,
        array: ArrayLike,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> NDArray[np.object_]: ...

    @overload
    def outer(  # type: ignore[overload-overlap]
        self,
        A: _ScalarLike_co,
        B: _ScalarLike_co,
        /, *,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> np.object_: ...
    @overload
    def outer(
        self,
        A: ArrayLike,
        B: ArrayLike,
        /, *,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> NDArray[np.object_]: ...
    @overload
    def outer(
        self,
        A: ArrayLike,
        B: ArrayLike,
        /, *,
        out: _ArrayT,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> _ArrayT: ...
    @overload
    def outer(
        self,
        A: _SupportsArrayUFunc,
        B: _SupportsArrayUFunc | ArrayLike,
        /, *,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> Any: ...
    @overload
    def outer(
        self,
        A: _ScalarLike_co,
        B: _SupportsArrayUFunc | ArrayLike,
        /, *,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3],
    ) -> Any: ...

@final
class _PyFunc_Nin3P_Nout1(ufunc, Generic[_IDType, _Nin]):  # type: ignore[misc]
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> _Nin: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def ntypes(self) -> Literal[1]: ...
    @property
    def signature(self) -> None: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self,
        x1: _ScalarLike_co,
        x2: _ScalarLike_co,
        x3: _ScalarLike_co,
        /,
        *xs: _ScalarLike_co,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> np.object_: ...
    @overload
    def __call__(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        x3: ArrayLike,
        /,
        *xs: ArrayLike,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> NDArray[np.object_]: ...
    @overload
    def __call__(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        x3: ArrayLike,
        /,
        *xs: ArrayLike,
        out: _ArrayT | tuple[_ArrayT],
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> _ArrayT: ...
    @overload
    def __call__(
        self,
        x1: _SupportsArrayUFunc | ArrayLike,
        x2: _SupportsArrayUFunc | ArrayLike,
        x3: _SupportsArrayUFunc | ArrayLike,
        /,
        *xs: _SupportsArrayUFunc | ArrayLike,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs4P],
    ) -> Any: ...

    def at(self, *args, **kwargs) -> NoReturn: ...
    def reduce(self, *args, **kwargs) -> NoReturn: ...
    def accumulate(self, *args, **kwargs) -> NoReturn: ...
    def reduceat(self, *args, **kwargs) -> NoReturn: ...
    def outer(self, *args, **kwargs) -> NoReturn: ...

@final
class _PyFunc_Nin1P_Nout2P(ufunc, Generic[_IDType, _Nin, _Nout]):  # type: ignore[misc]
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> _Nin: ...
    @property
    def nout(self) -> _Nout: ...
    @property
    def ntypes(self) -> Literal[1]: ...
    @property
    def signature(self) -> None: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self,
        x1: _ScalarLike_co,
        /,
        *xs: _ScalarLike_co,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> _2PTuple[np.object_]: ...
    @overload
    def __call__(
        self,
        x1: ArrayLike,
        /,
        *xs: ArrayLike,
        out: None = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> _2PTuple[NDArray[np.object_]]: ...
    @overload
    def __call__(
        self,
        x1: ArrayLike,
        /,
        *xs: ArrayLike,
        out: _2PTuple[_ArrayT],
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> _2PTuple[_ArrayT]: ...
    @overload
    def __call__(
        self,
        x1: _SupportsArrayUFunc | ArrayLike,
        /,
        *xs: _SupportsArrayUFunc | ArrayLike,
        out: None | _2PTuple[NDArray[Any]] = ...,
        **kwargs: Unpack[_PyFunc_Kwargs_Nargs3P],
    ) -> Any: ...

    def at(self, *args, **kwargs) -> NoReturn: ...
    def reduce(self, *args, **kwargs) -> NoReturn: ...
    def accumulate(self, *args, **kwargs) -> NoReturn: ...
    def reduceat(self, *args, **kwargs) -> NoReturn: ...
    def outer(self, *args, **kwargs) -> NoReturn: ...
