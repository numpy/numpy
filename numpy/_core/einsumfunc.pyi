from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
from numpy import _OrderKACF
from numpy._typing import (
    NDArray,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeUInt_co,
    _DTypeLikeBool,
    _DTypeLikeComplex,
    _DTypeLikeComplex_co,
    _DTypeLikeFloat,
    _DTypeLikeInt,
    _DTypeLikeObject,
    _DTypeLikeUInt,
)

__all__ = ["einsum", "einsum_path"]

type _OptimizeKind = bool | Literal["greedy", "optimal"] | Sequence[Any] | None
type _CastingSafe = Literal["no", "equiv", "safe", "same_kind"]
type _CastingUnsafe = Literal["unsafe"]

# TODO: Properly handle the `casting`-based combinatorics
# TODO: We need to evaluate the content `__subscripts` in order
# to identify whether or an array or scalar is returned. At a cursory
# glance this seems like something that can quite easily be done with
# a mypy plugin.
# Something like `is_scalar = bool(__subscripts.partition("->")[-1])`
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeBool_co,
    out: None = None,
    dtype: _DTypeLikeBool | None = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = False,
) -> Any: ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeUInt_co,
    out: None = None,
    dtype: _DTypeLikeUInt | None = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = False,
) -> Any: ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeInt_co,
    out: None = None,
    dtype: _DTypeLikeInt | None = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = False,
) -> Any: ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeFloat_co,
    out: None = None,
    dtype: _DTypeLikeFloat | None = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = False,
) -> Any: ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeComplex_co,
    out: None = None,
    dtype: _DTypeLikeComplex | None = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = False,
) -> Any: ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: Any,
    casting: _CastingUnsafe,
    dtype: _DTypeLikeComplex_co | None = ...,
    out: None = None,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = False,
) -> Any: ...
@overload
def einsum[OutT: NDArray[np.bool | np.number]](
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeComplex_co,
    out: OutT,
    dtype: _DTypeLikeComplex_co | None = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = False,
) -> OutT: ...
@overload
def einsum[OutT: NDArray[np.bool | np.number]](
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: Any,
    out: OutT,
    casting: _CastingUnsafe,
    dtype: _DTypeLikeComplex_co | None = ...,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = False,
) -> OutT: ...

@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeObject_co,
    out: None = None,
    dtype: _DTypeLikeObject | None = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = False,
) -> Any: ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: Any,
    casting: _CastingUnsafe,
    dtype: _DTypeLikeObject | None = ...,
    out: None = None,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = False,
) -> Any: ...
@overload
def einsum[OutT: NDArray[np.bool | np.number]](
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeObject_co,
    out: OutT,
    dtype: _DTypeLikeObject | None = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = False,
) -> OutT: ...
@overload
def einsum[OutT: NDArray[np.bool | np.number]](
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: Any,
    out: OutT,
    casting: _CastingUnsafe,
    dtype: _DTypeLikeObject | None = ...,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = False,
) -> OutT: ...

# NOTE: `einsum_call` is a hidden kwarg unavailable for public use.
# It is therefore excluded from the signatures below.
# NOTE: In practice the list consists of a `str` (first element)
# and a variable number of integer tuples.
def einsum_path(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeComplex_co | _DTypeLikeObject,
    optimize: _OptimizeKind = "greedy",
    einsum_call: Literal[False] = False,
) -> tuple[list[Any], str]: ...
