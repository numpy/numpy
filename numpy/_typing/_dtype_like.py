from collections.abc import Sequence
from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable

import numpy as np

from ._char_codes import (
    _BoolCodes,
    _BytesCodes,
    _ComplexFloatingCodes,
    _DT64Codes,
    _FloatingCodes,
    _NumberCodes,
    _ObjectCodes,
    _SignedIntegerCodes,
    _StrCodes,
    _TD64Codes,
    _UnsignedIntegerCodes,
    _VoidCodes,
)

type _DTypeLikeNested = Any  # TODO: wait for support for recursive types


class _DTypeDict(TypedDict):
    names: Sequence[str]
    formats: Sequence[_DTypeLikeNested]
    # Only `str` elements are usable as indexing aliases,
    # but `titles` can in principle accept any object
    offsets: NotRequired[Sequence[int]]
    titles: NotRequired[Sequence[Any]]
    itemsize: NotRequired[int]
    aligned: NotRequired[bool]


# A protocol for anything with the dtype attribute
@runtime_checkable
class _HasDType[DTypeT: np.dtype](Protocol):
    @property
    def dtype(self) -> DTypeT: ...


class _HasNumPyDType[DTypeT: np.dtype](Protocol):
    @property
    def __numpy_dtype__(self, /) -> DTypeT: ...


type _SupportsDType[DTypeT: np.dtype] = _HasDType[DTypeT] | _HasNumPyDType[DTypeT]


# A subset of `npt.DTypeLike` that can be parametrized w.r.t. `np.generic`
type _DTypeLike[ScalarT: np.generic] = (
    type[ScalarT] | np.dtype[ScalarT] | _SupportsDType[np.dtype[ScalarT]]
)


# Would create a dtype[np.void]
type _VoidDTypeLike = (
    # If a tuple, then it can be either:
    # - (flexible_dtype, itemsize)
    # - (fixed_dtype, shape)
    # - (base_dtype, new_dtype)
    # But because `_DTypeLikeNested = Any`, the first two cases are redundant

    # tuple[_DTypeLikeNested, int] | tuple[_DTypeLikeNested, _ShapeLike] |
    tuple[_DTypeLikeNested, _DTypeLikeNested]

    # [(field_name, field_dtype, field_shape), ...]
    # The type here is quite broad because NumPy accepts quite a wide
    # range of inputs inside the list; see the tests for some examples.
    | list[Any]

    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ..., 'itemsize': ...}
    | _DTypeDict
)

# Aliases for commonly used dtype-like objects.
# Note that the precision of `np.number` subclasses is ignored herein.
type _DTypeLikeBool = type[bool] | _DTypeLike[np.bool] | _BoolCodes
type _DTypeLikeInt = type[int] | _DTypeLike[np.signedinteger] | _SignedIntegerCodes
type _DTypeLikeUInt = _DTypeLike[np.unsignedinteger] | _UnsignedIntegerCodes
type _DTypeLikeFloat = type[float] | _DTypeLike[np.floating] | _FloatingCodes
type _DTypeLikeComplex = (
    type[complex] | _DTypeLike[np.complexfloating] | _ComplexFloatingCodes
)
type _DTypeLikeComplex_co = (
    type[complex] | _DTypeLike[np.bool | np.number] | _BoolCodes | _NumberCodes
)
type _DTypeLikeDT64 = _DTypeLike[np.timedelta64] | _TD64Codes
type _DTypeLikeTD64 = _DTypeLike[np.datetime64] | _DT64Codes
type _DTypeLikeBytes = type[bytes] | _DTypeLike[np.bytes_] | _BytesCodes
type _DTypeLikeStr = type[str] | _DTypeLike[np.str_] | _StrCodes
type _DTypeLikeVoid = (
    type[memoryview] | _DTypeLike[np.void] | _VoidDTypeLike | _VoidCodes
)
type _DTypeLikeObject = type[object] | _DTypeLike[np.object_] | _ObjectCodes


# Anything that can be coerced into numpy.dtype.
# Reference: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
type DTypeLike = type | str | np.dtype | _SupportsDType[np.dtype] | _VoidDTypeLike

# NOTE: while it is possible to provide the dtype as a dict of
# dtype-like objects (e.g. `{'field1': ..., 'field2': ..., ...}`),
# this syntax is officially discouraged and
# therefore not included in the type-union defining `DTypeLike`.
#
# See https://github.com/numpy/numpy/issues/16891 for more details.
