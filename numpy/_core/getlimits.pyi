from types import GenericAlias
from typing import Final, Generic, Self, overload
from typing_extensions import TypeVar

import numpy as np
from numpy._typing import (
    _CLongDoubleCodes,
    _Complex64Codes,
    _Complex128Codes,
    _DTypeLike,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _IntPCodes,
    _LongDoubleCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
)

__all__ = ["finfo", "iinfo"]

###

_IntegerT_co = TypeVar("_IntegerT_co", bound=np.integer, default=np.integer, covariant=True)
_FloatingT_co = TypeVar("_FloatingT_co", bound=np.floating, default=np.floating, covariant=True)

###

class iinfo(Generic[_IntegerT_co]):
    dtype: np.dtype[_IntegerT_co]
    bits: Final[int]
    kind: Final[str]
    key: Final[str]

    @property
    def min(self, /) -> int: ...
    @property
    def max(self, /) -> int: ...

    #
    @overload
    def __init__(self, /, int_type: _IntegerT_co | _DTypeLike[_IntegerT_co]) -> None: ...
    @overload
    def __init__(self: iinfo[np.int_], /, int_type: _IntPCodes | type[int] | int) -> None: ...
    @overload
    def __init__(self: iinfo[np.int8], /, int_type: _Int8Codes) -> None: ...
    @overload
    def __init__(self: iinfo[np.uint8], /, int_type: _UInt8Codes) -> None: ...
    @overload
    def __init__(self: iinfo[np.int16], /, int_type: _Int16Codes) -> None: ...
    @overload
    def __init__(self: iinfo[np.uint16], /, int_type: _UInt16Codes) -> None: ...
    @overload
    def __init__(self: iinfo[np.int32], /, int_type: _Int32Codes) -> None: ...
    @overload
    def __init__(self: iinfo[np.uint32], /, int_type: _UInt32Codes) -> None: ...
    @overload
    def __init__(self: iinfo[np.int64], /, int_type: _Int64Codes) -> None: ...
    @overload
    def __init__(self: iinfo[np.uint64], /, int_type: _UInt64Codes) -> None: ...
    @overload
    def __init__(self, /, int_type: str) -> None: ...

    #
    @classmethod
    def __class_getitem__(cls, item: object, /) -> GenericAlias: ...

class finfo(Generic[_FloatingT_co]):
    dtype: np.dtype[_FloatingT_co]
    eps: _FloatingT_co
    epsneg: _FloatingT_co
    resolution: _FloatingT_co
    smallest_subnormal: _FloatingT_co
    max: _FloatingT_co
    min: _FloatingT_co

    bits: Final[int]
    iexp: Final[int]
    machep: Final[int]
    maxexp: Final[int]
    minexp: Final[int]
    negep: Final[int]
    nexp: Final[int]
    nmant: Final[int]
    precision: Final[int]

    @property
    def smallest_normal(self, /) -> _FloatingT_co: ...
    @property
    def tiny(self, /) -> _FloatingT_co: ...

    #
    @overload
    def __new__(cls, dtype: _FloatingT_co | _DTypeLike[_FloatingT_co]) -> Self: ...
    @overload
    def __new__(cls, dtype: _Float16Codes) -> finfo[np.float16]: ...
    @overload
    def __new__(cls, dtype: _Float32Codes | _Complex64Codes | _DTypeLike[np.complex64]) -> finfo[np.float32]: ...
    @overload
    def __new__(cls, dtype: _Float64Codes | _Complex128Codes | type[complex] | complex) -> finfo[np.float64]: ...
    @overload
    def __new__(cls, dtype: _LongDoubleCodes | _CLongDoubleCodes | _DTypeLike[np.clongdouble]) -> finfo[np.longdouble]: ...
    @overload
    def __new__(cls, dtype: str) -> finfo: ...

    #
    @classmethod
    def __class_getitem__(cls, item: object, /) -> GenericAlias: ...
