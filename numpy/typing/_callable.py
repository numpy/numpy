"""
A module with various ``typing.Protocol`` subclasses that implement
the ``__call__`` magic method.

See the `Mypy documentation`_ on protocols for more details.

.. _`Mypy documentation`: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols

"""

import sys
from typing import Union, TypeVar, overload, Any, NoReturn

from numpy import (
    _BoolLike,
    _IntLike,
    _FloatLike,
    _ComplexLike,
    _NumberLike,
    generic,
    bool_,
    timedelta64,
    number,
    integer,
    unsignedinteger,
    signedinteger,
    int32,
    int64,
    uint64,
    floating,
    float32,
    float64,
    complexfloating,
    complex128,
)

if sys.version_info >= (3, 8):
    from typing import Protocol
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

if HAVE_PROTOCOL:
    _IntType = TypeVar("_IntType", bound=integer)
    _NumberType = TypeVar("_NumberType", bound=number)
    _NumberType_co = TypeVar("_NumberType_co", covariant=True, bound=number)
    _GenericType_co = TypeVar("_GenericType_co", covariant=True, bound=generic)

    class _BoolOp(Protocol[_GenericType_co]):
        @overload
        def __call__(self, __other: _BoolLike) -> _GenericType_co: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> Union[int32, int64]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(self, __other: _NumberType) -> _NumberType: ...

    class _BoolBitOp(Protocol[_GenericType_co]):
        @overload
        def __call__(self, __other: _BoolLike) -> _GenericType_co: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> Union[int32, int64]: ...
        @overload
        def __call__(self, __other: _IntType) -> _IntType: ...

    class _BoolSub(Protocol):
        # Note that `__other: bool_` is absent here
        @overload  # platform dependent
        def __call__(self, __other: int) -> Union[int32, int64]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(self, __other: _NumberType) -> _NumberType: ...

    class _BoolTrueDiv(Protocol):
        @overload
        def __call__(self, __other: Union[float, _IntLike, _BoolLike]) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(self, __other: _NumberType) -> _NumberType: ...

    class _TD64Div(Protocol[_NumberType_co]):
        @overload
        def __call__(self, __other: timedelta64) -> _NumberType_co: ...
        @overload
        def __call__(self, __other: _FloatLike) -> timedelta64: ...

    class _IntTrueDiv(Protocol):
        @overload
        def __call__(self, __other: Union[_IntLike, float]) -> floating: ...
        @overload
        def __call__(self, __other: complex) -> complexfloating[floating]: ...

    class _UnsignedIntOp(Protocol):
        # NOTE: `uint64 + signedinteger -> float64`
        @overload
        def __call__(self, __other: Union[bool, unsignedinteger]) -> unsignedinteger: ...
        @overload
        def __call__(self, __other: Union[int, signedinteger]) -> Union[signedinteger, float64]: ...
        @overload
        def __call__(self, __other: float) -> floating: ...
        @overload
        def __call__(self, __other: complex) -> complexfloating[floating]: ...

    class _UnsignedIntBitOp(Protocol):
        # The likes of `uint64 | np.signedinteger` will fail as there
        # is no signed integer type large enough to hold a `uint64`
        # See https://github.com/numpy/numpy/issues/2524
        @overload
        def __call__(self, __other: Union[bool, unsignedinteger]) -> unsignedinteger: ...
        @overload
        def __call__(self: uint64, __other: Union[int, signedinteger]) -> NoReturn: ...
        @overload
        def __call__(self, __other: Union[int, signedinteger]) -> signedinteger: ...

    class _SignedIntOp(Protocol):
        @overload
        def __call__(self, __other: Union[int, signedinteger]) -> signedinteger: ...
        @overload
        def __call__(self, __other: float) -> floating: ...
        @overload
        def __call__(self, __other: complex) -> complexfloating[floating]: ...

    class _SignedIntBitOp(Protocol):
        def __call__(self, __other: Union[int, signedinteger]) -> signedinteger: ...

    class _FloatOp(Protocol):
        @overload
        def __call__(self, __other: _FloatLike) -> floating: ...
        @overload
        def __call__(self, __other: complex) -> complexfloating[floating]: ...

    class _ComplexOp(Protocol):
        def __call__(self, __other: _ComplexLike) -> complexfloating[floating]: ...

    class _NumberOp(Protocol):
        def __call__(self, __other: _NumberLike) -> number: ...

else:
    _BoolOp = Any
    _BoolBitOp = Any
    _BoolSub = Any
    _BoolTrueDiv = Any
    _TD64Div = Any
    _IntTrueDiv = Any
    _UnsignedIntOp = Any
    _UnsignedIntBitOp = Any
    _SignedIntOp = Any
    _SignedIntBitOp = Any
    _FloatOp = Any
    _ComplexOp = Any
    _NumberOp = Any
