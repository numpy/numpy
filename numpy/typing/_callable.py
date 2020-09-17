"""
A module with various ``typing.Protocol`` subclasses that implement
the ``__call__`` magic method.

See the `Mypy documentation`_ on protocols for more details.

.. _`Mypy documentation`: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols

"""

import sys
from typing import Union, TypeVar, overload, Any, TYPE_CHECKING

from numpy import (
    _BoolLike,
    _IntLike,
    _FloatLike,
    _NumberLike,
    _NBitBase,
    generic,
    bool_,
    timedelta64,
    number,
    integer,
    unsignedinteger,
    signedinteger,
    int32,
    int64,
    floating,
    float32,
    float64,
    complexfloating,
    complex64,
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

if TYPE_CHECKING or HAVE_PROTOCOL:
    _NumberType = TypeVar("_NumberType", bound=number)
    _NumberType_co = TypeVar("_NumberType_co", covariant=True, bound=number)
    _GenericType_co = TypeVar("_GenericType_co", covariant=True, bound=generic)

    _NBit1 = TypeVar("_NBit1", bound=_NBitBase)
    _NBit2 = TypeVar("_NBit2", bound=_NBitBase)

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

    class _IntTrueDiv(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: Union[_IntLike, float]) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...

    class _UnsignedIntOp(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> unsignedinteger[_NBit1]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(
            self, __other: Union[int, signedinteger]
        ) -> Union[signedinteger[_NBitBase], float64]: ...
        @overload
        def __call__(
            self, __other: unsignedinteger[_NBit2]
        ) -> unsignedinteger[Union[_NBit1, _NBit2]]: ...

    class _SignedIntOp(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> signedinteger[_NBit1]: ...
        @overload
        def __call__(self, __other: int) -> Union[int32, int64]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(
            self, __other: signedinteger[_NBit2]
        ) -> signedinteger[Union[_NBit1, _NBit2]]: ...

    class _FloatOp(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> floating[_NBit1]: ...
        @overload
        def __call__(self, __other: int) -> Union[float32, float64]: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(
            self, __other: Union[integer[_NBit2], floating[_NBit2]]
        ) -> floating[Union[_NBit1, _NBit2]]: ...

    class _ComplexOp(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> complexfloating[_NBit1]: ...
        @overload
        def __call__(self, __other: int) -> Union[complex64, complex128]: ...
        @overload
        def __call__(self, __other: Union[float, complex]) -> complex128: ...
        @overload
        def __call__(
            self, __other: number[_NBit2]
        ) -> complexfloating[Union[_NBit1, _NBit2]]: ...

    class _NumberOp(Protocol[_NBit1]):
        def __call__(self, __other: _NumberLike) -> number: ...

else:
    _BoolOp = Any
    _BoolSub = Any
    _BoolTrueDiv = Any
    _TD64Div = Any
    _IntTrueDiv = Any
    _UnsignedIntOp = Any
    _SignedIntOp = Any
    _FloatOp = Any
    _ComplexOp = Any
    _NumberOp = Any
