"""
A module with various ``typing.Protocol`` subclasses that implement
the ``__call__`` magic method.

See the `Mypy documentation`_ on protocols for more details.

.. _`Mypy documentation`: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols

"""

from typing import Any, Protocol, TypeAlias, TypeVar, final, overload, type_check_only

import numpy as np
from numpy import complex128, complexfloating, float64, floating, integer

from . import NBitBase
from ._array_like import NDArray
from ._nbit import _NBitInt
from ._nested_sequence import _NestedSequence

_T = TypeVar("_T")
_T1_contra = TypeVar("_T1_contra", contravariant=True)
_T2_contra = TypeVar("_T2_contra", contravariant=True)

_2Tuple: TypeAlias = tuple[_T, _T]

_NBit1 = TypeVar("_NBit1", bound=NBitBase)
_NBit2 = TypeVar("_NBit2", bound=NBitBase)

@type_check_only
class _FloatOp(Protocol[_NBit1]):
    @overload
    def __call__(self, other: int, /) -> floating[_NBit1]: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1] | float64: ...
    @overload
    def __call__(
        self, other: complex, /
    ) -> complexfloating[_NBit1, _NBit1] | complex128: ...
    @overload
    def __call__(
        self, other: integer[_NBit2] | floating[_NBit2], /
    ) -> floating[_NBit1] | floating[_NBit2]: ...

@type_check_only
class _FloatMod(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> floating[_NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> floating[_NBit1] | floating[_NBitInt]: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1] | float64: ...
    @overload
    def __call__(
        self, other: integer[_NBit2] | floating[_NBit2], /
    ) -> floating[_NBit1] | floating[_NBit2]: ...

class _FloatDivMod(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> _2Tuple[floating[_NBit1]]: ...
    @overload
    def __call__(
        self, other: int, /
    ) -> _2Tuple[floating[_NBit1]] | _2Tuple[floating[_NBitInt]]: ...
    @overload
    def __call__(
        self, other: float, /
    ) -> _2Tuple[floating[_NBit1]] | _2Tuple[float64]: ...
    @overload
    def __call__(
        self, other: integer[_NBit2] | floating[_NBit2], /
    ) -> _2Tuple[floating[_NBit1]] | _2Tuple[floating[_NBit2]]: ...

@final
@type_check_only
class _SupportsLT(Protocol):
    def __lt__(self, other: Any, /) -> Any: ...

@final
@type_check_only
class _SupportsLE(Protocol):
    def __le__(self, other: Any, /) -> Any: ...

@final
@type_check_only
class _SupportsGT(Protocol):
    def __gt__(self, other: Any, /) -> Any: ...

@final
@type_check_only
class _SupportsGE(Protocol):
    def __ge__(self, other: Any, /) -> Any: ...

@final
@type_check_only
class _ComparisonOpLT(Protocol[_T1_contra, _T2_contra]):
    @overload
    def __call__(self, other: _T1_contra, /) -> np.bool: ...
    @overload
    def __call__(self, other: _T2_contra, /) -> NDArray[np.bool]: ...
    @overload
    def __call__(self, other: _NestedSequence[_SupportsGT], /) -> NDArray[np.bool]: ...
    @overload
    def __call__(self, other: _SupportsGT, /) -> np.bool: ...

@final
@type_check_only
class _ComparisonOpLE(Protocol[_T1_contra, _T2_contra]):
    @overload
    def __call__(self, other: _T1_contra, /) -> np.bool: ...
    @overload
    def __call__(self, other: _T2_contra, /) -> NDArray[np.bool]: ...
    @overload
    def __call__(self, other: _NestedSequence[_SupportsGE], /) -> NDArray[np.bool]: ...
    @overload
    def __call__(self, other: _SupportsGE, /) -> np.bool: ...

@final
@type_check_only
class _ComparisonOpGT(Protocol[_T1_contra, _T2_contra]):
    @overload
    def __call__(self, other: _T1_contra, /) -> np.bool: ...
    @overload
    def __call__(self, other: _T2_contra, /) -> NDArray[np.bool]: ...
    @overload
    def __call__(self, other: _NestedSequence[_SupportsLT], /) -> NDArray[np.bool]: ...
    @overload
    def __call__(self, other: _SupportsLT, /) -> np.bool: ...

@final
@type_check_only
class _ComparisonOpGE(Protocol[_T1_contra, _T2_contra]):
    @overload
    def __call__(self, other: _T1_contra, /) -> np.bool: ...
    @overload
    def __call__(self, other: _T2_contra, /) -> NDArray[np.bool]: ...
    @overload
    def __call__(self, other: _NestedSequence[_SupportsGT], /) -> NDArray[np.bool]: ...
    @overload
    def __call__(self, other: _SupportsGT, /) -> np.bool: ...
