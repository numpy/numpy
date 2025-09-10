"""
A module with various ``typing.Protocol`` subclasses that implement
the ``__call__`` magic method.

See the `Mypy documentation`_ on protocols for more details.

.. _`Mypy documentation`: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols

"""

from typing import Any, Protocol, TypeVar, final, overload, type_check_only

import numpy as np

from ._array_like import NDArray
from ._nested_sequence import _NestedSequence

_T1_contra = TypeVar("_T1_contra", contravariant=True)
_T2_contra = TypeVar("_T2_contra", contravariant=True)

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
