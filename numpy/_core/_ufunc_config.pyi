from _typeshed import SupportsWrite
from collections.abc import Callable
from typing import Any, Literal, TypeAlias, TypedDict, type_check_only

from numpy import errstate as errstate

_ErrKind: TypeAlias = Literal["ignore", "warn", "raise", "call", "print", "log"]
_ErrFunc: TypeAlias = Callable[[str, int], Any]
_ErrCall: TypeAlias = _ErrFunc | SupportsWrite[str]

@type_check_only
class _ErrDict(TypedDict):
    divide: _ErrKind
    over: _ErrKind
    under: _ErrKind
    invalid: _ErrKind

def seterr(
    all: _ErrKind | None = None,
    divide: _ErrKind | None = None,
    over: _ErrKind | None = None,
    under: _ErrKind | None = None,
    invalid: _ErrKind | None = None,
) -> _ErrDict: ...
def geterr() -> _ErrDict: ...
def setbufsize(size: int) -> int: ...
def getbufsize() -> int: ...
def seterrcall(func: _ErrCall | None) -> _ErrCall | None: ...
def geterrcall() -> _ErrCall | None: ...

# See `numpy/__init__.pyi` for the `errstate` class and `no_nep5_warnings`
