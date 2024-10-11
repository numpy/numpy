from collections.abc import Callable
from typing import Any, Literal, TypeAlias, TypedDict, type_check_only

from numpy import _SupportsWrite

_ErrKind: TypeAlias = Literal["ignore", "warn", "raise", "call", "print", "log"]
_ErrFunc: TypeAlias = Callable[[str, int], Any]

@type_check_only
class _ErrDict(TypedDict):
    divide: _ErrKind
    over: _ErrKind
    under: _ErrKind
    invalid: _ErrKind

@type_check_only
class _ErrDictOptional(TypedDict, total=False):
    all: None | _ErrKind
    divide: None | _ErrKind
    over: None | _ErrKind
    under: None | _ErrKind
    invalid: None | _ErrKind

def seterr(
    all: None | _ErrKind = ...,
    divide: None | _ErrKind = ...,
    over: None | _ErrKind = ...,
    under: None | _ErrKind = ...,
    invalid: None | _ErrKind = ...,
) -> _ErrDict: ...
def geterr() -> _ErrDict: ...
def setbufsize(size: int) -> int: ...
def getbufsize() -> int: ...
def seterrcall(
    func: None | _ErrFunc | _SupportsWrite[str]
) -> None | _ErrFunc | _SupportsWrite[str]: ...
def geterrcall() -> None | _ErrFunc | _SupportsWrite[str]: ...

# See `numpy/__init__.pyi` for the `errstate` class and `no_nep5_warnings`
