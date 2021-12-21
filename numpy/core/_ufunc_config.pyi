from typing import Union, Callable, Any, Literal, TypedDict

from numpy import _SupportsWrite

_ErrKind = Literal["ignore", "warn", "raise", "call", "print", "log"]
_ErrFunc = Callable[[str, int], Any]

class _ErrDict(TypedDict):
    divide: _ErrKind
    over: _ErrKind
    under: _ErrKind
    invalid: _ErrKind

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
    func: Union[None, _ErrFunc, _SupportsWrite[str]]
) -> Union[None, _ErrFunc, _SupportsWrite[str]]: ...
def geterrcall() -> Union[None, _ErrFunc, _SupportsWrite[str]]: ...

# See `numpy/__init__.pyi` for the `errstate` class
