from _typeshed import SupportsWrite
from collections.abc import Callable
from types import TracebackType
from typing import Any, Final, Literal, TypedDict, type_check_only

__all__ = [
    "seterr",
    "geterr",
    "setbufsize",
    "getbufsize",
    "seterrcall",
    "geterrcall",
    "errstate",
]

type _ErrKind = Literal["ignore", "warn", "raise", "call", "print", "log"]
type _ErrCall = Callable[[str, int], Any] | SupportsWrite[str]

@type_check_only
class _ErrDict(TypedDict):
    divide: _ErrKind
    over: _ErrKind
    under: _ErrKind
    invalid: _ErrKind

###

class _unspecified: ...

_Unspecified: Final[_unspecified]

class errstate:
    __slots__ = "_all", "_call", "_divide", "_invalid", "_over", "_token", "_under"

    def __init__(
        self,
        /,
        *,
        call: _ErrCall | _unspecified = ...,  # = _Unspecified
        all: _ErrKind | None = None,
        divide: _ErrKind | None = None,
        over: _ErrKind | None = None,
        under: _ErrKind | None = None,
        invalid: _ErrKind | None = None,
    ) -> None: ...
    def __call__[FuncT: Callable[..., object]](self, /, func: FuncT) -> FuncT: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None: ...

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
