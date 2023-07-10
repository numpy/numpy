from collections.abc import Callable
from typing import (
    Any,
    overload,
    TypeVar,
)

_FuncType = TypeVar("_FuncType", bound=Callable[..., Any])

@overload
def deprecate(
    *,
    old_name: None | str = ...,
    new_name: None | str = ...,
    message: None | str = ...,
) -> _Deprecate: ...
@overload
def deprecate(
    func: _FuncType,
    /,
    old_name: None | str = ...,
    new_name: None | str = ...,
    message: None | str = ...,
) -> _FuncType: ...

def deprecate_with_doc(msg: None | str) -> _Deprecate: ...
