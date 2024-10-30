from typing import (
    Any,
    TypeVar,
    Protocol,
    type_check_only,
)

__all__ = ["get_include", "info", "show_runtime"]

_T_contra = TypeVar("_T_contra", contravariant=True)

# A file-like object opened in `w` mode
@type_check_only
class _SupportsWrite(Protocol[_T_contra]):
    def write(self, s: _T_contra, /) -> Any: ...

def get_include() -> str: ...

def info(
    object: object = ...,
    maxwidth: int = ...,
    output: None | _SupportsWrite[str] = ...,
    toplevel: str = ...,
) -> None: ...

def source(
    object: object,
    output: None | _SupportsWrite[str] = ...,
) -> None: ...

def show_runtime() -> None: ...
