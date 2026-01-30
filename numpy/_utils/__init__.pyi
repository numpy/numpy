from _typeshed import IdentityFunction
from collections.abc import Callable, Iterable
from typing import Protocol, overload, type_check_only

from ._convertions import asbytes as asbytes, asunicode as asunicode

###

@type_check_only
class _HasModule(Protocol):
    __module__: str

###

@overload
def set_module(module: None) -> IdentityFunction: ...
@overload
def set_module[ModuleT: _HasModule](module: str) -> Callable[[ModuleT], ModuleT]: ...

#
def _rename_parameter[T](
    old_names: Iterable[str],
    new_names: Iterable[str],
    dep_version: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
