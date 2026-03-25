from collections.abc import Callable, Iterable
from typing import Any, Final, NamedTuple

from numpy._utils import set_module as set_module

type _FuncLike = type | Callable[..., object]
type _Dispatcher[**_Tss] = Callable[_Tss, Iterable[object]]

###

ARRAY_FUNCTIONS: set[Callable[..., Any]] = ...
array_function_like_doc: Final[str] = ...

class ArgSpec(NamedTuple):
    args: list[str]
    varargs: str | None
    keywords: str | None
    defaults: tuple[Any, ...]

def get_array_function_like_doc(public_api: Callable[..., object], docstring_template: str = "") -> str: ...
def finalize_array_function_like[FuncLikeT: _FuncLike](public_api: FuncLikeT) -> FuncLikeT: ...

#
def verify_matching_signatures[**Tss](implementation: Callable[Tss, object], dispatcher: _Dispatcher[Tss]) -> None: ...

# NOTE: This actually returns a `_ArrayFunctionDispatcher` callable wrapper object, with
# the original wrapped callable stored in the `._implementation` attribute. It checks
# for any `__array_function__` of the values of specific arguments that the dispatcher
# specifies. Since the dispatcher only returns an iterable of passed array-like args,
# this overridable behaviour is impossible to annotate.
def array_function_dispatch[**Tss, FuncLikeT: _FuncLike](
    dispatcher: _Dispatcher[Tss] | None = None,
    module: str | None = None,
    verify: bool = True,
    docs_from_dispatcher: bool = False,
) -> Callable[[FuncLikeT], FuncLikeT]: ...

#
def array_function_from_dispatcher[**Tss, T](
    implementation: Callable[Tss, T],
    module: str | None = None,
    verify: bool = True,
    docs_from_dispatcher: bool = True,
) -> Callable[[_Dispatcher[Tss]], Callable[Tss, T]]: ...
