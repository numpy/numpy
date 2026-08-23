from _typeshed import SupportsWrite
from typing import LiteralString

import numpy as np

__all__ = ["get_include", "info", "show_runtime"]

def get_include() -> LiteralString: ...
def show_runtime() -> None: ...
def info(
    object: object = None, maxwidth: int = 76, output: SupportsWrite[str] | None = None, toplevel: str = "numpy"
) -> None: ...
def drop_metadata[DTypeT: np.dtype](dtype: DTypeT, /) -> DTypeT: ...

# used internally by `lib._function_base_impl._median`
def _median_nancheck[ScalarOrArrayT: np.generic | np.ndarray](
    data: np.ndarray, result: ScalarOrArrayT, axis: int
) -> ScalarOrArrayT: ...
