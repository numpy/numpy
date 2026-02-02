# deprecated module

from types import ModuleType

from . import (
    _dtype,
    _dtype_ctypes,
    _internal,
    arrayprint,
    defchararray,
    einsumfunc,
    fromnumeric,
    function_base,
    getlimits,
    multiarray,
    numeric,
    numerictypes,
    overrides,
    records,
    shape_base,
    umath,
)

__all__ = [
    "_dtype",
    "_dtype_ctypes",
    "_internal",
    "_multiarray_umath",
    "arrayprint",
    "defchararray",
    "einsumfunc",
    "fromnumeric",
    "function_base",
    "getlimits",
    "multiarray",
    "numeric",
    "numerictypes",
    "overrides",
    "records",
    "shape_base",
    "umath",
]

# `numpy._core._multiarray_umath` has no stubs, so there's nothing to re-export
_multiarray_umath: ModuleType
