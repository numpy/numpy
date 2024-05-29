"""
Array API Inspection namespace

This is the namespace for inspection functions as defined by the array API standard.
See https://data-apis.org/array-api/latest/API_specification/inspection.html
for more details.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import (bool, int8, int16, int32, int64, uint8, uint16, uint32,
                   uint64, float32, float64, complex64, complex128)

if TYPE_CHECKING:
    from typing import Optional, Union, Tuple, List, ModuleType, TypedDict
    from numpy.dtyping import DtypeLike

    Capabilities = TypedDict(
        "Capabilities", {"boolean indexing": bool, "data-dependent shapes": bool}
    )

    DefaultDataTypes = TypedDict(
        "DefaultDataTypes",
        {
            "real floating": DtypeLike,
            "complex floating": DtypeLike,
            "integral": DtypeLike,
            "indexing": DtypeLike,
        },
    )

    DataTypes = TypedDict(
        "DataTypes",
        {
            "bool": DtypeLike,
            "float32": DtypeLike,
            "float64": DtypeLike,
            "complex64": DtypeLike,
            "complex128": DtypeLike,
            "int8": DtypeLike,
            "int16": DtypeLike,
            "int32": DtypeLike,
            "int64": DtypeLike,
            "uint8": DtypeLike,
            "uint16": DtypeLike,
            "uint32": DtypeLike,
            "uint64": DtypeLike,
        },
        total=False,
    )


def __array_namespace_info__() -> ModuleType:
    import numpy.lib._info
    return numpy.lib._info

def capabilities() -> Capabilities:
    return {"boolean indexing": True,
            "data-dependent shapes": True,
            }

def default_device() -> str:
    return 'cpu'

def default_dtypes(
    *,
    device: Optional[str] = None,
) -> DefaultDataTypes:
    if device not in ['cpu', None]:
        raise ValueError(f'Device not understood. Only "cpu" is allowed, but received: {device}')
    return {
        "real floating": float64,
        "complex floating": complex128,
        "integral": int64,
        "indexing": int64,
    }

def dtypes(
    *,
    device: Optional[str] = None,
    kind: Optional[Union[str, Tuple[str, ...]]] = None,
) -> DataTypes:
    if device not in ['cpu', None]:
        raise ValueError(f'Device not understood. Only "cpu" is allowed, but received: {device}')
    if kind is None:
        return {
            "bool": bool,
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
            "float32": float32,
            "float64": float64,
            "complex64": complex64,
            "complex128": complex128,
        }
    if kind == "bool":
        return {"bool": bool}
    if kind == "signed integer":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
        }
    if kind == "unsigned integer":
        return {
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
        }
    if kind == "integral":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
        }
    if kind == "real floating":
        return {
            "float32": float32,
            "float64": float64,
        }
    if kind == "complex floating":
        return {
            "complex64": complex64,
            "complex128": complex128,
        }
    if kind == "numeric":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
            "float32": float32,
            "float64": float64,
            "complex64": complex64,
            "complex128": complex128,
        }
    if isinstance(kind, tuple):
        res = {}
        for k in kind:
            res.update(dtypes(kind=k))
        return res
    raise ValueError(f"unsupported kind: {kind!r}")

def devices() -> List[str]:
    return ['cpu']
