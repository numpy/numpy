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
        "Capabilities", {
            "boolean indexing": bool,
            "data-dependent shapes": bool,
            # "max rank": int | None,
        },
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
    """
    Get the array API inspection namespace for NumPy.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.

    Returns
    -------
    info : ModuleType
        The array API inspection namespace for NumPy.

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': numpy.float64,
     'complex floating': numpy.complex128,
     'integral': numpy.int64,
     'indexing': numpy.int64}
    """
    import numpy.lib._info
    return numpy.lib._info

def capabilities() -> Capabilities:
    """
    Return a dictionary of array API library capabilities.

    The resulting dictionary has the following keys:

    - **"boolean indexing"**: boolean indicating whether an array library
      supports boolean indexing. Always ``True`` for NumPy.

    - **"data-dependent shapes"**: boolean indicating whether an array library
        supports data-dependent output shapes. Always ``True`` for NumPy.

    See
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.info.capabilities.html
    for more details.

    See Also
    --------
    default_device, default_dtypes, dtypes, devices

    Returns
    -------
    capabilities : Capabilities
        A dictionary of array API library capabilities.

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.capabilities()
    {'boolean indexing': True,
     'data-dependent shapes': True}

    """
    return {"boolean indexing": True,
            "data-dependent shapes": True,
            # 'max rank' will be part of the 2024.12 standard
            # "max rank": 64,
            }

def default_device() -> str:
    """
    The default device used for new NumPy arrays.

    For NumPy, this always returns ``'cpu'``.

    See Also
    --------
    capabilities, default_dtypes, dtypes, devices

    Returns
    -------
    device : str
        The default device used for new NumPy arrays.

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.default_device()
    'cpu'

    """
    return 'cpu'

def default_dtypes(
    *,
    device: Optional[str] = None,
) -> DefaultDataTypes:
    """
    The default data types used for new NumPy arrays.

    For NumPy, this always returns the following dictionary:

    - **"real floating"**: ``numpy.float64``
    - **"complex floating"**: ``numpy.complex128``
    - **"integral"**: ``numpy.int64``
    - **"indexing"**: ``numpy.int64``

    Parameters
    ----------
    device : str, optional
        The device to get the default data types for. For NumPy, only
        ``'cpu'`` is allowed.

    Returns
    -------
    dtypes : DefaultDataTypes
        A dictionary describing the default data types used for new NumPy
        arrays.

    See Also
    --------
    capabilities, default_device, dtypes, devices

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': numpy.float64,
     'complex floating': numpy.complex128,
     'integral': numpy.int64,
     'indexing': numpy.int64}

    """
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
    """
    The array API data types supported by NumPy.

    Note that this function only returns data types that are defined by the
    array API.

    Parameters
    ----------
    device : str, optional
        The device to get the data types for. For NumPy, only ``'cpu'`` is
        allowed.
    kind : str or tuple of str, optional
        The kind of data types to return. If ``None``, all data types are
        returned. If a string, only data types of that kind are returned. If a
        tuple, a dictionary containing the union of the given kinds is
        returned. The following kinds are supported:

            - ``'bool'``: boolean data types (e.g., ``bool``).
            - ``'signed integer'``: signed integer data types (e.g., ``int8``,
              ``int16``, ``int32``, ``int64``).
            - ``'unsigned integer'``: unsigned integer data types (e.g.,
              ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            - ``'integral'``: integer data types. Shorthand for ``('signed
              integer', 'unsigned integer')``.
            - ``'real floating'``: real-valued floating-point data types
              (e.g., ``float32``, ``float64``).
            - ``'complex floating'``: complex floating-point data types (e.g.,
              ``complex64``, ``complex128``).
            - ``'numeric'``: numeric data types. Shorthand for ``('integral',
              'real floating', 'complex floating')``.

    Returns
    -------
    dtypes : DataTypes
        A dictionary mapping the names of data types to the corresponding
        NumPy data types.

    See Also
    --------
    capabilities, default_device, default_dtypes, devices

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.dtypes(kind='signed integer')
    {'int8': numpy.int8,
     'int16': numpy.int16,
     'int32': numpy.int32,
     'int64': numpy.int64}

    """
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
    """
    The devices supported by NumPy.

    For NumPy, this always returns ``['cpu']``.

    Returns
    -------
    devices : list of str
        The devices supported by NumPy.

    See Also
    --------
    capabilities, default_device, default_dtypes, dtypes

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.devices()
    ['cpu']

    """
    return ['cpu']
