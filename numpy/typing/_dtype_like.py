from typing import Any, Dict, List, Sequence, Tuple, Union

from numpy import dtype
from ._shape import _ShapeLike

if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
else:
    from typing_extensions import Protocol, TypedDict

_DtypeLikeNested = Any  # TODO: wait for support for recursive types

# Mandatory keys
class _DtypeDictBase(TypedDict):
    names: Sequence[str]
    formats: Sequence[_DtypeLikeNested]

# Mandatory + optional keys
class _DtypeDict(_DtypeDictBase, total=False):
    offsets: Sequence[int]
    titles: Sequence[Union[bytes, Text, None]]
    itemsize: int
    aligned: bool

# A protocol for anything with the dtype attribute
class _SupportsDtype:
    dtype: _DtypeLikeNested

# Anything that can be coerced into numpy.dtype.
# Reference: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
DtypeLike = Union[
    dtype,
    # default data type (float64)
    None,
    # array-scalar types and generic types
    type,  # TODO: enumerate these when we add type hints for numpy scalars
    # anything with a dtype attribute
    _SupportsDtype,
    # character codes, type strings or comma-separated fields, e.g., 'float64'
    str,
    # (flexible_dtype, itemsize)
    Tuple[_DtypeLikeNested, int],
    # (fixed_dtype, shape)
    Tuple[_DtypeLikeNested, _ShapeLike],
    # [(field_name, field_dtype, field_shape), ...]
    #
    # The type here is quite broad because NumPy accepts quite a wide
    # range of inputs inside the list; see the tests for some
    # examples.
    List[Any],
    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ...,
    #  'itemsize': ...}
    _DtypeDict,
    # {'field1': ..., 'field2': ..., ...}
    Dict[str, Tuple[_DtypeLikeNested, int]],
    # (base_dtype, new_dtype)
    Tuple[_DtypeLikeNested, _DtypeLikeNested],
]
