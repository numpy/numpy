import sys
from typing import Any, Dict, List, overload, Sequence, Text, Tuple, Union

from numpy import dtype, ndarray

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

_Shape = Tuple[int, ...]

# Anything that can be coerced to a shape tuple
_ShapeLike = Union[int, Sequence[int]]

_DtypeLikeNested = Any  # TODO: wait for support for recursive types

# Anything that can be coerced into numpy.dtype.
# Reference: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
DtypeLike = Union[
    dtype,
    # default data type (float64)
    None,
    # array-scalar types and generic types
    type,  # TODO: enumerate these when we add type hints for numpy scalars
    # TODO: add a protocol for anything with a dtype attribute
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
    # TODO: use TypedDict when/if it's officially supported
    Dict[
        str,
        Union[
            Sequence[str],  # names
            Sequence[_DtypeLikeNested],  # formats
            Sequence[int],  # offsets
            Sequence[Union[bytes, Text, None]],  # titles
            int,  # itemsize
        ],
    ],
    # {'field1': ..., 'field2': ..., ...}
    Dict[str, Tuple[_DtypeLikeNested, int]],
    # (base_dtype, new_dtype)
    Tuple[_DtypeLikeNested, _DtypeLikeNested],
]

class _SupportsArray(Protocol):
    @overload
    def __array__(self, __dtype: DtypeLike = ...) -> ndarray: ...
    @overload
    def __array__(self, dtype: DtypeLike = ...) -> ndarray: ...

ArrayLike = Union[bool, int, float, complex, _SupportsArray, Sequence]
