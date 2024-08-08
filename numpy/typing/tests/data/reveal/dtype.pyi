import sys
import ctypes as ct
import datetime as dt
from decimal import Decimal
from fractions import Fraction
from typing import Any, TypeAlias

import numpy as np

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

dtype_U: np.dtype[np.str_]
dtype_V: np.dtype[np.void]
dtype_i8: np.dtype[np.int64]

# equivalent to type[int]
py_int_co: type[int] | type[bool]
# equivalent to type[float] (type-check only)
py_float_co: type[float] | type[int] | type[bool]
# equivalent to type[complex] (type-check only)
py_complex_co: type[complex] | type[float] | type[int] | type[bool]
# equivalent to type[object]
py_object_co: (
    type[object]
    | type[complex] | type[float] | type[int] | type[bool]
    | type[str] | type[bytes]
    # ...
)
py_character_co: type[str] | type[bytes]
py_flexible_co: type[str] | type[bytes] | type[memoryview]


assert_type(np.dtype(np.float64), np.dtype[np.float64])
assert_type(np.dtype(np.float64, metadata={"test": "test"}), np.dtype[np.float64])
assert_type(np.dtype(np.int64), np.dtype[np.int64])

# String aliases
assert_type(np.dtype("float64"), np.dtype[np.float64])
assert_type(np.dtype("float32"), np.dtype[np.float32])
assert_type(np.dtype("int64"), np.dtype[np.int64])
assert_type(np.dtype("int32"), np.dtype[np.int32])
assert_type(np.dtype("bool"), np.dtype[np.bool])
assert_type(np.dtype("bytes"), np.dtype[np.bytes_])
assert_type(np.dtype("str"), np.dtype[np.str_])

# Python types
assert_type(np.dtype(bool), np.dtype[np.bool])
assert_type(np.dtype(int), np.dtype[np.int_ | np.bool])
assert_type(np.dtype(py_int_co), np.dtype[np.int_ | np.bool])
assert_type(np.dtype(float), np.dtype[np.float64 | np.int_ | np.bool])
assert_type(np.dtype(py_float_co), np.dtype[np.float64 | np.int_ | np.bool])
assert_type(np.dtype(complex), np.dtype[np.complex128 | np.float64 | np.int_ | np.bool])
assert_type(np.dtype(py_complex_co), np.dtype[np.complex128 | np.float64 | np.int_ | np.bool])
assert_type(np.dtype(object), np.dtype[np.object_ | Any])
assert_type(np.dtype(py_object_co), np.dtype[np.object_ | Any])

assert_type(np.dtype(str), np.dtype[np.str_])
assert_type(np.dtype(bytes), np.dtype[np.bytes_])
assert_type(np.dtype(memoryview), np.dtype[np.void])

assert_type(np.dtype(list), np.dtype[np.object_])
assert_type(np.dtype(dt.datetime), np.dtype[np.object_])
assert_type(np.dtype(dt.timedelta), np.dtype[np.object_])
assert_type(np.dtype(Decimal), np.dtype[np.object_])
assert_type(np.dtype(Fraction), np.dtype[np.object_])

# ctypes
assert_type(np.dtype(ct.c_double), np.dtype[np.double])
assert_type(np.dtype(ct.c_longlong), np.dtype[np.longlong])
assert_type(np.dtype(ct.c_uint32), np.dtype[np.uint32])
assert_type(np.dtype(ct.c_bool), np.dtype[np.bool])
assert_type(np.dtype(ct.c_char), np.dtype[np.bytes_])
assert_type(np.dtype(ct.py_object), np.dtype[np.object_])

# Special case for None
assert_type(np.dtype(None), np.dtype[np.float64])

# Dtypes of dtypes
assert_type(np.dtype(np.dtype(np.float64)), np.dtype[np.float64])

# Parameterized dtypes
assert_type(np.dtype("S8"), np.dtype[Any])

# Void
assert_type(np.dtype(("U", 10)), np.dtype[np.void])

# Methods and attributes
assert_type(dtype_U.base, np.dtype[Any])
assert_type(dtype_U.subdtype, None | tuple[np.dtype[Any], tuple[int, ...]])
assert_type(dtype_U.newbyteorder(), np.dtype[np.str_])
assert_type(dtype_U.type, type[np.str_])
assert_type(dtype_U.name, str)
assert_type(dtype_U.names, None | tuple[str, ...])

assert_type(dtype_U * 0, np.dtype[np.str_])
assert_type(dtype_U * 1, np.dtype[np.str_])
assert_type(dtype_U * 2, np.dtype[np.str_])

assert_type(dtype_i8 * 0, np.dtype[np.void])
assert_type(dtype_i8 * 1, np.dtype[np.int64])
assert_type(dtype_i8 * 2, np.dtype[np.void])

assert_type(0 * dtype_U, np.dtype[np.str_])
assert_type(1 * dtype_U, np.dtype[np.str_])
assert_type(2 * dtype_U, np.dtype[np.str_])

assert_type(0 * dtype_i8, np.dtype[Any])
assert_type(1 * dtype_i8, np.dtype[Any])
assert_type(2 * dtype_i8, np.dtype[Any])

assert_type(dtype_V["f0"], np.dtype[Any])
assert_type(dtype_V[0], np.dtype[Any])
assert_type(dtype_V[["f0", "f1"]], np.dtype[np.void])
assert_type(dtype_V[["f0"]], np.dtype[np.void])
