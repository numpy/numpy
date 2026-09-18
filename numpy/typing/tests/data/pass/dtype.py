from typing import assert_type

import numpy as np

dtype_obj = np.dtype(np.str_)
void_dtype_obj = np.dtype([("f0", np.float64), ("f1", np.float32)])

np.dtype(dtype=np.int64)
np.dtype(int)
np.dtype("int")
np.dtype(None)

np.dtype((int, 2))
np.dtype((int, (1,)))

np.dtype({"names": ["a", "b"], "formats": [int, float]})
np.dtype({"names": ["a"], "formats": [int], "titles": [object]})
np.dtype({"names": ["a"], "formats": [int], "titles": [object()]})

np.dtype([("name", np.str_, 16), ("grades", np.float64, (2,)), ("age", "int32")])

np.dtype(
    {
        "names": ["a", "b"],
        "formats": [int, float],
        "itemsize": 9,
        "aligned": False,
        "titles": ["x", "y"],
        "offsets": [0, 1],
    }
)

np.dtype((np.float64, float))


class Test:
    dtype = np.dtype(float)


np.dtype(Test())

# Methods and attributes
dtype_obj.base
dtype_obj.subdtype
dtype_obj.newbyteorder()
dtype_obj.type
dtype_obj.name
dtype_obj.names

dtype_obj * 0
dtype_obj * 2

0 * dtype_obj
2 * dtype_obj

void_dtype_obj["f0"]
void_dtype_obj[0]
void_dtype_obj[["f0", "f1"]]
void_dtype_obj[["f0"]]


# Abstract DTypes; these functions are never called, they only exist so that
# `isinstance` narrowing against the abstract DTypes is type-checked.

def narrow_integer(dt: np.dtypes.Int64DType | np.dtypes.Float64DType) -> None:
    if isinstance(dt, np.dtypes.IntegerAbstractDType):
        assert_type(dt, np.dtypes.Int64DType)
    else:
        assert_type(dt, np.dtypes.Float64DType)


def narrow_signedness(dt: np.dtypes.Int64DType | np.dtypes.UInt64DType) -> None:
    if isinstance(dt, np.dtypes.UnsignedIntegerAbstractDType):
        assert_type(dt, np.dtypes.UInt64DType)
    else:
        assert_type(dt, np.dtypes.Int64DType)


def narrow_inexact(dt: np.dtypes.Float64DType | np.dtypes.Complex128DType) -> None:
    if isinstance(dt, np.dtypes.ComplexFloatingAbstractDType):
        assert_type(dt, np.dtypes.Complex128DType)
    else:
        assert_type(dt, np.dtypes.Float64DType)


def narrow_number(dt: np.dtypes.BoolDType | np.dtypes.Int64DType) -> None:
    # `bool` is not a number, matching `np.number` and the array API
    if isinstance(dt, np.dtypes.NumberAbstractDType):
        assert_type(dt, np.dtypes.Int64DType)
    else:
        assert_type(dt, np.dtypes.BoolDType)
