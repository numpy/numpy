import numpy as np

dtype_U: np.dtype[np.str_]
dtype_V: np.dtype[np.void]
dtype_i8: np.dtype[np.int64]

reveal_type(np.dtype(np.float64))  # E: numpy.dtype[{float64}]
reveal_type(np.dtype(np.int64))  # E: numpy.dtype[{int64}]

# String aliases
reveal_type(np.dtype("float64"))  # E: numpy.dtype[{float64}]
reveal_type(np.dtype("float32"))  # E: numpy.dtype[{float32}]
reveal_type(np.dtype("int64"))  # E: numpy.dtype[{int64}]
reveal_type(np.dtype("int32"))  # E: numpy.dtype[{int32}]
reveal_type(np.dtype("bool"))  # E: numpy.dtype[numpy.bool_]
reveal_type(np.dtype("bytes"))  # E: numpy.dtype[numpy.bytes_]
reveal_type(np.dtype("str"))  # E: numpy.dtype[numpy.str_]

# Python types
reveal_type(np.dtype(complex))  # E: numpy.dtype[{cdouble}]
reveal_type(np.dtype(float))  # E: numpy.dtype[{double}]
reveal_type(np.dtype(int))  # E: numpy.dtype[{int_}]
reveal_type(np.dtype(bool))  # E: numpy.dtype[numpy.bool_]
reveal_type(np.dtype(str))  # E: numpy.dtype[numpy.str_]
reveal_type(np.dtype(bytes))  # E: numpy.dtype[numpy.bytes_]

# Special case for None
reveal_type(np.dtype(None))  # E: numpy.dtype[{double}]

# Dtypes of dtypes
reveal_type(np.dtype(np.dtype(np.float64)))  # E: numpy.dtype[{float64}]

# Parameterized dtypes
reveal_type(np.dtype("S8"))  # E: numpy.dtype

# Void
reveal_type(np.dtype(("U", 10)))  # E: numpy.dtype[numpy.void]

# Methods and attributes
reveal_type(dtype_U.base)  # E: numpy.dtype[Any]
reveal_type(dtype_U.subdtype)  # E: Union[None, Tuple[numpy.dtype[Any], builtins.tuple[builtins.int]]]
reveal_type(dtype_U.newbyteorder())  # E: numpy.dtype[numpy.str_]
reveal_type(dtype_U.type)  # E: Type[numpy.str_]
reveal_type(dtype_U.name)  # E: str
reveal_type(dtype_U.names)  # E: Union[None, builtins.tuple[builtins.str]]

reveal_type(dtype_U * 0)  # E: numpy.dtype[numpy.str_]
reveal_type(dtype_U * 1)  # E: numpy.dtype[numpy.str_]
reveal_type(dtype_U * 2)  # E: numpy.dtype[numpy.str_]

reveal_type(dtype_i8 * 0)  # E: numpy.dtype[numpy.void]
reveal_type(dtype_i8 * 1)  # E: numpy.dtype[{int64}]
reveal_type(dtype_i8 * 2)  # E: numpy.dtype[numpy.void]

reveal_type(0 * dtype_U)  # E: numpy.dtype[numpy.str_]
reveal_type(1 * dtype_U)  # E: numpy.dtype[numpy.str_]
reveal_type(2 * dtype_U)  # E: numpy.dtype[numpy.str_]

reveal_type(0 * dtype_i8)  # E: numpy.dtype[Any]
reveal_type(1 * dtype_i8)  # E: numpy.dtype[Any]
reveal_type(2 * dtype_i8)  # E: numpy.dtype[Any]

reveal_type(dtype_V["f0"])  # E: numpy.dtype[Any]
reveal_type(dtype_V[0])  # E: numpy.dtype[Any]
reveal_type(dtype_V[["f0", "f1"]])  # E: numpy.dtype[numpy.void]
reveal_type(dtype_V[["f0"]])  # E: numpy.dtype[numpy.void]
