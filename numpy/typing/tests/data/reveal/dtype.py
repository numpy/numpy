import numpy as np

reveal_type(np.dtype(np.float64))  # E: numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(np.dtype(np.int64))  # E: numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]

# String aliases
reveal_type(np.dtype("float64"))  # E: numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(np.dtype("float32"))  # E: numpy.dtype[numpy.floating[numpy.typing._32Bit]]
reveal_type(np.dtype("int64"))  # E: numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(np.dtype("int32"))  # E: numpy.dtype[numpy.signedinteger[numpy.typing._32Bit]]
reveal_type(np.dtype("bool"))  # E: numpy.dtype[numpy.bool_]
reveal_type(np.dtype("bytes"))  # E: numpy.dtype[numpy.bytes_]
reveal_type(np.dtype("str"))  # E: numpy.dtype[numpy.str_]

# Python types
reveal_type(np.dtype(complex))  # E: numpy.dtype[numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]]
reveal_type(np.dtype(float))  # E: numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(np.dtype(int))  # E: numpy.dtype
reveal_type(np.dtype(bool))  # E: numpy.dtype[numpy.bool_]
reveal_type(np.dtype(str))  # E: numpy.dtype[numpy.str_]
reveal_type(np.dtype(bytes))  # E: numpy.dtype[numpy.bytes_]

# Special case for None
reveal_type(np.dtype(None))  # E: numpy.dtype[numpy.floating[numpy.typing._64Bit]]

# Dtypes of dtypes
reveal_type(np.dtype(np.dtype(np.float64)))  # E: numpy.dtype[numpy.floating[numpy.typing._64Bit]]

# Parameterized dtypes
reveal_type(np.dtype("S8"))  # E: numpy.dtype

# Void
reveal_type(np.dtype(("U", 10)))  # E: numpy.dtype[numpy.void]
