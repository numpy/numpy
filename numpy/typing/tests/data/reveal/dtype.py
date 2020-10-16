import numpy as np

reveal_type(np.dtype(np.float64))  # E: numpy.dtype[numpy.float64*]
reveal_type(np.dtype(np.int64))  # E: numpy.dtype[numpy.int64*]

# String aliases
reveal_type(np.dtype("float64"))  # E: numpy.dtype[numpy.float64]
reveal_type(np.dtype("float32"))  # E: numpy.dtype[numpy.float32]
reveal_type(np.dtype("int64"))  # E: numpy.dtype[numpy.int64]
reveal_type(np.dtype("int32"))  # E: numpy.dtype[numpy.int32]
reveal_type(np.dtype("bool"))  # E: numpy.dtype[numpy.bool_]
reveal_type(np.dtype("bytes"))  # E: numpy.dtype[numpy.bytes_]
reveal_type(np.dtype("str"))  # E: numpy.dtype[numpy.str_]

# Python types
reveal_type(np.dtype(complex))  # E: numpy.dtype[numpy.complex128]
reveal_type(np.dtype(float))  # E: numpy.dtype[numpy.float64]
reveal_type(np.dtype(int))  # E: numpy.dtype
reveal_type(np.dtype(bool))  # E: numpy.dtype[numpy.bool_]
reveal_type(np.dtype(str))  # E: numpy.dtype[numpy.str_]
reveal_type(np.dtype(bytes))  # E: numpy.dtype[numpy.bytes_]

# Special case for None
reveal_type(np.dtype(None))  # E: numpy.dtype[numpy.float64]

# Dtypes of dtypes
reveal_type(np.dtype(np.dtype(np.float64)))  # E: numpy.dtype[numpy.float64*]

# Parameterized dtypes
reveal_type(np.dtype("S8"))  # E: numpy.dtype

# Void
reveal_type(np.dtype(("U", 10)))  # E: numpy.dtype[numpy.void]
