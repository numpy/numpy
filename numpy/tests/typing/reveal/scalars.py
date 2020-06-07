import numpy as np

x = np.complex64(3 + 2j)

reveal_type(x.real)  # E: numpy.float32
reveal_type(x.imag)  # E: numpy.float32

reveal_type(x.real.real)  # E: numpy.float32
reveal_type(x.real.imag)  # E: numpy.float32

reveal_type(x.itemsize)  # E: int
reveal_type(x.shape)  # E: tuple[builtins.int]
reveal_type(x.strides)  # E: tuple[builtins.int]

# Time structures
dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

reveal_type(dt + td)  # E: numpy.datetime64
reveal_type(dt + 1)  # E: numpy.datetime64
reveal_type(dt - dt)  # E: numpy.timedelta64
reveal_type(dt - 1)  # E: numpy.timedelta64

reveal_type(td + td)  # E: numpy.timedelta64
reveal_type(td + 1)  # E: numpy.timedelta64
reveal_type(td - td)  # E: numpy.timedelta64
reveal_type(td - 1)  # E: numpy.timedelta64
reveal_type(td / 1.0)  # E: numpy.timedelta64
reveal_type(td / td)  # E: float
reveal_type(td % td)  # E: numpy.timedelta64
