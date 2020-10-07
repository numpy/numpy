import numpy as np

x = np.complex64(3 + 2j)

reveal_type(x.real)  # E: numpy.float32
reveal_type(x.imag)  # E: numpy.float32

reveal_type(x.real.real)  # E: numpy.float32
reveal_type(x.real.imag)  # E: numpy.float32

reveal_type(x.itemsize)  # E: int
reveal_type(x.shape)  # E: tuple[builtins.int]
reveal_type(x.strides)  # E: tuple[builtins.int]

reveal_type(np.complex64().real)  # E: numpy.float32
reveal_type(np.complex128().imag)  # E: numpy.float64

reveal_type(np.unicode_('foo'))  # E: numpy.str_
