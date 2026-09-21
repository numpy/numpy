import numpy as np

class DTypeLike:
    dtype: np.dtype[np.int_]

dtype_like: DTypeLike

np.expand_dims(dtype_like, (5, 10))  # type: ignore[call-overload]

###

_i8_0d: np.ndarray[tuple[()], np.dtype[np.int64]]
_py_f_1d: list[float]

np.unstack(_i8_0d)  # type: ignore[arg-type]
np.unstack(_py_f_1d)  # type: ignore[call-overload]
