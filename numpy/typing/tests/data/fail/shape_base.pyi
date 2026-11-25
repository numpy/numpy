import numpy as np
import numpy.typing as npt

class DTypeLike:
    dtype: np.dtype[np.int_]

_i8_2d: npt.Array2D[np.int64]
_i8_0d: np.ndarray[tuple[()], np.dtype[np.int64]]
_py_f_1d: list[float]

dtype_like: DTypeLike

def _func_axis_f(limestone: npt.Array2D[np.int64], axis: int) -> float: ...
def _func_axis_f64(axolotl: npt.Array2D[np.int64], axis: int) -> np.float64: ...
def _func_axis_f64_2d(foam: npt.Array3D[np.int64], axis: int) -> npt.Array2D[np.float64]: ...

###

np.expand_dims(dtype_like, (5, 10))  # type: ignore[call-overload]

np.unstack(_i8_0d)  # type: ignore[arg-type]
np.unstack(_py_f_1d)  # type: ignore[call-overload]

np.apply_over_axes(np.sum, _py_f_1d, 0)  # type: ignore[call-overload]
np.apply_over_axes(_func_axis_f, _i8_2d, 0)  # type: ignore[arg-type]
np.apply_over_axes(_func_axis_f64, _i8_2d, 0)  # type: ignore[arg-type]
np.apply_over_axes(_func_axis_f64_2d, _i8_2d, 0)  # type: ignore[arg-type]
