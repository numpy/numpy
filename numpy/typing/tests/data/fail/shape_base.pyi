import numpy as np

class DTypeLike:
    dtype: np.dtype[np.int_]

_i8_2d: _Array2D[np.int64]
_i8_0d: np.ndarray[tuple[()], np.dtype[np.int64]]
_py_f_1d: list[float]

dtype_like: DTypeLike

type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]

def _func_axis_f(limestone: _Array2D[np.int64], axis: int) -> float: ...
def _func_axis_f64(axolotl: _Array2D[np.int64], axis: int) -> np.float64: ...
def _func_axis_f64_2d(foam: _Array3D[np.int64], axis: int) -> _Array2D[np.float64]: ...

###

np.expand_dims(dtype_like, (5, 10))  # type: ignore[call-overload]

np.unstack(_i8_0d)  # type: ignore[arg-type]
np.unstack(_py_f_1d)  # type: ignore[call-overload]

np.apply_over_axes(np.sum, _py_f_1d, 0)  # type: ignore[call-overload]
np.apply_over_axes(_func_axis_f, _i8_2d, 0)  # type: ignore[arg-type]
np.apply_over_axes(_func_axis_f64, _i8_2d, 0)  # type: ignore[arg-type]
np.apply_over_axes(_func_axis_f64_2d, _i8_2d, 0)  # type: ignore[arg-type]
