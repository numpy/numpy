import numpy as np
import numpy.typing as npt

_i64_nd: npt.NDArray[np.int64]
_m64_nd: npt.NDArray[np.timedelta64]
_U_nd: npt.NDArray[np.str_]

np.einsum("i,i->i", _i64_nd, _m64_nd)  # type: ignore[arg-type]
np.einsum("i,i->i", _i64_nd, _i64_nd, out=_U_nd)  # type: ignore[type-var]
np.einsum("i,i->i", _i64_nd, _i64_nd, out=_U_nd, casting="unsafe")  # type: ignore[call-overload]
