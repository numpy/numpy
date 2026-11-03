from typing import Any, assert_type, type_check_only

import numpy as np
import numpy.lib.recfunctions as rfn
import numpy.typing as npt

###

type _RecArray1D[ST: np.generic] = np.recarray[tuple[int], np.dtype[ST]]

_i64_1d: npt.Array1D[np.int64]
_f64_1d: npt.Array1D[np.float64]
_void_1d: npt.Array1D[np.void]

@type_check_only
def _func_f64(a: npt.NDArray[Any], /, *, axis: int) -> npt.Array1D[np.float64]: ...

###

assert_type(rfn.merge_arrays((_i64_1d, _f64_1d), -1, False, False, True), _RecArray1D[np.record])
assert_type(rfn.merge_arrays((_i64_1d, _f64_1d), asrecarray=True), _RecArray1D[np.record])
assert_type(rfn.merge_arrays((_i64_1d, _f64_1d), usemask=True), npt.Array1D[np.void])

assert_type(rfn.apply_along_fields(_func_f64, _void_1d), npt.Array1D[np.float64])

assert_type(rfn.rec_append_fields(_void_1d, "y", _f64_1d), _RecArray1D[np.record])
assert_type(rfn.rec_append_fields(_void_1d, ["y", "z"], [_f64_1d, _i64_1d], [np.float64, "i8"]), _RecArray1D[np.record])
