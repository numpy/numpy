from typing import assert_type

import numpy as np
import numpy.lib.recfunctions as rfn
import numpy.typing as npt

###

_i64_1d: npt.Array1D[np.int64]
_f64_1d: npt.Array1D[np.float64]

###

assert_type(rfn.merge_arrays((_i64_1d, _f64_1d), -1, False, False, True), np.recarray[tuple[int], np.dtype[np.record]])
assert_type(rfn.merge_arrays((_i64_1d, _f64_1d), asrecarray=True), np.recarray[tuple[int], np.dtype[np.record]])
assert_type(rfn.merge_arrays((_i64_1d, _f64_1d), usemask=True), npt.Array1D[np.void])
