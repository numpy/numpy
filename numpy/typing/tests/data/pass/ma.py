from __future__ import annotations

from typing import Any, TypeAlias, TypeVar

import numpy as np
import numpy.ma
import numpy.typing as npt
from numpy import dtype, generic
from numpy._typing import _Shape

_ScalarT = TypeVar("_ScalarT", bound=generic)
MaskedArray: TypeAlias = np.ma.MaskedArray[_Shape, dtype[_ScalarT]]
ar_b: npt.NDArray[np.bool] = np.array([True, False, True])
m: np.ma.MaskedArray[Any, np.dtype[np.float64]] = np.ma.masked_array(
    [1.5, 2, 3], mask=[True, False, True]
)

m.mask = ar_b
m.mask = np.False_

MAR_b: MaskedArray[np.bool] = np.ma.MaskedArray([True])
MAR_u: MaskedArray[np.uint32] = np.ma.MaskedArray([1], dtype=np.uint32)
MAR_i: MaskedArray[np.int64] = np.ma.MaskedArray([1])
MAR_f: MaskedArray[np.float64] = np.ma.MaskedArray([1.0])
MAR_c: MaskedArray[np.complex128] = np.ma.MaskedArray([1j])
MAR_td64: MaskedArray[np.timedelta64] = np.ma.MaskedArray([np.timedelta64(1, "D")])
MAR_M_dt64: MaskedArray[np.datetime64] = np.ma.MaskedArray([np.datetime64(1, "D")])

AR_LIKE_b = [True]
AR_LIKE_u = [np.uint32(1)]
AR_LIKE_i = [1]
AR_LIKE_f = [1.0]
AR_LIKE_c = [1j]
AR_LIKE_m = [np.timedelta64(1, "D")]
AR_LIKE_M = [np.datetime64(1, "D")]

# Inplace addition

MAR_b += AR_LIKE_b

MAR_u += AR_LIKE_b
MAR_u += AR_LIKE_u

MAR_i += AR_LIKE_b
MAR_i += 2
MAR_i += AR_LIKE_i

MAR_f += AR_LIKE_b
MAR_f += 2
MAR_f += AR_LIKE_u
MAR_f += AR_LIKE_i
MAR_f += AR_LIKE_f

MAR_c += AR_LIKE_b
MAR_c += AR_LIKE_u
MAR_c += AR_LIKE_i
MAR_c += AR_LIKE_f
MAR_c += AR_LIKE_c

MAR_td64 += AR_LIKE_b
MAR_td64 += AR_LIKE_u
MAR_td64 += AR_LIKE_i
MAR_td64 += AR_LIKE_m
MAR_M_dt64 += AR_LIKE_b
MAR_M_dt64 += AR_LIKE_u
MAR_M_dt64 += AR_LIKE_i
MAR_M_dt64 += AR_LIKE_m

# Inplace subtraction

MAR_u -= AR_LIKE_b
MAR_u -= AR_LIKE_u

MAR_i -= AR_LIKE_b
MAR_i -= AR_LIKE_i

MAR_f -= AR_LIKE_b
MAR_f -= AR_LIKE_u
MAR_f -= AR_LIKE_i
MAR_f -= AR_LIKE_f

MAR_c -= AR_LIKE_b
MAR_c -= AR_LIKE_u
MAR_c -= AR_LIKE_i
MAR_c -= AR_LIKE_f
MAR_c -= AR_LIKE_c

MAR_td64 -= AR_LIKE_b
MAR_td64 -= AR_LIKE_u
MAR_td64 -= AR_LIKE_i
MAR_td64 -= AR_LIKE_m
MAR_M_dt64 -= AR_LIKE_b
MAR_M_dt64 -= AR_LIKE_u
MAR_M_dt64 -= AR_LIKE_i
MAR_M_dt64 -= AR_LIKE_m

# Inplace floor division

MAR_f //= AR_LIKE_b
MAR_f //= 2
MAR_f //= AR_LIKE_u
MAR_f //= AR_LIKE_i
MAR_f //= AR_LIKE_f

MAR_td64 //= AR_LIKE_i

# Inplace true division

MAR_f /= AR_LIKE_b
MAR_f /= 2
MAR_f /= AR_LIKE_u
MAR_f /= AR_LIKE_i
MAR_f /= AR_LIKE_f

MAR_c /= AR_LIKE_b
MAR_c /= AR_LIKE_u
MAR_c /= AR_LIKE_i
MAR_c /= AR_LIKE_f
MAR_c /= AR_LIKE_c

MAR_td64 /= AR_LIKE_i

# Inplace multiplication

MAR_b *= AR_LIKE_b

MAR_u *= AR_LIKE_b
MAR_u *= AR_LIKE_u

MAR_i *= AR_LIKE_b
MAR_i *= 2
MAR_i *= AR_LIKE_i

MAR_f *= AR_LIKE_b
MAR_f *= 2
MAR_f *= AR_LIKE_u
MAR_f *= AR_LIKE_i
MAR_f *= AR_LIKE_f

MAR_c *= AR_LIKE_b
MAR_c *= AR_LIKE_u
MAR_c *= AR_LIKE_i
MAR_c *= AR_LIKE_f
MAR_c *= AR_LIKE_c

MAR_td64 *= AR_LIKE_b
MAR_td64 *= AR_LIKE_u
MAR_td64 *= AR_LIKE_i
MAR_td64 *= AR_LIKE_f

# Inplace power

MAR_u **= AR_LIKE_b
MAR_u **= AR_LIKE_u

MAR_i **= AR_LIKE_b
MAR_i **= AR_LIKE_i

MAR_f **= AR_LIKE_b
MAR_f **= AR_LIKE_u
MAR_f **= AR_LIKE_i
MAR_f **= AR_LIKE_f

MAR_c **= AR_LIKE_b
MAR_c **= AR_LIKE_u
MAR_c **= AR_LIKE_i
MAR_c **= AR_LIKE_f
MAR_c **= AR_LIKE_c
