from typing import List
import numpy as np
import numpy.typing as npt

i8: np.int64

AR_b: npt.NDArray[np.bool_]
AR_u1: npt.NDArray[np.uint8]
AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_M: npt.NDArray[np.datetime64]

AR_LIKE_f: List[float]

np.where(AR_b, 1)  # E: No overload variant

np.can_cast(AR_f8, 1)  # E: incompatible type

np.vdot(AR_M, AR_M)  # E: incompatible type

np.copyto(AR_LIKE_f, AR_f8)  # E: incompatible type

np.putmask(AR_LIKE_f, [True, True, False], 1.5)  # E: incompatible type

np.packbits(AR_f8)  # E: incompatible type
np.packbits(AR_u1, bitorder=">")  # E: incompatible type

np.unpackbits(AR_i8)  # E: incompatible type
np.unpackbits(AR_u1, bitorder=">")  # E: incompatible type

np.shares_memory(1, 1, max_work=i8)  # E: incompatible type
np.may_share_memory(1, 1, max_work=i8)  # E: incompatible type
