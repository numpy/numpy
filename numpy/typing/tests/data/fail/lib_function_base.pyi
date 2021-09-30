from typing import Any

import numpy as np
import numpy.typing as npt

AR_m: npt.NDArray[np.timedelta64]
AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]

np.average(AR_m)  # E: incompatible type
np.select(1, [AR_f8])  # E: incompatible type
np.angle(AR_m)  # E: incompatible type
np.unwrap(AR_m)  # E: incompatible type
np.unwrap(AR_c16)  # E: incompatible type
np.trim_zeros(1)  # E: incompatible type
np.place(1, [True], 1.5)  # E: incompatible type
np.vectorize(1)  # E: incompatible type
np.add_newdoc("__main__", 1.5, "docstring")  # E: incompatible type
