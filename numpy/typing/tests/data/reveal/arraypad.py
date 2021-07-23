from typing import List, Any, Mapping, Tuple
from typing_extensions import SupportsIndex

import numpy as np
import numpy.typing as npt


def mode_func(
    ar: npt.NDArray[np.number[Any]],
    width: Tuple[int, int],
    iaxis: SupportsIndex,
    kwargs: Mapping[str, Any],
) -> None:
    ...


AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_LIKE: List[int]

reveal_type(
    np.pad(AR_i8, (2, 3), "constant")
)  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(
    np.pad(AR_LIKE, (2, 3), "constant")
)  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(
    np.pad(AR_f8, (2, 3), mode_func)
)  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(
    np.pad(AR_f8, (2, 3), mode_func, a=1, b=2)
)  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
