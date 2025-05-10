import numpy as np
from numpy.typing import NDArray

arr = np.array([1, 2, 3])

out_int: NDArray[np.int64] = np.empty(())
_ = np.argmin(arr, out=out_int)  # expected to pass
_ = np.argmax(arr, out=out_int)  # expected to pass

out_bool: NDArray[np.bool_] = np.empty(())
_ = np.argmin(arr, out=out_bool)  # expected to pass
_ = np.argmax(arr, out=out_bool)  # expected to pass

out_bad: NDArray[np.float64] = np.empty(())
_ = np.argmin(arr, out=out_bad)  # should fail static typing
_ = np.argmax(arr, out=out_bad)  # should fail static typing