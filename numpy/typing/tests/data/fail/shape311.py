import sys

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 11):
    npt.Array[str, str, np.float64]   # E: Value of type variable
