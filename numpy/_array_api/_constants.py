from ._array_object import ndarray
from ._dtypes import float64

import numpy as np

e = ndarray._new(np.array(np.e, dtype=float64))
inf = ndarray._new(np.array(np.inf, dtype=float64))
nan = ndarray._new(np.array(np.nan, dtype=float64))
pi = ndarray._new(np.array(np.pi, dtype=float64))
