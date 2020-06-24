import numpy as np


class Test:
    not_dtype = float


np.dtype(Test())  # E: Argument 1 to "dtype" has incompatible type
