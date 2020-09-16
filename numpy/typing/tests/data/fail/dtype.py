import numpy as np


class Test:
    not_dtype = float


np.dtype(Test())  # E: Argument 1 to "dtype" has incompatible type

np.dtype(
    {  # E: Argument 1 to "dtype" has incompatible type
        "field1": (float, 1),
        "field2": (int, 3),
    }
)
