
import numpy as np
import numpy.typing as npt


# Fails for different reason with same syntax < 3.11
npt.Array[str, str, np.float64]   # E: Value of type variable
