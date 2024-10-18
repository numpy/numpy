import numpy as np
assert np.ma.MaskedArray([1,1,1], [True, False, True], dtype='uint64').filled().dtype == np.dtype("uint64")