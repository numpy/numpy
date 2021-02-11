from typing import List
import numpy as np

AR_LIKE_i: List[int]
AR_LIKE_f: List[float]

np.unravel_index(AR_LIKE_f, (1, 2, 3))  # E: incompatible type
np.ravel_multi_index(AR_LIKE_i, (1, 2, 3), mode="bob")  # E: incompatible type
np.mgrid[1]  # E: Invalid index type
np.mgrid[...]  # E: Invalid index type
np.ogrid[1]  # E: Invalid index type
np.ogrid[...]  # E: Invalid index type
np.index_exp[1]  # E: No overload variant
np.index_exp[...]  # E: No overload variant
np.s_[1]  # E: cannot be "int"
np.s_[...]  # E: cannot be "ellipsis"
np.fill_diagonal(AR_LIKE_f, 2)  # E: incompatible type
np.diag_indices(1.0)  # E: incompatible type
