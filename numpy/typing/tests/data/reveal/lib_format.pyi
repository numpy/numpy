from typing import Any, assert_type

import numpy as np
import numpy.lib.format as fmt

###

_dt_f8: np.dtype[np.float64]
_dt_void: np.dtype[np.void]
_dt_any: np.dtype[Any]

_descr: list[tuple[str, str]]

###

assert_type(fmt.dtype_to_descr(_dt_f8), str)
assert_type(fmt.dtype_to_descr(_dt_void), str | list[tuple[Any, ...]])
assert_type(fmt.dtype_to_descr(_dt_any), Any)

assert_type(fmt.descr_to_dtype("<f8"), np.dtype[Any])
assert_type(fmt.descr_to_dtype(_descr), np.dtype[np.void])
