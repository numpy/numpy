from typing import Any

import numpy as np
from numpy.typing import DtypeLike, _SupportsArray


class Index:
    def __index__(self) -> int:
        ...


a: "np.flatiter[np.ndarray]"
supports_array: _SupportsArray

a.base = Any  # E: Property "base" defined in "flatiter" is read-only
a.copy = Any  # E: Property "copy" defined in "flatiter" is read-only
a.coords = Any  # E: Property "coords" defined in "flatiter" is read-only
a.index = Any  # E: Property "index" defined in "flatiter" is read-only
a[np.bool_()]  # E: No overload variant of "__getitem__"
a[Index()]  # E: No overload variant of "__getitem__"
a[supports_array]  # E: No overload variant of "__getitem__"
