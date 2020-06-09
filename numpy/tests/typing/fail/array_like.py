from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
else:
    ArrayLike = Any


class A:
    pass


x1: ArrayLike = (i for i in range(10))  # E: Incompatible types in assignment
x2: ArrayLike = A()  # E: Incompatible types in assignment
x3: ArrayLike = {1: "foo", 2: "bar"}  # E: Incompatible types in assignment

scalar = np.int64(1)
scalar.__array__(dtype=np.float64)  # E: Unexpected keyword argument
array = np.array([1])
array.__array__(dtype=np.float64)  # E: Unexpected keyword argument
