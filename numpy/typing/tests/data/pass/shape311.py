import sys
from typing import Literal, NewType, TypeVar, TypeVarTuple

import numpy as np
import numpy.typing as npt
from typing_extensions import assert_type

if sys.version_info >= (3, 11):
    DType = TypeVar("DType", bound=np.generic)
    Shapes = TypeVarTuple("Shapes")

    def stack(
        a: npt.Array[*Shapes, DType],
        b: npt.Array[*Shapes, DType],
    ) -> npt.Array[Literal[2], *Shapes, DType]:
        return np.stack((a, b))

    arr: npt.Array[Literal[3], Literal[4], np.uint16]
    arr = np.arange(12, dtype=np.uint16).reshape((3, 4))

    double_arr = stack(arr, arr)
    assert_type(double_arr, npt.Array[Literal[2], Literal[3], Literal[4], np.uint16])

    Length = NewType("Length", int)
    Width = NewType("Width", int)
    arr2: npt.Array[Length, Width, np.int8] = np.array([[0]])
    assert_type(arr2, npt.Array[Length, Width, np.int8])
