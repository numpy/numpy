from typing import Final

from numpy._pytesttester import PytestTester
from numpy._typing import (  # type: ignore[deprecated]
    Array0D,
    Array1D,
    Array2D,
    Array3D,
    Array4D,
    ArrayLike,
    DTypeLike,
    NBitBase,
    NDArray,
)

__all__ = ["Array0D", "Array1D", "Array2D", "Array3D", "Array4D", "ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

test: Final[PytestTester] = ...
