from typing import Final

from numpy._pytesttester import PytestTester
from numpy._typing import (  # type: ignore[deprecated]
    ArrayLike,
    DTypeLike,
    NBitBase,
    NDArray,
)

__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

test: Final[PytestTester] = ...
