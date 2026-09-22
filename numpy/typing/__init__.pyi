from typing import Final

from numpy._pytesttester import PytestTester
from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray  # type: ignore[deprecated]

__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

test: Final[PytestTester] = ...
