from typing import Any, List

from numpy._pytesttester import PytestTester

from numpy import (
    matrix as matrix,
)

__all__: List[str]
test: PytestTester

def bmat(obj, ldict=..., gdict=...): ...
def asmatrix(data, dtype=...): ...
mat = asmatrix
