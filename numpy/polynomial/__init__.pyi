from typing import Final, Literal

from .polynomial import Polynomial
from .chebyshev import Chebyshev
from .legendre import Legendre
from .hermite import Hermite
from .hermite_e import HermiteE
from .laguerre import Laguerre
from . import polynomial, chebyshev, legendre, hermite, hermite_e, laguerre

__all__ = [
    "set_default_printstyle",
    "polynomial", "Polynomial",  # noqa: F822
    "chebyshev", "Chebyshev",  # noqa: F822
    "legendre", "Legendre",  # noqa: F822
    "hermite", "Hermite",  # noqa: F822
    "hermite_e", "HermiteE",  # noqa: F822
    "laguerre", "Laguerre",  # noqa: F822
]

def set_default_printstyle(style: Literal["ascii", "unicode"]) -> None: ...

from numpy._pytesttester import PytestTester as _PytestTester
test: Final[_PytestTester]
