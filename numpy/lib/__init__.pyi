import math as math
from typing import Any

from numpy._pytesttester import PytestTester

from numpy import (
    ndenumerate as ndenumerate,
    ndindex as ndindex,
)

from numpy.version import version

from numpy.lib import (
    format as format,
    mixins as mixins,
    scimath as scimath,
    stride_tricks as stride_tricks,
    npyio as npyio,
)

from numpy.lib._version import (
    NumpyVersion as NumpyVersion,
)

from numpy.lib.arrayterator import (
    Arrayterator as Arrayterator,
)

from numpy._core.multiarray import (
    add_docstring as add_docstring,
    tracemalloc_domain as tracemalloc_domain,
)

from numpy._core.function_base import (
    add_newdoc as add_newdoc,
)

__all__: list[str]
test: PytestTester

__version__ = version
emath = scimath
