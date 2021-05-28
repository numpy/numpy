import sys
from typing import Sequence, Tuple, Union

from . import _HAS_TYPING_EXTENSIONS

if sys.version_info >= (3, 8):
    from typing import SupportsIndex
elif _HAS_TYPING_EXTENSIONS:
    from typing_extensions import SupportsIndex
else:
    SupportsIndex = NotImplemented

_Shape = Tuple[int, ...]

# Anything that can be coerced to a shape tuple
_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
