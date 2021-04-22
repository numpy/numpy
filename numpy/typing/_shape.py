import sys
from typing import Sequence, Tuple, Union

if sys.version_info >= (3, 8):
    from typing import SupportsIndex
else:
    try:
        from typing_extensions import SupportsIndex
    except ImportError:
        SupportsIndex = NotImplemented

_Shape = Tuple[int, ...]

# Anything that can be coerced to a shape tuple
_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
