import sys
from typing import Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    if sys.version_info >= (3, 8):
        from typing import SupportsIndex
    else:
        from typing_extensions import Protocol
        class SupportsIndex(Protocol):
            def __index__(self) -> int: ...

_Shape = Tuple[int, ...]

# Anything that can be coerced to a shape tuple
_ShapeLike = Union["SupportsIndex", Sequence["SupportsIndex"]]
