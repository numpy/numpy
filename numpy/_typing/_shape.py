from collections.abc import Sequence
from typing import Any, SupportsIndex

type _Shape = tuple[int, ...]
type _AnyShape = tuple[Any, ...]

# Anything that can be coerced to a shape tuple
type _ShapeLike = SupportsIndex | Sequence[SupportsIndex]
