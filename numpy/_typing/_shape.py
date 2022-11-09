from typing import Sequence, SupportsIndex, Tuple, Union

_Shape = Tuple[int, ...]

# Anything that can be coerced to a shape tuple
_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
