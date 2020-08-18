from typing import Union, Tuple, Any
from datetime import datetime, timedelta

import numpy as np

_DatetimeLike = Union[datetime, np.datetime64]
_TimedeltaLike = Union[timedelta, np.timedelta64]

# NOTE: mypy has special rules which ensures that `int` is treated as
# a `float` (and `complex`) superclass, however these rules do not apply
# to its `np.generic` counterparts.
# Solution: manually add them them to the unions below
_IntLike = Union[int, np.integer]
_FloatLike = Union[float, np.floating, np.integer]
_ComplexLike = Union[complex, np.complexfloating, np.floating, np.integer]
_BoolLike = Union[bool, np.bool_]
_NumberLike = Union[int, float, complex, timedelta, np.number, np.bool_]

_StrLike = Union[str, np.str_]
_BytesLike = Union[bytes, np.bytes_]
_CharacterLike = Union[str, bytes, np.character]

_VoidLikeNested = Any  # TODO: wait for support for recursive types
_ScalarLike = Union[
    datetime,
    timedelta,
    int,
    float,
    complex,
    str,
    bytes,
    Tuple[_VoidLikeNested, ...],
    np.generic,
]

# _VoidLike is technically not a scalar, but it's close enough
_VoidLike = Union[Tuple[_ScalarLike, ...], np.void]
