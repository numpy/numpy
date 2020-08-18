from typing import Union
from datetime import datetime, timedelta

import numpy as np

_DatetimeLike = Union[datetime, np.datetime64]
_TimedeltaLike = Union[timedelta, np.timedelta64]

_IntLike = Union[int, np.integer]
_FloatLike = Union[float, np.floating]
_ComplexLike = Union[complex, np.complexfloating]
_BoolLike = Union[bool, np.bool_]
_NumberLike = Union[int, float, complex, timedelta, np.number, np.bool_]

_StrLike = Union[str, np.str_]
_BytesLike = Union[bytes, np.bytes_]
_CharacterLike = Union[str, bytes, np.character]

_VoidLike = Union[tuple, np.void]
