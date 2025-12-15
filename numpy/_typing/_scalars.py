from typing import Any

import numpy as np

# NOTE: `_StrLike_co` and `_BytesLike_co` are pointless, as `np.str_` and
# `np.bytes_` are already subclasses of their builtin counterpart
type _CharLike_co = str | bytes

# The `<X>Like_co` type-aliases below represent all scalars that can be
# coerced into `<X>` (with the casting rule `same_kind`)
type _BoolLike_co = bool | np.bool
type _UIntLike_co = bool | np.unsignedinteger | np.bool
type _IntLike_co = int | np.integer | np.bool
type _FloatLike_co = float | np.floating | np.integer | np.bool
type _ComplexLike_co = complex | np.number | np.bool
type _NumberLike_co = _ComplexLike_co
type _TD64Like_co = int | np.timedelta64 | np.integer | np.bool
# `_VoidLike_co` is technically not a scalar, but it's close enough
type _VoidLike_co = tuple[Any, ...] | np.void
type _ScalarLike_co = complex | str | bytes | np.generic
