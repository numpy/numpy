from collections.abc import Sequence
from typing import Any, overload

import numpy as np
from numpy import add, equal, greater, greater_equal, less, less_equal, not_equal
from numpy._globals import _NoValueType
from numpy._typing import (
    NDArray,
    _AnyShape,
    _ArrayLikeAnyString_co as UST_co,
    _ArrayLikeBytes_co as S_co,
    _ArrayLikeInt_co as i_co,
    _ArrayLikeStr_co as U_co,
    _ArrayLikeString_co as T_co,
    _CharLike_co,
    _IntLike_co,
    _NestedSequence,
    _Shape,
    _SupportsArray,
)

from .defchararray import mod
from .umath import (
    isalnum,
    isalpha,
    isdecimal,
    isdigit,
    islower,
    isnumeric,
    isspace,
    istitle,
    isupper,
    str_len,
)

__all__ = [
    "add",
    "capitalize",
    "center",
    "count",
    "decode",
    "encode",
    "endswith",
    "equal",
    "expandtabs",
    "find",
    "greater",
    "greater_equal",
    "index",
    "isalnum",
    "isalpha",
    "isdecimal",
    "isdigit",
    "islower",
    "isnumeric",
    "isspace",
    "istitle",
    "isupper",
    "less",
    "less_equal",
    "ljust",
    "lower",
    "lstrip",
    "mod",
    "multiply",
    "not_equal",
    "partition",
    "replace",
    "rfind",
    "rindex",
    "rjust",
    "rpartition",
    "rstrip",
    "startswith",
    "str_len",
    "strip",
    "swapcase",
    "title",
    "translate",
    "upper",
    "zfill",
    "slice",
]

type _Array0D[ScalarT: np.generic] = np.ndarray[tuple[()], np.dtype[ScalarT]]
type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

type _StringDTypeArray = np.ndarray[_AnyShape, np.dtypes.StringDType]
type _StringDTypeSupportsArray = _SupportsArray[np.dtypes.StringDType]
type _StringDTypeOrUnicodeArray = NDArray[np.str_] | _StringDTypeArray

type _tuple3[T] = tuple[T, T, T]

###

@overload
def multiply(a: U_co, i: i_co) -> NDArray[np.str_]: ...
@overload
def multiply(a: S_co, i: i_co) -> NDArray[np.bytes_]: ...
@overload
def multiply(a: _StringDTypeSupportsArray, i: i_co) -> _StringDTypeArray: ...
@overload
def multiply(a: T_co, i: i_co) -> _StringDTypeOrUnicodeArray: ...

#
@overload  # Nd str | vstr
def find[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.str_] | np.dtypes.StringDType],
    sub: str,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # Nd bytes
def find[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    sub: bytes,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # 0d
def find[T: (bytes, str)](
    a: T,
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.int_: ...
@overload  # 1d
def find[T: (bytes, str)](
    a: list[T],
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array1D[np.int_]: ...
@overload  # 2d
def find[T: (bytes, str)](
    a: Sequence[list[T]],
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array2D[np.int_]: ...
@overload  # ?d str | vstr  (fallback)
def find(
    a: U_co | T_co,
    sub: U_co | T_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.int_] | Any: ...
@overload  # ?d bytes  (fallback)
def find(
    a: S_co,
    sub: S_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.int_] | Any: ...

# keep in sync with `find`
@overload  # Nd str | vstr
def rfind[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.str_] | np.dtypes.StringDType],
    sub: str,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # Nd bytes
def rfind[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    sub: bytes,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # 0d
def rfind[T: (bytes, str)](
    a: T,
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.int_: ...
@overload  # 1d
def rfind[T: (bytes, str)](
    a: list[T],
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array1D[np.int_]: ...
@overload  # 2d
def rfind[T: (bytes, str)](
    a: Sequence[list[T]],
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array2D[np.int_]: ...
@overload  # ?d str | vstr  (fallback)
def rfind(
    a: U_co | T_co,
    sub: U_co | T_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.int_] | Any: ...
@overload  # ?d bytes  (fallback)
def rfind(
    a: S_co,
    sub: S_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.int_] | Any: ...

# keep in sync with `find`
@overload  # Nd str | vstr
def index[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.str_] | np.dtypes.StringDType],
    sub: str,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # Nd bytes
def index[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    sub: bytes,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # 0d
def index[T: (bytes, str)](
    a: T,
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.int_: ...
@overload  # 1d
def index[T: (bytes, str)](
    a: list[T],
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array1D[np.int_]: ...
@overload  # 2d
def index[T: (bytes, str)](
    a: Sequence[list[T]],
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array2D[np.int_]: ...
@overload  # ?d str | vstr  (fallback)
def index(
    a: U_co | T_co,
    sub: U_co | T_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.int_] | Any: ...
@overload  # ?d bytes  (fallback)
def index(
    a: S_co,
    sub: S_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.int_] | Any: ...

# keep in sync with `find`
@overload  # Nd str | vstr
def rindex[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.str_] | np.dtypes.StringDType],
    sub: str,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # Nd bytes
def rindex[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    sub: bytes,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # 0d
def rindex[T: (bytes, str)](
    a: T,
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.int_: ...
@overload  # 1d
def rindex[T: (bytes, str)](
    a: list[T],
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array1D[np.int_]: ...
@overload  # 2d
def rindex[T: (bytes, str)](
    a: Sequence[list[T]],
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array2D[np.int_]: ...
@overload  # ?d str | vstr  (fallback)
def rindex(
    a: U_co | T_co,
    sub: U_co | T_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.int_] | Any: ...
@overload  # ?d bytes  (fallback)
def rindex(
    a: S_co,
    sub: S_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.int_] | Any: ...

# keep in sync with `find`
@overload  # Nd str | vstr
def count[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.str_] | np.dtypes.StringDType],
    sub: str,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # Nd bytes
def count[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    sub: bytes,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # 0d
def count[T: (bytes, str)](
    a: T,
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.int_: ...
@overload  # 1d
def count[T: (bytes, str)](
    a: list[T],
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array1D[np.int_]: ...
@overload  # 2d
def count[T: (bytes, str)](
    a: Sequence[list[T]],
    sub: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array2D[np.int_]: ...
@overload  # ?d str | vstr  (fallback)
def count(
    a: U_co | T_co,
    sub: U_co | T_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.int_] | Any: ...
@overload  # ?d bytes  (fallback)
def count(
    a: S_co,
    sub: S_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.int_] | Any: ...

# keep in sync with `endswith`
@overload  # Nd str | vstr
def startswith[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.str_] | np.dtypes.StringDType],
    prefix: str,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # Nd bytes
def startswith[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    prefix: bytes,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # 0d
def startswith[T: (bytes, str)](
    a: T,
    prefix: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.bool: ...
@overload  # 1d
def startswith[T: (bytes, str)](
    a: list[T],
    prefix: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array1D[np.bool]: ...
@overload  # 2d
def startswith[T: (bytes, str)](
    a: Sequence[list[T]],
    prefix: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array2D[np.bool]: ...
@overload  # ?d str | vstr  (fallback)
def startswith(
    a: U_co | T_co,
    prefix: U_co | T_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.bool] | Any: ...
@overload  # ?d bytes  (fallback)
def startswith(
    a: S_co,
    prefix: S_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.bool] | Any: ...

# keep in sync with `startswith`
@overload  # Nd str | vstr
def endswith[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.str_] | np.dtypes.StringDType],
    suffix: str,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # Nd bytes
def endswith[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    suffix: bytes,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # 0d
def endswith[T: (bytes, str)](
    a: T,
    suffix: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> np.bool: ...
@overload  # 1d
def endswith[T: (bytes, str)](
    a: list[T],
    suffix: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array1D[np.bool]: ...
@overload  # 2d
def endswith[T: (bytes, str)](
    a: Sequence[list[T]],
    suffix: T,
    start: _IntLike_co = 0,
    end: _IntLike_co | None = None,
) -> _Array2D[np.bool]: ...
@overload  # ?d str | vstr  (fallback)
def endswith(
    a: U_co | T_co,
    suffix: U_co | T_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.bool] | Any: ...
@overload  # ?d bytes  (fallback)
def endswith(
    a: S_co,
    suffix: S_co,
    start: i_co = 0,
    end: i_co | None = None,
) -> NDArray[np.bool] | Any: ...

#
def decode(
    a: S_co,
    encoding: str | None = None,
    errors: str | None = None,
) -> NDArray[np.str_]: ...
def encode(
    a: U_co | T_co,
    encoding: str | None = None,
    errors: str | None = None,
) -> NDArray[np.bytes_]: ...

@overload
def expandtabs(a: U_co, tabsize: i_co = 8) -> NDArray[np.str_]: ...
@overload
def expandtabs(a: S_co, tabsize: i_co = 8) -> NDArray[np.bytes_]: ...
@overload
def expandtabs(a: _StringDTypeSupportsArray, tabsize: i_co = 8) -> _StringDTypeArray: ...
@overload
def expandtabs(a: T_co, tabsize: i_co = 8) -> _StringDTypeOrUnicodeArray: ...

# keep in sync with `ljust` and `rjust`
@overload  # Nd str | bytes
def center[ShapeT: _Shape, CharT: np.character](
    a: np.ndarray[ShapeT, np.dtype[CharT]],
    width: _IntLike_co,
    fillchar: _CharLike_co = " ",
) -> np.ndarray[ShapeT, np.dtype[CharT]]: ...
@overload  # Nd vstr
def center[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtypes.StringDType],
    width: _IntLike_co,
    fillchar: str = " ",
) -> np.ndarray[ShapeT, np.dtypes.StringDType]: ...
@overload  # 0d str
def center(a: str, width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array0D[np.str_]: ...
@overload  # 0d bytes
def center(a: bytes, width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array0D[np.bytes_]: ...
@overload  # 1d str
def center(a: list[str], width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array1D[np.str_]: ...
@overload  # 1d bytes
def center(a: list[bytes], width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array1D[np.bytes_]: ...
@overload  # 2d str
def center(a: Sequence[list[str]], width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array2D[np.str_]: ...
@overload  # 2d bytes
def center(a: Sequence[list[bytes]], width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array2D[np.bytes_]: ...
@overload  # ?d str  (fallback)
def center(a: U_co, width: i_co, fillchar: U_co | S_co = " ") -> NDArray[np.str_]: ...
@overload  # ?d bytes  (fallback)
def center(a: S_co, width: i_co, fillchar: S_co | U_co = " ") -> NDArray[np.bytes_]: ...
@overload  # ?d vstr
def center(a: _StringDTypeSupportsArray, width: i_co, fillchar: U_co | T_co = " ") -> _StringDTypeArray: ...
@overload  # ?d vstr | str  (fallback)
def center(a: T_co, width: i_co, fillchar: U_co | T_co = " ") -> _StringDTypeOrUnicodeArray: ...

# keep in sync with `center`
@overload  # Nd str | bytes
def ljust[ShapeT: _Shape, CharT: np.character](
    a: np.ndarray[ShapeT, np.dtype[CharT]],
    width: _IntLike_co,
    fillchar: _CharLike_co = " ",
) -> np.ndarray[ShapeT, np.dtype[CharT]]: ...
@overload  # Nd vstr
def ljust[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtypes.StringDType],
    width: _IntLike_co,
    fillchar: str = " ",
) -> np.ndarray[ShapeT, np.dtypes.StringDType]: ...
@overload  # 0d str
def ljust(a: str, width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array0D[np.str_]: ...
@overload  # 0d bytes
def ljust(a: bytes, width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array0D[np.bytes_]: ...
@overload  # 1d str
def ljust(a: list[str], width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array1D[np.str_]: ...
@overload  # 1d bytes
def ljust(a: list[bytes], width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array1D[np.bytes_]: ...
@overload  # 2d str
def ljust(a: Sequence[list[str]], width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array2D[np.str_]: ...
@overload  # 2d bytes
def ljust(a: Sequence[list[bytes]], width: _IntLike_co, fillchar: _CharLike_co = " ") -> _Array2D[np.bytes_]: ...
@overload  # ?d str  (fallback)
def ljust(a: U_co, width: i_co, fillchar: U_co | S_co = " ") -> NDArray[np.str_]: ...
@overload  # ?d bytes  (fallback)
def ljust(a: S_co, width: i_co, fillchar: S_co | U_co = " ") -> NDArray[np.bytes_]: ...
@overload  # ?d vstr
def ljust(a: _StringDTypeSupportsArray, width: i_co, fillchar: U_co | T_co = " ") -> _StringDTypeArray: ...
@overload  # ?d vstr | str  (fallback)
def ljust(a: T_co, width: i_co, fillchar: U_co | T_co = " ") -> _StringDTypeOrUnicodeArray: ...

@overload
def rjust(a: U_co, width: i_co, fillchar: UST_co = " ") -> NDArray[np.str_]: ...
@overload
def rjust(a: S_co, width: i_co, fillchar: UST_co = " ") -> NDArray[np.bytes_]: ...
@overload
def rjust(a: _StringDTypeSupportsArray, width: i_co, fillchar: UST_co = " ") -> _StringDTypeArray: ...
@overload
def rjust(a: T_co, width: i_co, fillchar: UST_co = " ") -> _StringDTypeOrUnicodeArray: ...

# keep in sync with `strip`
@overload  # Nd str | vstr
def lstrip[ShapeT: _Shape, DTypeT: np.dtype[np.str_] | np.dtypes.StringDType](
    a: np.ndarray[ShapeT, DTypeT],
    chars: str | None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd bytes
def lstrip[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    chars: bytes | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.bytes_]]: ...
@overload  # 0d str
def lstrip(a: str, chars: str | None = None) -> np.str_: ...
@overload  # 0d bytes
def lstrip(a: bytes, chars: bytes | None = None) -> np.bytes_: ...
@overload  # 1d str
def lstrip(a: list[str], chars: str | None = None) -> _Array1D[np.str_]: ...
@overload  # 1d bytes
def lstrip(a: list[bytes], chars: bytes | None = None) -> _Array1D[np.bytes_]: ...
@overload  # 2d str
def lstrip(a: Sequence[list[str]], chars: str | None = None) -> _Array2D[np.str_]: ...
@overload  # 2d bytes
def lstrip(a: Sequence[list[bytes]], chars: bytes | None = None) -> _Array2D[np.bytes_]: ...
@overload  # ?d str  (fallback)
def lstrip(a: U_co, chars: U_co | None = None) -> NDArray[np.str_] | Any: ...
@overload  # ?d bytes  (fallback)
def lstrip(a: S_co, chars: S_co | None = None) -> NDArray[np.bytes_] | Any: ...
@overload  # ?d vstr
def lstrip(a: _StringDTypeSupportsArray, chars: T_co | None = None) -> _StringDTypeArray: ...
@overload  # ?d vstr | str  (fallback)
def lstrip(a: T_co, chars: T_co | None = None) -> _StringDTypeOrUnicodeArray | Any: ...

# keep in sync with `strip`
@overload  # Nd str | vstr
def rstrip[ShapeT: _Shape, DTypeT: np.dtype[np.str_] | np.dtypes.StringDType](
    a: np.ndarray[ShapeT, DTypeT],
    chars: str | None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd bytes
def rstrip[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    chars: bytes | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.bytes_]]: ...
@overload  # 0d str
def rstrip(a: str, chars: str | None = None) -> np.str_: ...
@overload  # 0d bytes
def rstrip(a: bytes, chars: bytes | None = None) -> np.bytes_: ...
@overload  # 1d str
def rstrip(a: list[str], chars: str | None = None) -> _Array1D[np.str_]: ...
@overload  # 1d bytes
def rstrip(a: list[bytes], chars: bytes | None = None) -> _Array1D[np.bytes_]: ...
@overload  # 2d str
def rstrip(a: Sequence[list[str]], chars: str | None = None) -> _Array2D[np.str_]: ...
@overload  # 2d bytes
def rstrip(a: Sequence[list[bytes]], chars: bytes | None = None) -> _Array2D[np.bytes_]: ...
@overload  # ?d str  (fallback)
def rstrip(a: U_co, chars: U_co | None = None) -> NDArray[np.str_] | Any: ...
@overload  # ?d bytes  (fallback)
def rstrip(a: S_co, chars: S_co | None = None) -> NDArray[np.bytes_] | Any: ...
@overload  # ?d vstr
def rstrip(a: _StringDTypeSupportsArray, chars: T_co | None = None) -> _StringDTypeArray: ...
@overload  # ?d vstr | str  (fallback)
def rstrip(a: T_co, chars: T_co | None = None) -> _StringDTypeOrUnicodeArray | Any: ...

# keep in sync with `lstrip` and `rstrip`
@overload  # Nd str | vstr
def strip[ShapeT: _Shape, DTypeT: np.dtype[np.str_] | np.dtypes.StringDType](
    a: np.ndarray[ShapeT, DTypeT],
    chars: str | None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # Nd bytes
def strip[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    chars: bytes | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.bytes_]]: ...
@overload  # 0d str
def strip(a: str, chars: str | None = None) -> np.str_: ...
@overload  # 0d bytes
def strip(a: bytes, chars: bytes | None = None) -> np.bytes_: ...
@overload  # 1d str
def strip(a: list[str], chars: str | None = None) -> _Array1D[np.str_]: ...
@overload  # 1d bytes
def strip(a: list[bytes], chars: bytes | None = None) -> _Array1D[np.bytes_]: ...
@overload  # 2d str
def strip(a: Sequence[list[str]], chars: str | None = None) -> _Array2D[np.str_]: ...
@overload  # 2d bytes
def strip(a: Sequence[list[bytes]], chars: bytes | None = None) -> _Array2D[np.bytes_]: ...
@overload  # ?d str  (fallback)
def strip(a: U_co, chars: U_co | None = None) -> NDArray[np.str_] | Any: ...
@overload  # ?d bytes  (fallback)
def strip(a: S_co, chars: S_co | None = None) -> NDArray[np.bytes_] | Any: ...
@overload  # ?d vstr
def strip(a: _StringDTypeSupportsArray, chars: T_co | None = None) -> _StringDTypeArray: ...
@overload  # ?d vstr | str  (fallback)
def strip(a: T_co, chars: T_co | None = None) -> _StringDTypeOrUnicodeArray | Any: ...

#
@overload
def zfill(a: U_co, width: i_co) -> NDArray[np.str_]: ...
@overload
def zfill(a: S_co, width: i_co) -> NDArray[np.bytes_]: ...
@overload
def zfill(a: _StringDTypeSupportsArray, width: i_co) -> _StringDTypeArray: ...
@overload
def zfill(a: T_co, width: i_co) -> _StringDTypeOrUnicodeArray: ...

@overload
def upper(a: U_co) -> NDArray[np.str_]: ...
@overload
def upper(a: S_co) -> NDArray[np.bytes_]: ...
@overload
def upper(a: _StringDTypeSupportsArray) -> _StringDTypeArray: ...
@overload
def upper(a: T_co) -> _StringDTypeOrUnicodeArray: ...

@overload
def lower(a: U_co) -> NDArray[np.str_]: ...
@overload
def lower(a: S_co) -> NDArray[np.bytes_]: ...
@overload
def lower(a: _StringDTypeSupportsArray) -> _StringDTypeArray: ...
@overload
def lower(a: T_co) -> _StringDTypeOrUnicodeArray: ...

@overload
def swapcase(a: U_co) -> NDArray[np.str_]: ...
@overload
def swapcase(a: S_co) -> NDArray[np.bytes_]: ...
@overload
def swapcase(a: _StringDTypeSupportsArray) -> _StringDTypeArray: ...
@overload
def swapcase(a: T_co) -> _StringDTypeOrUnicodeArray: ...

@overload
def capitalize(a: U_co) -> NDArray[np.str_]: ...
@overload
def capitalize(a: S_co) -> NDArray[np.bytes_]: ...
@overload
def capitalize(a: _StringDTypeSupportsArray) -> _StringDTypeArray: ...
@overload
def capitalize(a: T_co) -> _StringDTypeOrUnicodeArray: ...

@overload
def title(a: U_co) -> NDArray[np.str_]: ...
@overload
def title(a: S_co) -> NDArray[np.bytes_]: ...
@overload
def title(a: _StringDTypeSupportsArray) -> _StringDTypeArray: ...
@overload
def title(a: T_co) -> _StringDTypeOrUnicodeArray: ...

#
@overload  # Nd str
def replace[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.str_]],
    old: _CharLike_co,
    new: _CharLike_co,
    count: _IntLike_co = -1,
) -> np.ndarray[ShapeT, np.dtype[np.str_]]: ...
@overload  # Nd bytes
def replace[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    old: _CharLike_co,
    new: _CharLike_co,
    count: _IntLike_co = -1,
) -> np.ndarray[ShapeT, np.dtype[np.bytes_]]: ...
@overload  # Nd vstr
def replace[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtypes.StringDType],
    old: str,
    new: str,
    count: _IntLike_co = -1,
) -> np.ndarray[ShapeT, np.dtypes.StringDType]: ...
@overload  # 0d str
def replace(
    a: str,
    old: _CharLike_co,
    new: _CharLike_co,
    count: _IntLike_co = -1,
) -> _Array0D[np.str_]: ...
@overload  # 0d bytes
def replace(
    a: bytes,
    old: _CharLike_co,
    new: _CharLike_co,
    count: _IntLike_co = -1,
) -> _Array0D[np.bytes_]: ...
@overload  # 1d str
def replace(
    a: list[str],
    old: _CharLike_co,
    new: _CharLike_co,
    count: _IntLike_co = -1,
) -> _Array1D[np.str_]: ...
@overload  # 1d bytes
def replace(
    a: list[bytes],
    old: _CharLike_co,
    new: _CharLike_co,
    count: _IntLike_co = -1,
) -> _Array1D[np.bytes_]: ...
@overload  # 2d str
def replace(
    a: Sequence[list[str]],
    old: _CharLike_co,
    new: _CharLike_co,
    count: _IntLike_co = -1,
) -> _Array2D[np.str_]: ...
@overload  # 2d bytes
def replace(
    a: Sequence[list[bytes]],
    old: _CharLike_co,
    new: _CharLike_co,
    count: _IntLike_co = -1,
) -> _Array2D[np.bytes_]: ...
@overload  # ?d str  (fallback)
def replace(
    a: U_co,
    old: U_co | bytes | _NestedSequence[bytes],
    new: U_co | bytes | _NestedSequence[bytes],
    count: i_co = -1,
) -> NDArray[np.str_]: ...
@overload  # ?d bytes  (fallback)
def replace(
    a: S_co,
    old: S_co | str | _NestedSequence[str],
    new: S_co | str | _NestedSequence[str],
    count: i_co = -1,
) -> NDArray[np.bytes_]: ...
@overload  # ?d vstr
def replace(
    a: _StringDTypeSupportsArray,
    old: _StringDTypeSupportsArray,
    new: _StringDTypeSupportsArray,
    count: i_co = -1,
) -> _StringDTypeArray: ...
@overload  # ?d vstr | str  (fallback)
def replace(
    a: T_co,
    old: T_co,
    new: T_co,
    count: i_co = -1,
) -> _StringDTypeOrUnicodeArray: ...

# keep in sync with `rpartition`
@overload  # Nd str
def partition[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.str_]],
    sep: _CharLike_co,
) -> _tuple3[np.ndarray[ShapeT, np.dtype[np.str_]]]: ...
@overload  # Nd bytes
def partition[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    sep: _CharLike_co,
) -> _tuple3[np.ndarray[ShapeT, np.dtype[np.bytes_]]]: ...
@overload  # Nd vstr
def partition[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtypes.StringDType],
    sep: str,
) -> _tuple3[np.ndarray[ShapeT, np.dtypes.StringDType]]: ...
@overload  # 0d str
def partition(a: str, sep: _CharLike_co) -> _tuple3[_Array0D[np.str_]]: ...
@overload  # 0d bytes
def partition(a: bytes, sep: _CharLike_co) -> _tuple3[_Array0D[np.bytes_]]: ...
@overload  # 1d str
def partition(a: list[str], sep: _CharLike_co) -> _tuple3[_Array1D[np.str_]]: ...
@overload  # 1d bytes
def partition(a: list[bytes], sep: _CharLike_co) -> _tuple3[_Array1D[np.bytes_]]: ...
@overload  # 2d str
def partition(a: Sequence[list[str]], sep: _CharLike_co) -> _tuple3[_Array2D[np.str_]]: ...
@overload  # 2d bytes
def partition(a: Sequence[list[bytes]], sep: _CharLike_co) -> _tuple3[_Array2D[np.bytes_]]: ...
@overload  # ?d str  (fallback)
def partition(a: U_co, sep: U_co | S_co) -> _tuple3[NDArray[np.str_]]: ...
@overload  # ?d bytes  (fallback)
def partition(a: S_co, sep: S_co | U_co) -> _tuple3[NDArray[np.bytes_]]: ...
@overload  # ?d vstr
def partition(a: _StringDTypeSupportsArray, sep: _StringDTypeSupportsArray) -> _tuple3[_StringDTypeArray]: ...
@overload  # ?d vstr | str  (fallback)
def partition(a: T_co, sep: T_co) -> _tuple3[_StringDTypeOrUnicodeArray]: ...

# keep in sync with `partition`
@overload  # Nd str
def rpartition[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.str_]],
    sep: _CharLike_co,
) -> _tuple3[np.ndarray[ShapeT, np.dtype[np.str_]]]: ...
@overload  # Nd bytes
def rpartition[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bytes_]],
    sep: _CharLike_co,
) -> _tuple3[np.ndarray[ShapeT, np.dtype[np.bytes_]]]: ...
@overload  # Nd vstr
def rpartition[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtypes.StringDType],
    sep: str,
) -> _tuple3[np.ndarray[ShapeT, np.dtypes.StringDType]]: ...
@overload  # 0d str
def rpartition(a: str, sep: _CharLike_co) -> _tuple3[_Array0D[np.str_]]: ...
@overload  # 0d bytes
def rpartition(a: bytes, sep: _CharLike_co) -> _tuple3[_Array0D[np.bytes_]]: ...
@overload  # 1d str
def rpartition(a: list[str], sep: _CharLike_co) -> _tuple3[_Array1D[np.str_]]: ...
@overload  # 1d bytes
def rpartition(a: list[bytes], sep: _CharLike_co) -> _tuple3[_Array1D[np.bytes_]]: ...
@overload  # 2d str
def rpartition(a: Sequence[list[str]], sep: _CharLike_co) -> _tuple3[_Array2D[np.str_]]: ...
@overload  # 2d bytes
def rpartition(a: Sequence[list[bytes]], sep: _CharLike_co) -> _tuple3[_Array2D[np.bytes_]]: ...
@overload  # ?d str  (fallback)
def rpartition(a: U_co, sep: U_co | S_co) -> _tuple3[NDArray[np.str_]]: ...
@overload  # ?d bytes  (fallback)
def rpartition(a: S_co, sep: S_co | U_co) -> _tuple3[NDArray[np.bytes_]]: ...
@overload  # ?d vstr
def rpartition(a: _StringDTypeSupportsArray, sep: _StringDTypeSupportsArray) -> _tuple3[_StringDTypeArray]: ...
@overload  # ?d vstr | str  (fallback)
def rpartition(a: T_co, sep: T_co) -> _tuple3[_StringDTypeOrUnicodeArray]: ...

#
@overload
def translate(
    a: U_co,
    table: str,
    deletechars: str | None = None,
) -> NDArray[np.str_]: ...
@overload
def translate(
    a: S_co,
    table: str,
    deletechars: str | None = None,
) -> NDArray[np.bytes_]: ...
@overload
def translate(
    a: _StringDTypeSupportsArray,
    table: str,
    deletechars: str | None = None,
) -> _StringDTypeArray: ...
@overload
def translate(
    a: T_co,
    table: str,
    deletechars: str | None = None,
) -> _StringDTypeOrUnicodeArray: ...

#
@overload
def slice(
    a: U_co,
    start: i_co | None = None,
    stop: i_co | _NoValueType | None = ...,  # = np._NoValue
    step: i_co | None = None,
    /,
) -> NDArray[np.str_]: ...
@overload
def slice(
    a: S_co,
    start: i_co | None = None,
    stop: i_co | _NoValueType | None = ...,  # = np._NoValue
    step: i_co | None = None,
    /,
) -> NDArray[np.bytes_]: ...
@overload
def slice(
    a: _StringDTypeSupportsArray,
    start: i_co | None = None,
    stop: i_co | _NoValueType | None = ...,  # = np._NoValue
    step: i_co | None = None,
    /,
) -> _StringDTypeArray: ...
@overload
def slice(
    a: T_co,
    start: i_co | None = None,
    stop: i_co | _NoValueType | None = ...,  # = np._NoValue
    step: i_co | None = None,
    /,
) -> _StringDTypeOrUnicodeArray: ...
