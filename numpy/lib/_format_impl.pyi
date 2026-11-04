import os
from _typeshed import SupportsRead, SupportsWrite
from collections.abc import Sequence
from typing import Any, BinaryIO, Final, TypeGuard, overload

import numpy as np
import numpy.typing as npt
from numpy.dtypes import StringDType
from numpy.lib._utils_impl import drop_metadata as drop_metadata

__all__: list[str] = []

###

EXPECTED_KEYS: Final[set[str]] = ...
MAGIC_PREFIX: Final = b"\x93NUMPY"
MAGIC_LEN: Final = 8
ARRAY_ALIGN: Final = 64
BUFFER_SIZE: Final = 262_144  # 1 << 18
GROWTH_AXIS_MAX_DIGITS: Final = 21
_MAX_HEADER_SIZE: Final = 10_000

def magic(major: int, minor: int) -> bytes: ...
def read_magic(fp: SupportsRead[bytes]) -> tuple[int, int]: ...

#
@overload  # void  (workaround)
def dtype_to_descr(dtype: np.dtype[np.void]) -> str | list[tuple[Any, ...]]: ...
@overload  # non-void
def dtype_to_descr(
    dtype: np.dtype[np.bool | np.number | np.character | np.datetime64 | np.timedelta64 | np.object_] | StringDType,
) -> str: ...
@overload  # fallback
def dtype_to_descr(dtype: np.dtype) -> str | list[tuple[Any, ...]]: ...

#
@overload  # dtype.str
def descr_to_dtype(descr: str) -> np.dtype[Any]: ...
@overload  # dtype.descr
def descr_to_dtype(descr: Sequence[tuple[Any, ...]]) -> np.dtype[np.void]: ...

#
def header_data_from_array_1_0(array: np.ndarray) -> dict[str, Any]: ...
def write_array_header_1_0(fp: SupportsWrite[bytes], d: dict[str, Any]) -> None: ...
def write_array_header_2_0(fp: SupportsWrite[bytes], d: dict[str, Any]) -> None: ...
def read_array_header_1_0(fp: SupportsRead[bytes], max_header_size: int = 10_000) -> tuple[tuple[int, ...], bool, np.dtype]: ...
def read_array_header_2_0(fp: SupportsRead[bytes], max_header_size: int = 10_000) -> tuple[tuple[int, ...], bool, np.dtype]: ...
def write_array(
    fp: SupportsWrite[bytes],
    array: np.ndarray,
    version: tuple[int, int] | None = None,
    allow_pickle: bool = True,
    pickle_kwargs: dict[str, Any] | None = None,
) -> None: ...
def read_array(
    fp: SupportsRead[bytes],
    allow_pickle: bool = False,
    pickle_kwargs: dict[str, Any] | None = None,
    *,
    max_header_size: int = 10_000,
) -> np.ndarray: ...
def open_memmap(
    filename: str | os.PathLike[Any],
    mode: str = "r+",
    dtype: npt.DTypeLike | None = None,
    shape: tuple[int, ...] | None = None,
    fortran_order: bool = False,
    version: tuple[int, int] | None = None,
    *,
    max_header_size: int = 10_000,
) -> np.memmap: ...
def isfileobj(f: object) -> TypeGuard[BinaryIO]: ...  # don't use `typing.TypeIs`
