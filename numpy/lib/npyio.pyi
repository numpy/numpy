import os
import sys
import zipfile
import types
from typing import (
    Literal as L,
    Any,
    Mapping,
    TypeVar,
    Generic,
    List,
    Type,
    Iterator,
    Union,
    IO,
    overload,
    Sequence,
    Callable,
    Pattern,
    Protocol,
)

from numpy import (
    DataSource as DataSource,
    ndarray,
    recarray,
    dtype,
    generic,
    float64,
    void,
)

from numpy.ma.mrecords import MaskedRecords
from numpy.typing import ArrayLike, DTypeLike, NDArray, _SupportsDType

from numpy.core.multiarray import (
    packbits as packbits,
    unpackbits as unpackbits,
)

_T = TypeVar("_T")
_T_contra = TypeVar("_T_contra", contravariant=True)
_T_co = TypeVar("_T_co", covariant=True)
_SCT = TypeVar("_SCT", bound=generic)

_DTypeLike = Union[
    Type[_SCT],
    dtype[_SCT],
    _SupportsDType[dtype[_SCT]],
]

class _SupportsGetItem(Protocol[_T_contra, _T_co]):
    def __getitem__(self, key: _T_contra) -> _T_co: ...

__all__: List[str]

class BagObj(Generic[_T_co]):
    def __init__(self, obj: _SupportsGetItem[str, _T_co]) -> None: ...
    def __getattribute__(self, key: str) -> _T_co: ...
    def __dir__(self) -> List[str]: ...

class NpzFile(Mapping[str, NDArray[Any]]):
    zip: zipfile.ZipFile
    fid: None | IO[str]
    files: List[str]
    allow_pickle: bool
    pickle_kwargs: None | Mapping[str, Any]
    # Represent `f` as a mutable property so we can access the type of `self`
    @property
    def f(self: _T) -> BagObj[_T]: ...
    @f.setter
    def f(self: _T, value: BagObj[_T]) -> None: ...
    def __init__(
        self,
        fid: IO[str],
        own_fid: bool = ...,
        allow_pickle: bool = ...,
        pickle_kwargs: None | Mapping[str, Any] = ...,
    ) -> None: ...
    def __enter__(self: _T) -> _T: ...
    def __exit__(
        self,
        exc_type: None | Type[BaseException],
        exc_value: None | BaseException,
        traceback: None | types.TracebackType,
        /,
    ) -> None: ...
    def close(self) -> None: ...
    def __del__(self) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: str) -> NDArray[Any]: ...

# NOTE: Returns a `NpzFile` if file is a zip file;
# returns an `ndarray`/`memmap` otherwise
def load(
    file: str | bytes | os.PathLike[Any] | IO[bytes],
    mmap_mode: L[None, "r+", "r", "w+", "c"] = ...,
    allow_pickle: bool = ...,
    fix_imports: bool = ...,
    encoding: L["ASCII", "latin1", "bytes"] = ...,
) -> Any: ...

def save(
    file: str | os.PathLike[str] | IO[bytes],
    arr: ArrayLike,
    allow_pickle: bool = ...,
    fix_imports: bool = ...,
) -> None: ...

def savez(
    file: str | os.PathLike[str] | IO[bytes],
    *args: ArrayLike,
    **kwds: ArrayLike,
) -> None: ...

def savez_compressed(
    file: str | os.PathLike[str] | IO[bytes],
    *args: ArrayLike,
    **kwds: ArrayLike,
) -> None: ...

@overload
def loadtxt(
    fname: str | os.PathLike[str] | IO[Any],
    dtype: None = ...,
    comments: str | Sequence[str] = ...,
    delimiter: None | str = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    skiprows: int = ...,
    usecols: int | Sequence[int] = ...,
    unpack: bool = ...,
    ndmin: L[0, 1, 2] = ...,
    encoding: None | str = ...,
    max_rows: None | int = ...,
    *,
    like: None | ArrayLike = ...
) -> NDArray[float64]: ...
@overload
def loadtxt(
    fname: str | os.PathLike[str] | IO[Any],
    dtype: _DTypeLike[_SCT],
    comments: str | Sequence[str] = ...,
    delimiter: None | str = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    skiprows: int = ...,
    usecols: int | Sequence[int] = ...,
    unpack: bool = ...,
    ndmin: L[0, 1, 2] = ...,
    encoding: None | str = ...,
    max_rows: None | int = ...,
    *,
    like: None | ArrayLike = ...
) -> NDArray[_SCT]: ...
@overload
def loadtxt(
    fname: str | os.PathLike[str] | IO[Any],
    dtype: DTypeLike,
    comments: str | Sequence[str] = ...,
    delimiter: None | str = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    skiprows: int = ...,
    usecols: int | Sequence[int] = ...,
    unpack: bool = ...,
    ndmin: L[0, 1, 2] = ...,
    encoding: None | str = ...,
    max_rows: None | int = ...,
    *,
    like: None | ArrayLike = ...
) -> NDArray[Any]: ...

def savetxt(
    fname: str | os.PathLike[str] | IO[Any],
    X: ArrayLike,
    fmt: str | Sequence[str] = ...,
    delimiter: str = ...,
    newline: str = ...,
    header: str = ...,
    footer: str = ...,
    comments: str = ...,
    encoding: None | str = ...,
) -> None: ...

@overload
def fromregex(
    file: str | os.PathLike[str] | IO[Any],
    regexp: str | bytes | Pattern[Any],
    dtype: _DTypeLike[_SCT],
    encoding: None | str = ...
) -> NDArray[_SCT]: ...
@overload
def fromregex(
    file: str | os.PathLike[str] | IO[Any],
    regexp: str | bytes | Pattern[Any],
    dtype: DTypeLike,
    encoding: None | str = ...
) -> NDArray[Any]: ...

# TODO: Sort out arguments
@overload
def genfromtxt(
    fname: str | os.PathLike[str] | IO[Any],
    dtype: None = ...,
    *args: Any,
    **kwargs: Any,
) -> NDArray[float64]: ...
@overload
def genfromtxt(
    fname: str | os.PathLike[str] | IO[Any],
    dtype: _DTypeLike[_SCT],
    *args: Any,
    **kwargs: Any,
) -> NDArray[_SCT]: ...
@overload
def genfromtxt(
    fname: str | os.PathLike[str] | IO[Any],
    dtype: DTypeLike,
    *args: Any,
    **kwargs: Any,
) -> NDArray[Any]: ...

@overload
def recfromtxt(
    fname: str | os.PathLike[str] | IO[Any],
    *,
    usemask: L[False] = ...,
    **kwargs: Any,
) -> recarray[Any, dtype[void]]: ...
@overload
def recfromtxt(
    fname: str | os.PathLike[str] | IO[Any],
    *,
    usemask: L[True],
    **kwargs: Any,
) -> MaskedRecords[Any, dtype[void]]: ...

@overload
def recfromcsv(
    fname: str | os.PathLike[str] | IO[Any],
    *,
    usemask: L[False] = ...,
    **kwargs: Any,
) -> recarray[Any, dtype[void]]: ...
@overload
def recfromcsv(
    fname: str | os.PathLike[str] | IO[Any],
    *,
    usemask: L[True],
    **kwargs: Any,
) -> MaskedRecords[Any, dtype[void]]: ...
