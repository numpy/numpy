from _typeshed import OpenBinaryMode, OpenTextMode, StrPath
from typing import IO, Any

type _Mode = OpenBinaryMode | OpenTextMode

###

# exported in numpy.lib.nppyio
class DataSource:
    def __init__(self, /, destpath: StrPath | None = ".") -> None: ...
    def __del__(self, /) -> None: ...
    def abspath(self, /, path: StrPath) -> str: ...
    def exists(self, /, path: StrPath) -> bool: ...

    # Whether the file-object is opened in string or bytes mode (by default)
    # depends on the file-extension of `path`
    def open(self, /, path: StrPath, mode: _Mode = "r", encoding: str | None = None, newline: str | None = None) -> IO[Any]: ...

class Repository(DataSource):
    def __init__(self, /, baseurl: StrPath, destpath: StrPath | None = ".") -> None: ...
    def listdir(self, /) -> list[str]: ...

def open(
    path: StrPath,
    mode: _Mode = "r",
    destpath: StrPath | None = ".",
    encoding: str | None = None,
    newline: str | None = None,
) -> IO[Any]: ...
