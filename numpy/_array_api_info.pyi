from typing import TypedDict, Optional, Union, Tuple, List
from numpy._typing import DtypeLike

Capabilities = TypedDict(
    "Capabilities",
    {
        "boolean indexing": bool,
        "data-dependent shapes": bool,
    },
)

DefaultDataTypes = TypedDict(
    "DefaultDataTypes",
    {
        "real floating": DtypeLike,
        "complex floating": DtypeLike,
        "integral": DtypeLike,
        "indexing": DtypeLike,
    },
)

DataTypes = TypedDict(
    "DataTypes",
    {
        "bool": DtypeLike,
        "float32": DtypeLike,
        "float64": DtypeLike,
        "complex64": DtypeLike,
        "complex128": DtypeLike,
        "int8": DtypeLike,
        "int16": DtypeLike,
        "int32": DtypeLike,
        "int64": DtypeLike,
        "uint8": DtypeLike,
        "uint16": DtypeLike,
        "uint32": DtypeLike,
        "uint64": DtypeLike,
    },
    total=False,
)

class __array_namespace_info__:
    __module__: str

    def capabilities(self) -> Capabilities: ...

    def default_device(self) -> str: ...

    def default_dtypes(
        self,
        *,
        device: Optional[str] = None,
    ) -> DefaultDataTypes: ...

    def dtypes(
        self,
        *,
        device: Optional[str] = None,
        kind: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> DataTypes: ...

    def devices(self) -> List[str]: ...
