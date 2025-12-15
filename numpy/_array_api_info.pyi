from typing import Literal, Never, TypedDict, final, overload, type_check_only

import numpy as np

type _Device = Literal["cpu"]
type _DeviceLike = _Device | None

_Capabilities = TypedDict(
    "_Capabilities",
    {
        "boolean indexing": Literal[True],
        "data-dependent shapes": Literal[True],
    },
)

_DefaultDTypes = TypedDict(
    "_DefaultDTypes",
    {
        "real floating": np.dtype[np.float64],
        "complex floating": np.dtype[np.complex128],
        "integral": np.dtype[np.intp],
        "indexing": np.dtype[np.intp],
    },
)

type _KindBool = Literal["bool"]
type _KindInt = Literal["signed integer"]
type _KindUInt = Literal["unsigned integer"]
type _KindInteger = Literal["integral"]
type _KindFloat = Literal["real floating"]
type _KindComplex = Literal["complex floating"]
type _KindNumber = Literal["numeric"]
type _Kind = _KindBool | _KindInt | _KindUInt | _KindInteger | _KindFloat | _KindComplex | _KindNumber

type _Permute1[T1] = T1 | tuple[T1]
type _Permute2[T1, T2] = tuple[T1, T2] | tuple[T2, T1]
type _Permute3[T1, T2, T3] = (
    tuple[T1, T2, T3] | tuple[T1, T3, T2]
    | tuple[T2, T1, T3] | tuple[T2, T3, T1]
    | tuple[T3, T1, T2] | tuple[T3, T2, T1]
)  # fmt: skip

@type_check_only
class _DTypesBool(TypedDict):
    bool: np.dtype[np.bool]

@type_check_only
class _DTypesInt(TypedDict):
    int8: np.dtype[np.int8]
    int16: np.dtype[np.int16]
    int32: np.dtype[np.int32]
    int64: np.dtype[np.int64]

@type_check_only
class _DTypesUInt(TypedDict):
    uint8: np.dtype[np.uint8]
    uint16: np.dtype[np.uint16]
    uint32: np.dtype[np.uint32]
    uint64: np.dtype[np.uint64]

@type_check_only
class _DTypesInteger(_DTypesInt, _DTypesUInt): ...

@type_check_only
class _DTypesFloat(TypedDict):
    float32: np.dtype[np.float32]
    float64: np.dtype[np.float64]

@type_check_only
class _DTypesComplex(TypedDict):
    complex64: np.dtype[np.complex64]
    complex128: np.dtype[np.complex128]

@type_check_only
class _DTypesNumber(_DTypesInteger, _DTypesFloat, _DTypesComplex): ...

@type_check_only
class _DTypes(_DTypesBool, _DTypesNumber): ...

@type_check_only
class _DTypesUnion(TypedDict, total=False):
    bool: np.dtype[np.bool]
    int8: np.dtype[np.int8]
    int16: np.dtype[np.int16]
    int32: np.dtype[np.int32]
    int64: np.dtype[np.int64]
    uint8: np.dtype[np.uint8]
    uint16: np.dtype[np.uint16]
    uint32: np.dtype[np.uint32]
    uint64: np.dtype[np.uint64]
    float32: np.dtype[np.float32]
    float64: np.dtype[np.float64]
    complex64: np.dtype[np.complex64]
    complex128: np.dtype[np.complex128]

type _EmptyDict = dict[Never, Never]

@final
class __array_namespace_info__:
    __module__: Literal["numpy"] = "numpy"

    def capabilities(self) -> _Capabilities: ...
    def default_device(self) -> _Device: ...
    def default_dtypes(self, *, device: _DeviceLike = None) -> _DefaultDTypes: ...
    def devices(self) -> list[_Device]: ...

    @overload
    def dtypes(
        self,
        *,
        device: _DeviceLike = None,
        kind: None = None,
    ) -> _DTypes: ...
    @overload
    def dtypes(
        self,
        *,
        device: _DeviceLike = None,
        kind: _Permute1[_KindBool],
    ) -> _DTypesBool: ...
    @overload
    def dtypes(
        self,
        *,
        device: _DeviceLike = None,
        kind: _Permute1[_KindInt],
    ) -> _DTypesInt: ...
    @overload
    def dtypes(
        self,
        *,
        device: _DeviceLike = None,
        kind: _Permute1[_KindUInt],
    ) -> _DTypesUInt: ...
    @overload
    def dtypes(
        self,
        *,
        device: _DeviceLike = None,
        kind: _Permute1[_KindFloat],
    ) -> _DTypesFloat: ...
    @overload
    def dtypes(
        self,
        *,
        device: _DeviceLike = None,
        kind: _Permute1[_KindComplex],
    ) -> _DTypesComplex: ...
    @overload
    def dtypes(
        self,
        *,
        device: _DeviceLike = None,
        kind: _Permute1[_KindInteger] | _Permute2[_KindInt, _KindUInt],
    ) -> _DTypesInteger: ...
    @overload
    def dtypes(
        self,
        *,
        device: _DeviceLike = None,
        kind: _Permute1[_KindNumber] | _Permute3[_KindInteger, _KindFloat, _KindComplex],
    ) -> _DTypesNumber: ...
    @overload
    def dtypes(
        self,
        *,
        device: _DeviceLike = None,
        kind: tuple[()],
    ) -> _EmptyDict: ...
    @overload
    def dtypes(
        self,
        *,
        device: _DeviceLike = None,
        kind: tuple[_Kind, ...],
    ) -> _DTypesUnion: ...
