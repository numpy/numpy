from collections.abc import Iterable
from typing import (
    Literal as L,
    overload,
    TypeAlias,
    TypeVar,
    Any,
    SupportsIndex,
    SupportsInt,
    NamedTuple,
)

import numpy as np
import numpy.typing as npt
from numpy._typing import (
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeInt_co,
    _ArrayLikeUInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeNumber_co,
    _ArrayLikeTD64_co,
    _ArrayLikeObject_co,
    _SupportsArray,
    _DTypeLike,
    _FloatLike_co,
    _IntLike_co,
    _ShapeLike,
)
from numpy.linalg import LinAlgError

_T = TypeVar("_T")
_ArrayType = TypeVar("_ArrayType", bound=np.ndarray[Any, Any])
_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_number = TypeVar("_SCT_number", bound=np.number)

_2Tuple: TypeAlias = tuple[_T, _T]
_ModeKind: TypeAlias = L["reduced", "complete", "r", "raw"]
_NormOrder: TypeAlias = L["fro", "nuc", 0, 1, -1, 2, -2] | float | None
_UPLO: TypeAlias = L["L", "U", "l", "u"]

__all__ = [
    'matrix_power',
    'solve',
    'tensorsolve',
    'tensorinv',
    'inv',
    'cholesky',
    'eigvals',
    'eigvalsh',
    'pinv',
    'slogdet',
    'det',
    'svd',
    'svdvals',
    'eig',
    'eigh',
    'lstsq',
    'norm',
    'qr',
    'cond',
    'matrix_rank',
    'LinAlgError',
    'multi_dot',
    'trace',
    'diagonal',
    'cross',
    'outer',
    'tensordot',
    'matmul',
    'matrix_transpose',
    'matrix_norm',
    'vector_norm',
    'vecdot',
]

class EigResult(NamedTuple):
    eigenvalues: npt.NDArray[Any]
    eigenvectors: npt.NDArray[Any]

class EighResult(NamedTuple):
    eigenvalues: npt.NDArray[Any]
    eigenvectors: npt.NDArray[Any]

class QRResult(NamedTuple):
    Q: npt.NDArray[Any]
    R: npt.NDArray[Any]

class SlogdetResult(NamedTuple):
    sign: npt.NDArray[Any]
    logabsdet: npt.NDArray[Any]

class SVDResult(NamedTuple):
    U: npt.NDArray[Any]
    S: npt.NDArray[Any]
    Vh: npt.NDArray[Any]

@overload
def tensorsolve(  # type: ignore[overload-overlap]
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axes: None | Iterable[SupportsIndex] =...,
) -> npt.NDArray[np.float64]: ...
@overload
def tensorsolve(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axes: None | Iterable[SupportsIndex] =...,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def tensorsolve(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axes: None | Iterable[SupportsIndex] =...,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...

@overload
def solve(  # type: ignore[overload-overlap]
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
) -> npt.NDArray[np.float64]: ...
@overload
def solve(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def solve(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...

@overload
def tensorinv(  # type: ignore[overload-overlap]
    a: _ArrayLikeInt_co,
    ind: _IntLike_co = ...,
) -> npt.NDArray[np.float64]: ...
@overload
def tensorinv(
    a: _ArrayLikeFloat_co,
    ind: _IntLike_co = ...,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def tensorinv(
    a: _ArrayLikeComplex_co,
    ind: _IntLike_co = ...,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...

@overload
def inv(a: _ArrayLikeInt_co) -> npt.NDArray[np.float64]: ...  # type: ignore[overload-overlap]
@overload
def inv(a: _ArrayLikeFloat_co) -> npt.NDArray[np.floating[Any]]: ...
@overload
def inv(a: _ArrayLikeComplex_co) -> npt.NDArray[np.complexfloating[Any, Any]]: ...

# TODO: The supported input and output dtypes are dependent on the value of `n`.
# For example: `n < 0` always casts integer types to float64
def matrix_power(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    n: SupportsIndex,
) -> npt.NDArray[Any]: ...

@overload
def cholesky(  # type: ignore[overload-overlap]
    a: _ArrayLikeInt_co,
    /, *,
    upper: bool | np.bool = ...,
) -> npt.NDArray[np.float64]: ...
@overload
def cholesky(
    a: _ArrayLikeFloat_co,
    /, *,
    upper: bool | np.bool = ...,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def cholesky(
    a: _ArrayLikeComplex_co,
    /, *,
    upper: bool | np.bool = ...,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...

@overload
def outer(
    x1: _SupportsArray[np.dtype[_SCT_number]],
    x2: _SupportsArray[np.dtype[_SCT_number]],
    /,
) -> npt.NDArray[_SCT_number]: ...
@overload
def outer(  # type: ignore[overload-overlap]
    x1: _ArrayLikeBool_co,
    x2: _ArrayLikeBool_co,
    /,
) -> npt.NDArray[np.bool]: ...
@overload
def outer(
    x1: _ArrayLikeUInt_co,
    x2: _ArrayLikeUInt_co,
    /,
) -> npt.NDArray[np.unsignedinteger[Any]]: ...
@overload
def outer(
    x1: _ArrayLikeInt_co,
    x2: _ArrayLikeInt_co,
    /,
) -> npt.NDArray[np.signedinteger[Any]]: ...
@overload
def outer(
    x1: _ArrayLikeFloat_co,
    x2: _ArrayLikeFloat_co,
    /,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def outer(
    x1: _ArrayLikeComplex_co,
    x2: _ArrayLikeComplex_co,
    /,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
@overload
def outer(
    x1: _ArrayLikeTD64_co,
    x2: _ArrayLikeTD64_co,
    /,
) -> npt.NDArray[np.timedelta64]: ...
@overload
def outer(
    x1: _ArrayLikeObject_co | _ArrayLikeComplex_co | _ArrayLikeTD64_co,
    x2: _ArrayLikeObject_co | _ArrayLikeComplex_co | _ArrayLikeTD64_co,
    /,
) -> npt.NDArray[np.object_]: ...

def qr(a: _ArrayLikeComplex_co, mode: _ModeKind = ...) -> QRResult: ...

@overload
def eigvals(  # type: ignore[overload-overlap]
    a: _ArrayLikeInt_co,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]: ...
@overload
def eigvals(
    a: _ArrayLikeFloat_co,
) -> npt.NDArray[np.floating[Any]] | npt.NDArray[np.complexfloating[Any, Any]]: ...
@overload
def eigvals(
    a: _ArrayLikeComplex_co,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...

@overload
def eigvalsh(
    a: _ArrayLikeInt_co,
    UPLO: _UPLO = ...,
) -> npt.NDArray[np.float64]: ...
@overload
def eigvalsh(
    a: _ArrayLikeComplex_co,
    UPLO: _UPLO = ...,
) -> npt.NDArray[np.floating[Any]]: ...

def eig(a: _ArrayLikeComplex_co) -> EigResult: ...

def eigh(a: _ArrayLikeComplex_co, UPLO: _UPLO = ...) -> EighResult: ...

@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool = ...,
    compute_uv: L[True] = ...,
    hermitian: bool = ...,
) -> SVDResult: ...
@overload
def svd(
    a: _ArrayLikeInt_co,
    full_matrices: bool = ...,
    compute_uv: L[False] = ...,
    hermitian: bool = ...,
) -> npt.NDArray[np.float64]: ...
@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool = ...,
    compute_uv: L[False] = ...,
    hermitian: bool = ...,
) -> npt.NDArray[np.floating[Any]]: ...

@overload
def svdvals(x: _ArrayLikeInt_co, /) -> npt.NDArray[np.float64]: ...
@overload
def svdvals(x: _ArrayLikeComplex_co, /) -> npt.NDArray[np.floating[Any]]: ...

# TODO: Returns a scalar for 2D arrays and
# a `(x.ndim - 2)`` dimensionl array otherwise
@overload
def cond(  # type: ignore[overload-overlap]
    x: _ArrayLikeInt_co,
    p: _NormOrder = ...,
) -> np.float64 | npt.NDArray[np.float64]: ...
@overload
def cond(  # type: ignore[overload-overlap]
    x: _ArrayLikeFloat_co,
    p: _NormOrder = ...,
) -> np.floating[Any] | npt.NDArray[np.floating[Any]]: ...
@overload
def cond(
    x: _ArrayLikeComplex_co,
    p: _NormOrder = ...,
) -> np.complexfloating[Any, Any] | npt.NDArray[np.complexfloating[Any, Any]]: ...

def matrix_rank(
    A: _ArrayLikeComplex_co,
    tol: None | _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
    *,
    rtol: None | _ArrayLikeFloat_co = ...,
) -> np.int_ | npt.NDArray[np.int_] : ...

@overload
def pinv(  # type: ignore[overload-overlap]
    a: _ArrayLikeInt_co,
    rcond: _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
    *,
    rtol: _ArrayLikeFloat_co = ...,
) -> npt.NDArray[np.float64]: ...
@overload
def pinv(
    a: _SupportsArray[np.dtype[_SCT_number]],
    rcond: _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
    *,
    rtol: _ArrayLikeFloat_co = ...,
) -> npt.NDArray[_SCT_number]: ...
@overload
def pinv(
    a: _ArrayLikeFloat_co,
    rcond: _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
    *,
    rtol: _ArrayLikeFloat_co = ...,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def pinv(
    a: _ArrayLikeComplex_co,
    rcond: _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
    *,
    rtol: _ArrayLikeFloat_co = ...,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...

# TODO: Returns a 2-tuple of scalars for 2D arrays and
# a 2-tuple of `(a.ndim - 2)`` dimensionl arrays otherwise
def slogdet(a: _ArrayLikeComplex_co) -> SlogdetResult: ...

@overload
def det(a: _ArrayLikeInt_co) -> np.float64 | npt.NDArray[np.float64]: ...  # type: ignore[overload-overlap]
@overload
def det(a: _SupportsArray[np.dtype[_SCT_number]]) -> _SCT_number | npt.NDArray[_SCT_number]: ...
@overload
def det(a: _ArrayLikeFloat_co) -> np.floating[Any] | npt.NDArray[np.floating[Any]]: ...
@overload
def det(a: _ArrayLikeComplex_co) -> (
    np.complexfloating[Any, Any]
    | npt.NDArray[np.complexfloating[Any, Any]]
): ...

@overload
def lstsq(  # type: ignore[overload-overlap]
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    rcond: None | _FloatLike_co = ...,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    np.int32,
    npt.NDArray[np.float64],
]: ...
@overload
def lstsq(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    rcond: None | _FloatLike_co = ...,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    np.int32,
    npt.NDArray[np.floating[Any]],
]: ...
@overload
def lstsq(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    rcond: None | _FloatLike_co = ...,
) -> tuple[
    npt.NDArray[np.complexfloating[Any, Any]],
    npt.NDArray[np.floating[Any]],
    np.int32,
    npt.NDArray[np.floating[Any]],
]: ...

@overload
def norm(
    x: npt.ArrayLike,
    /, *,
    ord: _NormOrder = ...,
    axis: None = ...,
    keepdims: L[False] = ...,
) -> np.floating[Any]: ...
@overload
def norm(
    x: npt.ArrayLike,
    /, *,
    ord: _NormOrder = ...,
    axis: None = ...,
    keepdims: L[True],
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def norm(
    x: npt.ArrayLike,
    /, *,
    ord: _NormOrder = ...,
    axis: SupportsInt | SupportsIndex | tuple[int, ...],
    keepdims: L[False] = ...,
) -> Any: ...
@overload
def norm(
    x: npt.ArrayLike,
    /, *,
    ord: _NormOrder = ...,
    axis: SupportsInt | SupportsIndex | tuple[int, ...],
    keepdims: L[True],
) -> npt.NDArray[Any]: ...

@overload
def tensordot(  # type: ignore[overload-overlap]
    x1: _ArrayLikeBool_co,
    x2: _ArrayLikeBool_co,
    /, *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> npt.NDArray[np.bool]: ...
@overload
def tensordot(
    x1: _ArrayLikeUInt_co,
    x2: _ArrayLikeUInt_co,
    /, *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> npt.NDArray[np.unsignedinteger[Any]]: ...
@overload
def tensordot(
    x1: _ArrayLikeInt_co,
    x2: _ArrayLikeInt_co,
    /, *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> npt.NDArray[np.signedinteger[Any]]: ...
@overload
def tensordot(
    x1: _ArrayLikeFloat_co,
    x2: _ArrayLikeFloat_co,
    /, *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def tensordot(
    x1: _ArrayLikeComplex_co,
    x2: _ArrayLikeComplex_co,
    /, *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
@overload
def tensordot(
    x1: _ArrayLikeTD64_co,
    x2: _ArrayLikeTD64_co,
    /, *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> npt.NDArray[np.timedelta64]: ...
@overload
def tensordot(
    x1: _ArrayLikeObject_co,
    x2: _ArrayLikeObject_co,
    /, *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> npt.NDArray[np.object_]: ...

@overload
def matrix_transpose(x: _ArrayLike[_SCT], /) -> npt.NDArray[_SCT]: ...
@overload
def matrix_transpose(x: npt.ArrayLike, /) -> npt.NDArray[Any]: ...

@overload
def matrix_norm(
    x: npt.ArrayLike,
    /, *,
    ord: _NormOrder = ...,
    keepdims: L[False] = ...,
) -> np.floating[Any]: ...
@overload
def matrix_norm(
    x: npt.ArrayLike,
    /, *,
    ord: _NormOrder = ...,
    keepdims: L[True],
) -> npt.NDArray[Any]: ...

@overload
def vector_norm(
    x: npt.ArrayLike,
    /, *,
    axis: None = ...,
    ord: _NormOrder = ...,
    keepdims: L[False] = ...,
) -> np.floating[Any]: ...
@overload
def vector_norm(
    x: npt.ArrayLike,
    /, *,
    axis: None = ...,
    ord: _NormOrder = ...,
    keepdims: L[True],
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def vector_norm(
    x: npt.ArrayLike,
    /, *,
    axis: SupportsInt | SupportsIndex | tuple[int, ...] = ...,
    ord: _NormOrder = ...,
    keepdims: L[False] = ...,
) -> Any: ...
@overload
def vector_norm(
    x: npt.ArrayLike,
    /, *,
    axis: SupportsInt | SupportsIndex | tuple[int, ...] = ...,
    ord: _NormOrder = ...,
    keepdims: L[True],
) -> npt.NDArray[Any]: ...

@overload
def multi_dot(
    arrays: Iterable[_ArrayLikeComplex_co | _ArrayLikeObject_co | _ArrayLikeTD64_co],
    *,
    out: _ArrayType,
) -> _ArrayType: ...
@overload
def multi_dot(
    arrays: Iterable[_ArrayLikeComplex_co | _ArrayLikeObject_co | _ArrayLikeTD64_co],
    *,
    out: None = ...,
) -> Any: ...

@overload
def diagonal(
    x: _SupportsArray[np.dtype[_SCT]],
    /, *,
    offset: SupportsIndex = ...,
) -> npt.NDArray[_SCT]: ...
@overload
def diagonal(
    x: npt.ArrayLike,
    /, *,
    offset: SupportsIndex = ...,
) -> npt.NDArray[Any]: ...

@overload
def trace(  # type: ignore[overload-overlap]
    x: _ArrayLikeBool_co,
    /, *,
    offset: SupportsIndex = ...,
    dtype: None = ...,
) -> np.int_: ...
@overload
def trace(  # type: ignore[overload-overlap]
    x: _ArrayLikeUInt_co,
    /, *,
    offset: SupportsIndex = ...,
    dtype: None = ...,
) -> np.uint: ...
@overload
def trace(
    x: _ArrayLikeInt_co,
    /, *,
    offset: SupportsIndex = ...,
    dtype: None = ...,
) -> np.int_: ...
@overload
def trace(
    x: _ArrayLikeObject_co,
    /, *,
    offset: SupportsIndex = ...,
    dtype: None = ...,
) -> object: ...
@overload
def trace(
    x: _SupportsArray[np.dtype[_SCT]],
    /, *,
    offset: SupportsIndex = ...,
    dtype: None = ...,
) -> _SCT: ...
@overload
def trace(
    x: _ArrayLikeFloat_co,
    /, *,
    offset: SupportsIndex = ...,
    dtype: None = ...,
) -> np.floating[Any]: ...
@overload
def trace(
    x: _ArrayLikeComplex_co,
    /, *,
    offset: SupportsIndex = ...,
    dtype: None = ...,
) -> np.complexfloating[Any, Any]: ...
@overload
def trace(
    x: _ArrayLikeNumber_co,
    /, *,
    offset: SupportsIndex = ...,
    dtype: None = ...,
) -> np.number[Any]: ...
@overload
def trace(
    x: npt.ArrayLike,
    /, *,
    offset: SupportsIndex = ...,
    dtype: _DTypeLike[_SCT],
) -> _SCT: ...

@overload
def cross(
    x1: _SupportsArray[np.dtype[_SCT]],
    x2: _SupportsArray[np.dtype[_SCT]],
    /, *,
    axis: SupportsIndex = ...,
) -> npt.NDArray[_SCT]: ...
@overload
def cross(  # type: ignore[overload-overlap]
    x1: _ArrayLikeBool_co,
    x2: _ArrayLikeBool_co,
    /, *,
    axis: SupportsIndex = ...,
) -> npt.NDArray[np.bool]: ...
@overload
def cross(
    x1: _ArrayLikeUInt_co,
    x2: _ArrayLikeUInt_co,
    /, *,
    axis: SupportsIndex = ...,
) -> npt.NDArray[np.unsignedinteger[Any]]: ...
@overload
def cross(
    x1: _ArrayLikeInt_co,
    x2: _ArrayLikeInt_co,
    /, *,
    axis: SupportsIndex = ...,
) -> npt.NDArray[np.signedinteger[Any]]: ...
@overload
def cross(
    x1: _ArrayLikeFloat_co,
    x2: _ArrayLikeFloat_co,
    /, *,
    axis: SupportsIndex = ...,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def cross(
    x1: _ArrayLikeComplex_co,
    x2: _ArrayLikeComplex_co,
    /, *,
    axis: SupportsIndex = ...,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
@overload
def cross(
    x1: _ArrayLikeNumber_co,
    x2: _ArrayLikeNumber_co,
    /, *,
    axis: SupportsIndex = ...,
) -> npt.NDArray[np.number[Any]]: ...
@overload
def cross(
    x1: _ArrayLikeObject_co | _ArrayLikeNumber_co,
    x2: _ArrayLikeObject_co | _ArrayLikeNumber_co,
    /, *,
    axis: SupportsIndex = ...,
) -> npt.NDArray[np.object_]: ...

@overload
def matmul(
    x1: _SupportsArray[np.dtype[_SCT]],
    x2: _SupportsArray[np.dtype[_SCT]],
    /,
) -> npt.NDArray[_SCT]: ...
@overload
def matmul(  # type: ignore[overload-overlap]
    x1: _ArrayLikeBool_co,
    x2: _ArrayLikeBool_co,
    /,
) -> npt.NDArray[np.bool]: ...
@overload
def matmul(
    x1: _ArrayLikeUInt_co,
    x2: _ArrayLikeUInt_co,
    /,
) -> npt.NDArray[np.unsignedinteger[Any]]: ...
@overload
def matmul(
    x1: _ArrayLikeInt_co,
    x2: _ArrayLikeInt_co,
    /,
) -> npt.NDArray[np.signedinteger[Any]]: ...
@overload
def matmul(
    x1: _ArrayLikeFloat_co,
    x2: _ArrayLikeFloat_co,
    /,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def matmul(
    x1: _ArrayLikeComplex_co,
    x2: _ArrayLikeComplex_co,
    /,
) -> npt.NDArray[np.complexfloating[Any, Any]]: ...
@overload
def matmul(
    x1: _ArrayLikeNumber_co,
    x2: _ArrayLikeNumber_co,
    /,
) -> npt.NDArray[np.number[Any]]: ...
@overload
def matmul(
    x1: _ArrayLikeObject_co | _ArrayLikeNumber_co,
    x2: _ArrayLikeObject_co | _ArrayLikeNumber_co,
    /,
) -> npt.NDArray[np.object_]: ...

@overload
def vecdot(  # type: ignore[overload-overlap]
    x1: _ArrayLikeBool_co,
    x2: _ArrayLikeBool_co,
    /, *,
    axis: SupportsIndex = ...,
) -> np.bool | npt.NDArray[np.bool]: ...
@overload
def vecdot(
    x1: _ArrayLikeUInt_co,
    x2: _ArrayLikeUInt_co,
    /, *,
    axis: SupportsIndex = ...,
) -> np.unsignedinteger[Any] | npt.NDArray[np.unsignedinteger[Any]]: ...
@overload
def vecdot(
    x1: _ArrayLikeInt_co,
    x2: _ArrayLikeInt_co,
    /, *,
    axis: SupportsIndex = ...,
) -> np.signedinteger[Any] | npt.NDArray[np.signedinteger[Any]]: ...
@overload
def vecdot(
    x1: _ArrayLikeFloat_co,
    x2: _ArrayLikeFloat_co,
    /, *,
    axis: SupportsIndex = ...,
) -> np.floating[Any] | npt.NDArray[np.floating[Any]]: ...
@overload
def vecdot(
    x1: _ArrayLikeComplex_co,
    x2: _ArrayLikeComplex_co,
    /, *,
    axis: SupportsIndex = ...,
) -> (
    np.complexfloating[Any, Any]
    | npt.NDArray[np.complexfloating[Any, Any]]
): ...
@overload
def vecdot(
    x1: _ArrayLikeNumber_co,
    x2: _ArrayLikeNumber_co,
    /, *,
    axis: SupportsIndex = ...,
) -> np.number[Any] | npt.NDArray[np.number[Any]]: ...
@overload
def vecdot(
    x1: _ArrayLikeObject_co | _ArrayLikeNumber_co,
    x2: _ArrayLikeObject_co | _ArrayLikeNumber_co,
    /, *,
    axis: SupportsIndex = ...,
) -> object | npt.NDArray[np.object_]: ...
