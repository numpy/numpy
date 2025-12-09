from collections.abc import Iterable
from typing import (
    Any,
    Literal as L,
    NamedTuple,
    Never,
    SupportsIndex,
    SupportsInt,
    overload,
)

import numpy as np
from numpy import (
    complexfloating,
    float64,
    floating,
    int32,
    object_,
    signedinteger,
    timedelta64,
    unsignedinteger,
    vecdot,
)
from numpy._core.fromnumeric import matrix_transpose
from numpy._globals import _NoValue, _NoValueType
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeTD64_co,
    _ArrayLikeUInt_co,
    _NestedSequence,
    _ShapeLike,
)
from numpy.linalg import LinAlgError

__all__ = [
    "matrix_power",
    "solve",
    "tensorsolve",
    "tensorinv",
    "inv",
    "cholesky",
    "eigvals",
    "eigvalsh",
    "pinv",
    "slogdet",
    "det",
    "svd",
    "svdvals",
    "eig",
    "eigh",
    "lstsq",
    "norm",
    "qr",
    "cond",
    "matrix_rank",
    "LinAlgError",
    "multi_dot",
    "trace",
    "diagonal",
    "cross",
    "outer",
    "tensordot",
    "matmul",
    "matrix_transpose",
    "matrix_norm",
    "vector_norm",
    "vecdot",
]

type _ModeKind = L["reduced", "complete", "r", "raw"]
type _SideKind = L["L", "U", "l", "u"]

type _inexact64 = np.float32 | np.complex64

# anything that safe-casts (from floating) into float64/complex128
type _ToArrayF64 = _ArrayLike[np.float64 | np.integer | np.bool] | _NestedSequence[float]
type _ToArrayC128 = _ArrayLike[np.complex128 | np.float64 | np.integer | np.bool] | _NestedSequence[complex]
# the invariant `list` type avoids overlap with `_IntoArrayF64`
type _AsArrayC128 = _ArrayLike[np.complex128] | list[complex] | _NestedSequence[list[complex]]

###

fortran_int = np.intc

# TODO: generic
class EigResult(NamedTuple):
    eigenvalues: NDArray[Any]
    eigenvectors: NDArray[Any]

# TODO: generic
class EighResult(NamedTuple):
    eigenvalues: NDArray[Any]
    eigenvectors: NDArray[Any]

# TODO: generic
class QRResult(NamedTuple):
    Q: NDArray[Any]
    R: NDArray[Any]

# TODO: generic
class SlogdetResult(NamedTuple):
    # TODO: `sign` and `logabsdet` are scalars for input 2D arrays and
    # a `(x.ndim - 2)`` dimensional arrays otherwise
    sign: Any
    logabsdet: Any

# TODO: generic
class SVDResult(NamedTuple):
    U: NDArray[Any]
    S: NDArray[Any]
    Vh: NDArray[Any]

# TODO: narrow return types
@overload
def tensorsolve(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axes: Iterable[int] | None = None,
) -> NDArray[float64]: ...
@overload
def tensorsolve(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axes: Iterable[int] | None = None,
) -> NDArray[floating]: ...
@overload
def tensorsolve(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axes: Iterable[int] | None = None,
) -> NDArray[complexfloating]: ...

# TODO: narrow return types
@overload
def solve(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
) -> NDArray[float64]: ...
@overload
def solve(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
) -> NDArray[floating]: ...
@overload
def solve(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
) -> NDArray[complexfloating]: ...

# keep in sync with the other inverse functions and cholesky
@overload  # inexact32 array-likes
def tensorinv[ScalarT: _inexact64](a: _ArrayLike[ScalarT], ind: int = 2) -> NDArray[ScalarT]: ...
@overload  # +float64 array-likes
def tensorinv(a: _ToArrayF64, ind: int = 2) -> NDArray[np.float64]: ...
@overload  # ~complex128 array-likes
def tensorinv(a: _AsArrayC128, ind: int = 2) -> NDArray[np.complex128]: ...
@overload  # fallback
def tensorinv(a: _ArrayLikeComplex_co, ind: int = 2) -> np.ndarray: ...

# keep in sync with the other inverse functions and cholesky
@overload  # inexact32 array-likes
def inv[ScalarT: _inexact64](a: _ArrayLike[ScalarT]) -> NDArray[ScalarT]: ...
@overload  # +float64 array-likes
def inv(a: _ToArrayF64) -> NDArray[np.float64]: ...
@overload  # ~complex128 array-likes
def inv(a: _AsArrayC128) -> NDArray[np.complex128]: ...
@overload  # fallback
def inv(a: _ArrayLikeComplex_co) -> np.ndarray: ...

# keep in sync with the other inverse functions and cholesky
@overload  # inexact32 array-likes
def pinv[ScalarT: _inexact64](
    a: _ArrayLike[ScalarT],
    rcond: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | _NoValueType = _NoValue,
) -> NDArray[ScalarT]: ...
@overload  # +float64 array-likes
def pinv(
    a: _ToArrayF64,
    rcond: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | _NoValueType = _NoValue,
) -> NDArray[np.float64]: ...
@overload  # ~complex128 array-likes
def pinv(
    a: _AsArrayC128,
    rcond: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | _NoValueType = _NoValue,
) -> NDArray[np.complex128]: ...
@overload  # fallback
def pinv(
    a: _ArrayLikeComplex_co,
    rcond: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | _NoValueType = _NoValue,
) -> NDArray[Any]: ...

# keep in sync with the inverse functions
@overload  # inexact32 array-likes
def cholesky[ScalarT: _inexact64](a: _ArrayLike[ScalarT], /, *, upper: bool = False) -> NDArray[ScalarT]: ...
@overload  # +float64 array-likes
def cholesky(a: _ToArrayF64, /, *, upper: bool = False) -> NDArray[np.float64]: ...
@overload  # ~complex128 array-likes
def cholesky(a: _AsArrayC128, /, *, upper: bool = False) -> NDArray[np.complex128]: ...
@overload  # fallback
def cholesky(a: _ArrayLikeComplex_co, /, *, upper: bool = False) -> np.ndarray: ...

# TODO: The supported input and output dtypes are dependent on the value of `n`.
# For example: `n < 0` always casts integer types to float64
def matrix_power(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    n: SupportsIndex,
) -> NDArray[Any]: ...

# TODO: narrow return types
@overload
def outer(x1: _ArrayLike[Never], x2: _ArrayLike[Never], /) -> NDArray[Any]: ...
@overload
def outer(x1: _ArrayLikeBool_co, x2: _ArrayLikeBool_co, /) -> NDArray[np.bool]: ...
@overload
def outer[ScalarT: np.number](x1: _ArrayLike[ScalarT], x2: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload
def outer(x1: _ArrayLikeUInt_co, x2: _ArrayLikeUInt_co, /) -> NDArray[unsignedinteger]: ...
@overload
def outer(x1: _ArrayLikeInt_co, x2: _ArrayLikeInt_co, /) -> NDArray[signedinteger]: ...
@overload
def outer(x1: _ArrayLikeFloat_co, x2: _ArrayLikeFloat_co, /) -> NDArray[floating]: ...
@overload
def outer(x1: _ArrayLikeComplex_co, x2: _ArrayLikeComplex_co, /) -> NDArray[complexfloating]: ...
@overload
def outer(x1: _ArrayLikeTD64_co, x2: _ArrayLikeTD64_co, /) -> NDArray[timedelta64]: ...
@overload
def outer(x1: _ArrayLikeObject_co, x2: _ArrayLikeObject_co, /) -> NDArray[object_]: ...
@overload
def outer(
    x1: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    x2: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    /,
) -> NDArray[Any]: ...

# TODO: narrow return types
@overload
def qr(a: _ArrayLikeInt_co, mode: _ModeKind = "reduced") -> QRResult: ...
@overload
def qr(a: _ArrayLikeFloat_co, mode: _ModeKind = "reduced") -> QRResult: ...
@overload
def qr(a: _ArrayLikeComplex_co, mode: _ModeKind = "reduced") -> QRResult: ...

# TODO: narrow return types
@overload
def eig(a: _ArrayLikeInt_co) -> EigResult: ...
@overload
def eig(a: _ArrayLikeFloat_co) -> EigResult: ...
@overload
def eig(a: _ArrayLikeComplex_co) -> EigResult: ...

# TODO: narrow return types
@overload
def eigh(a: _ArrayLikeInt_co, UPLO: _SideKind = "L") -> EighResult: ...
@overload
def eigh(a: _ArrayLikeFloat_co, UPLO: _SideKind = "L") -> EighResult: ...
@overload
def eigh(a: _ArrayLikeComplex_co, UPLO: _SideKind = "L") -> EighResult: ...

# TODO: narrow return types
@overload
def svd(
    a: _ArrayLikeInt_co,
    full_matrices: bool = True,
    compute_uv: L[True] = True,
    hermitian: bool = False,
) -> SVDResult: ...
@overload
def svd(
    a: _ArrayLikeFloat_co,
    full_matrices: bool = True,
    compute_uv: L[True] = True,
    hermitian: bool = False,
) -> SVDResult: ...
@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool = True,
    compute_uv: L[True] = True,
    hermitian: bool = False,
) -> SVDResult: ...
@overload
def svd(
    a: _ArrayLikeInt_co,
    full_matrices: bool = True,
    *,
    compute_uv: L[False],
    hermitian: bool = False,
) -> NDArray[float64]: ...
@overload
def svd(
    a: _ArrayLikeInt_co,
    full_matrices: bool,
    compute_uv: L[False],
    hermitian: bool = False,
) -> NDArray[float64]: ...
@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool = True,
    *,
    compute_uv: L[False],
    hermitian: bool = False,
) -> NDArray[floating]: ...
@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool,
    compute_uv: L[False],
    hermitian: bool = False,
) -> NDArray[floating]: ...

# NOTE: for real input the output dtype (floating/complexfloating) depends on the specific values
@overload  # abstract `complexfloating` (excluding concrete types)
def eigvals(a: NDArray[np.complexfloating[Never]]) -> NDArray[np.complexfloating]: ...
@overload  # abstract `inexact` and `floating` (excluding concrete types)
def eigvals(a: NDArray[np.inexact[Never]]) -> np.ndarray: ...
@overload  # ~complex128
def eigvals(a: _AsArrayC128) -> NDArray[np.complex128]: ...
@overload  # +float64
def eigvals(a: _ToArrayF64) -> NDArray[np.complex128] | NDArray[np.float64]: ...
@overload  # ~complex64
def eigvals(a: _ArrayLike[np.complex64]) -> NDArray[np.complex64]: ...
@overload  # ~float32
def eigvals(a: _ArrayLike[np.float32]) -> NDArray[np.complex64] | NDArray[np.float32]: ...
@overload  # fallback
def eigvals(a: _ArrayLikeComplex_co) -> np.ndarray: ...

# keep in sync with svdvals
@overload  # abstract `inexact` (excluding concrete types)
def eigvalsh(a: NDArray[np.inexact[Never]], UPLO: _SideKind = "L") -> NDArray[np.floating]: ...
@overload  # ~inexact32
def eigvalsh(a: _ArrayLike[_inexact64], UPLO: _SideKind = "L") -> NDArray[np.float32]: ...
@overload  # +complex128
def eigvalsh(a: _ToArrayC128, UPLO: _SideKind = "L") -> NDArray[np.float64]: ...
@overload  # fallback
def eigvalsh(a: _ArrayLikeComplex_co, UPLO: _SideKind = "L") -> NDArray[np.floating]: ...

# keep in sync with eigvalsh
@overload  # abstract `inexact` (excluding concrete types)
def svdvals(a: NDArray[np.inexact[Never]], /) -> NDArray[np.floating]: ...
@overload  # ~inexact32
def svdvals(x: _ArrayLike[_inexact64], /) -> NDArray[np.float32]: ...
@overload  # +complex128
def svdvals(x: _ToArrayC128, /) -> NDArray[np.float64]: ...
@overload  # fallback
def svdvals(a: _ArrayLikeComplex_co, /) -> NDArray[np.floating]: ...

# TODO: Returns a scalar for 2D arrays and
# a `(x.ndim - 2)`` dimensional array otherwise
def cond(x: _ArrayLikeComplex_co, p: float | L["fro", "nuc"] | None = None) -> Any: ...

# TODO: Returns `int` for <2D arrays and `intp` otherwise
def matrix_rank(
    A: _ArrayLikeComplex_co,
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> Any: ...

# TODO: Returns a 2-tuple of scalars for 2D arrays and
# a 2-tuple of `(a.ndim - 2)`` dimensional arrays otherwise
def slogdet(a: _ArrayLikeComplex_co) -> SlogdetResult: ...

# TODO: Returns a 2-tuple of scalars for 2D arrays and
# a 2-tuple of `(a.ndim - 2)`` dimensional arrays otherwise
def det(a: _ArrayLikeComplex_co) -> Any: ...

# TODO: narrow return types
@overload
def lstsq(
    a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, rcond: float | None = None
) -> tuple[
    NDArray[float64],
    NDArray[float64],
    int32,
    NDArray[float64],
]: ...
@overload
def lstsq(
    a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, rcond: float | None = None
) -> tuple[
    NDArray[floating],
    NDArray[floating],
    int32,
    NDArray[floating],
]: ...
@overload
def lstsq(
    a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, rcond: float | None = None
) -> tuple[
    NDArray[complexfloating],
    NDArray[floating],
    int32,
    NDArray[floating],
]: ...

# TODO: narrow return types
@overload
def norm(
    x: ArrayLike,
    ord: float | L["fro", "nuc"] | None = None,
    axis: None = None,
    keepdims: L[False] = False,
) -> floating: ...
@overload
def norm(
    x: ArrayLike,
    ord: float | L["fro", "nuc"] | None,
    axis: SupportsInt | SupportsIndex | tuple[int, ...] | None,
    keepdims: bool = False,
) -> Any: ...
@overload
def norm(
    x: ArrayLike,
    ord: float | L["fro", "nuc"] | None = None,
    *,
    axis: SupportsInt | SupportsIndex | tuple[int, ...] | None,
    keepdims: bool = False,
) -> Any: ...

# TODO: narrow return types
@overload
def matrix_norm(
    x: ArrayLike,
    /,
    *,
    ord: float | L["fro", "nuc"] | None = "fro",
    keepdims: L[False] = False,
) -> floating: ...
@overload
def matrix_norm(
    x: ArrayLike,
    /,
    *,
    ord: float | L["fro", "nuc"] | None = "fro",
    keepdims: bool = False,
) -> Any: ...

# TODO: narrow return types
@overload
def vector_norm(
    x: ArrayLike,
    /,
    *,
    axis: None = None,
    ord: float | None = 2,
    keepdims: L[False] = False,
) -> floating: ...
@overload
def vector_norm(
    x: ArrayLike,
    /,
    *,
    axis: SupportsInt | SupportsIndex | tuple[int, ...],
    ord: float | None = 2,
    keepdims: bool = False,
) -> Any: ...

# keep in sync with numpy._core.numeric.tensordot (ignoring `/, *`)
@overload
def tensordot[ScalarT: np.number | np.timedelta64 | np.object_](
    a: _ArrayLike[ScalarT],
    b: _ArrayLike[ScalarT],
    /,
    *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = 2,
) -> NDArray[ScalarT]: ...
@overload
def tensordot(
    a: _ArrayLikeBool_co,
    b: _ArrayLikeBool_co,
    /,
    *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = 2,
) -> NDArray[np.bool_]: ...
@overload
def tensordot(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    /,
    *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = 2,
) -> NDArray[np.int_ | Any]: ...
@overload
def tensordot(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    /,
    *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = 2,
) -> NDArray[np.float64 | Any]: ...
@overload
def tensordot(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    /,
    *,
    axes: int | tuple[_ShapeLike, _ShapeLike] = 2,
) -> NDArray[np.complex128 | Any]: ...

# TODO: Returns a scalar or array
def multi_dot(
    arrays: Iterable[_ArrayLikeComplex_co | _ArrayLikeObject_co | _ArrayLikeTD64_co],
    *,
    out: NDArray[Any] | None = None,
) -> Any: ...

# TODO: narrow return types
def diagonal(
    x: ArrayLike,  # >= 2D array
    /,
    *,
    offset: SupportsIndex = 0,
) -> NDArray[Any]: ...

# TODO: narrow return types
def trace(
    x: ArrayLike,  # >= 2D array
    /,
    *,
    offset: SupportsIndex = 0,
    dtype: DTypeLike | None = None,
) -> Any: ...

# TODO: narrow return types
@overload
def cross(
    x1: _ArrayLikeUInt_co,
    x2: _ArrayLikeUInt_co,
    /,
    *,
    axis: int = -1,
) -> NDArray[unsignedinteger]: ...
@overload
def cross(
    x1: _ArrayLikeInt_co,
    x2: _ArrayLikeInt_co,
    /,
    *,
    axis: int = -1,
) -> NDArray[signedinteger]: ...
@overload
def cross(
    x1: _ArrayLikeFloat_co,
    x2: _ArrayLikeFloat_co,
    /,
    *,
    axis: int = -1,
) -> NDArray[floating]: ...
@overload
def cross(
    x1: _ArrayLikeComplex_co,
    x2: _ArrayLikeComplex_co,
    /,
    *,
    axis: int = -1,
) -> NDArray[complexfloating]: ...

# TODO: narrow return types
@overload
def matmul[ScalarT: np.number](x1: _ArrayLike[ScalarT], x2: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload
def matmul(x1: _ArrayLikeUInt_co, x2: _ArrayLikeUInt_co, /) -> NDArray[unsignedinteger]: ...
@overload
def matmul(x1: _ArrayLikeInt_co, x2: _ArrayLikeInt_co, /) -> NDArray[signedinteger]: ...
@overload
def matmul(x1: _ArrayLikeFloat_co, x2: _ArrayLikeFloat_co, /) -> NDArray[floating]: ...
@overload
def matmul(x1: _ArrayLikeComplex_co, x2: _ArrayLikeComplex_co, /) -> NDArray[complexfloating]: ...
