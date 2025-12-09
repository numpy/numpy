from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Generic,
    Literal as L,
    NamedTuple,
    Never,
    SupportsIndex,
    SupportsInt,
    overload,
)
from typing_extensions import TypeVar

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

type _AtMost1D = tuple[()] | tuple[int]
type _AtLeast3D = tuple[int, int, int, *tuple[int, ...]]
type _AtLeast4D = tuple[int, int, int, int, *tuple[int, ...]]
type _JustAnyShape = tuple[Never, ...]  # workaround for microsoft/pyright#10232

type _tuple2[T] = tuple[T, T]

type _inexact32 = np.float32 | np.complex64
type _to_float64 = np.float64 | np.integer | np.bool
type _to_inexact64 = np.complex128 | _to_float64

type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3ND[ScalarT: np.generic] = np.ndarray[_AtLeast3D, np.dtype[ScalarT]]

# anything that safe-casts (from floating) into float64/complex128
type _ToArrayF64 = _ArrayLike[_to_float64] | _NestedSequence[float]
type _ToArrayC128 = _ArrayLike[_to_inexact64] | _NestedSequence[complex]
# the invariant `list` type avoids overlap with bool, int, etc
type _AsArrayF64 = _ArrayLike[np.float64] | list[float] | _NestedSequence[list[float]]
type _AsArrayC128 = _ArrayLike[np.complex128] | list[complex] | _NestedSequence[list[complex]]

type _ToArrayF64_2d = _Array2D[_to_float64] | Sequence[Sequence[float]]
type _ToArrayF64_3nd = _Array3ND[_to_float64] | Sequence[Sequence[_NestedSequence[float]]]
type _ToArrayC128_2d = _Array2D[_to_inexact64] | Sequence[Sequence[complex]]
type _ToArrayC128_3nd = _Array3ND[_to_inexact64] | Sequence[Sequence[_NestedSequence[complex]]]

type _OrderKind = L[1, -1, 2, -2, "fro", "nuc"] | float  # only accepts `-inf` and `inf` as `float`
type _SideKind = L["L", "U", "l", "u"]
type _NonNegInt = L[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
type _NegInt = L[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16]

_FloatingT_co = TypeVar("_FloatingT_co", bound=np.floating, default=Any, covariant=True)
_FloatingOrArrayT_co = TypeVar("_FloatingOrArrayT_co", bound=np.floating | NDArray[np.floating], default=Any, covariant=True)
_InexactT_co = TypeVar("_InexactT_co", bound=np.inexact, default=Any, covariant=True)
_InexactOrArrayT_co = TypeVar("_InexactOrArrayT_co", bound=np.inexact | NDArray[np.inexact], default=Any, covariant=True)

###

fortran_int = np.intc

# NOTE: These named tuple types are only generic when `typing.TYPE_CHECKING`

class EigResult(NamedTuple, Generic[_InexactT_co]):
    eigenvalues: NDArray[_InexactT_co]
    eigenvectors: NDArray[_InexactT_co]

class EighResult(NamedTuple, Generic[_FloatingT_co, _InexactT_co]):
    eigenvalues: NDArray[_FloatingT_co]
    eigenvectors: NDArray[_InexactT_co]

class QRResult(NamedTuple, Generic[_InexactT_co]):
    Q: NDArray[_InexactT_co]
    R: NDArray[_InexactT_co]

class SVDResult(NamedTuple, Generic[_FloatingT_co, _InexactT_co]):
    U: NDArray[_InexactT_co]
    S: NDArray[_FloatingT_co]
    Vh: NDArray[_InexactT_co]

class SlogdetResult(NamedTuple, Generic[_FloatingOrArrayT_co, _InexactOrArrayT_co]):
    sign: _FloatingOrArrayT_co
    logabsdet: _InexactOrArrayT_co

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
@overload  # inexact32
def tensorinv[ScalarT: _inexact32](a: _ArrayLike[ScalarT], ind: int = 2) -> NDArray[ScalarT]: ...
@overload  # +float64
def tensorinv(a: _ToArrayF64, ind: int = 2) -> NDArray[np.float64]: ...
@overload  # ~complex128
def tensorinv(a: _AsArrayC128, ind: int = 2) -> NDArray[np.complex128]: ...
@overload  # fallback
def tensorinv(a: _ArrayLikeComplex_co, ind: int = 2) -> np.ndarray: ...

# keep in sync with the other inverse functions and cholesky
@overload  # inexact32
def inv[ScalarT: _inexact32](a: _ArrayLike[ScalarT]) -> NDArray[ScalarT]: ...
@overload  # +float64
def inv(a: _ToArrayF64) -> NDArray[np.float64]: ...
@overload  # ~complex128
def inv(a: _AsArrayC128) -> NDArray[np.complex128]: ...
@overload  # fallback
def inv(a: _ArrayLikeComplex_co) -> np.ndarray: ...

# keep in sync with the other inverse functions and cholesky
@overload  # inexact32
def pinv[ScalarT: _inexact32](
    a: _ArrayLike[ScalarT],
    rcond: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | _NoValueType = _NoValue,
) -> NDArray[ScalarT]: ...
@overload  # +float64
def pinv(
    a: _ToArrayF64,
    rcond: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | _NoValueType = _NoValue,
) -> NDArray[np.float64]: ...
@overload  # ~complex128
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
@overload  # inexact32
def cholesky[ScalarT: _inexact32](a: _ArrayLike[ScalarT], /, *, upper: bool = False) -> NDArray[ScalarT]: ...
@overload  # +float64
def cholesky(a: _ToArrayF64, /, *, upper: bool = False) -> NDArray[np.float64]: ...
@overload  # ~complex128
def cholesky(a: _AsArrayC128, /, *, upper: bool = False) -> NDArray[np.complex128]: ...
@overload  # fallback
def cholesky(a: _ArrayLikeComplex_co, /, *, upper: bool = False) -> np.ndarray: ...

# NOTE: Technically this also accepts boolean array-likes, but that case is not very useful, so we skip it.
#       If you have a use case for it, please open an issue.
@overload  # +int, n ≥ 0
def matrix_power(a: _NestedSequence[int], n: _NonNegInt) -> NDArray[np.int_]: ...
@overload  # +integer | ~object, n ≥ 0
def matrix_power[ScalarT: np.integer | np.object_](a: _ArrayLike[ScalarT], n: _NonNegInt) -> NDArray[ScalarT]: ...
@overload  # +float64, n < 0
def matrix_power(a: _ToArrayF64, n: _NegInt) -> NDArray[np.float64]: ...
@overload  # ~float64
def matrix_power(a: _AsArrayF64, n: SupportsIndex) -> NDArray[np.float64]: ...
@overload  # ~complex128
def matrix_power(a: _AsArrayC128, n: SupportsIndex) -> NDArray[np.complex128]: ...
@overload  # ~inexact32
def matrix_power[ScalarT: _inexact32](a: _ArrayLike[ScalarT], n: SupportsIndex) -> NDArray[ScalarT]: ...
@overload  # fallback
def matrix_power(a: _ArrayLikeComplex_co | _ArrayLikeObject_co, n: SupportsIndex) -> np.ndarray: ...

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

# NOTE: for real input the output dtype (floating/complexfloating) depends on the specific values
@overload  # abstract `inexact` and `floating` (excluding concrete types)
def eig(a: NDArray[np.inexact[Never]]) -> EigResult: ...
@overload  # ~complex128
def eig(a: _AsArrayC128) -> EigResult[np.complex128]: ...
@overload  # +float64
def eig(a: _ToArrayF64) -> EigResult[np.complex128] | EigResult[np.float64]: ...
@overload  # ~complex64
def eig(a: _ArrayLike[np.complex64]) -> EigResult[np.complex64]: ...
@overload  # ~float32
def eig(a: _ArrayLike[np.float32]) -> EigResult[np.complex64] | EigResult[np.float32]: ...
@overload  # fallback
def eig(a: _ArrayLikeComplex_co) -> EigResult: ...

#
@overload  # workaround for microsoft/pyright#10232
def eigh(a: NDArray[Never], UPLO: _SideKind = "L") -> EighResult: ...
@overload  # ~inexact32
def eigh[ScalarT: _inexact32](a: _ArrayLike[ScalarT], UPLO: _SideKind = "L") -> EighResult[np.float32, ScalarT]: ...
@overload  # +float64
def eigh(a: _ToArrayF64, UPLO: _SideKind = "L") -> EighResult[np.float64, np.float64]: ...
@overload  # ~complex128
def eigh(a: _AsArrayC128, UPLO: _SideKind = "L") -> EighResult[np.float64, np.complex128]: ...
@overload  # fallback
def eigh(a: _ArrayLikeComplex_co, UPLO: _SideKind = "L") -> EighResult: ...

#
@overload  # ~inexact32,  reduced|complete
def qr[ScalarT: _inexact32](a: _ArrayLike[ScalarT], mode: L["reduced", "complete"] = "reduced") -> QRResult[ScalarT]: ...
@overload  # ~inexact32,  r
def qr[ScalarT: _inexact32](a: _ArrayLike[ScalarT], mode: L["r"]) -> NDArray[ScalarT]: ...
@overload  # ~inexact32,  raw
def qr[ScalarT: _inexact32](a: _ArrayLike[ScalarT], mode: L["raw"]) -> _tuple2[NDArray[ScalarT]]: ...
@overload  # +float64,    reduced|complete
def qr(a: _ToArrayF64, mode: L["reduced", "complete"] = "reduced") -> QRResult[np.float64]: ...
@overload  # +float64,    r
def qr(a: _ToArrayF64, mode: L["r"]) -> NDArray[np.float64]: ...
@overload  # +float64,    raw
def qr(a: _ToArrayF64, mode: L["raw"]) -> _tuple2[NDArray[np.float64]]: ...
@overload  # ~complex128, reduced|complete
def qr(a: _AsArrayC128, mode: L["reduced", "complete"] = "reduced") -> QRResult[np.complex128]: ...
@overload  # ~complex128, r
def qr(a: _AsArrayC128, mode: L["r"]) -> NDArray[np.complex128]: ...
@overload  # ~complex128, raw
def qr(a: _AsArrayC128, mode: L["raw"]) -> _tuple2[NDArray[np.complex128]]: ...
@overload  # fallback,    reduced|complete
def qr(a: _ArrayLikeComplex_co, mode: L["reduced", "complete"] = "reduced") -> QRResult: ...
@overload  # fallback,    r
def qr(a: _ArrayLikeComplex_co, mode: L["r"]) -> np.ndarray: ...
@overload  # fallback,    raw
def qr(a: _ArrayLikeComplex_co, mode: L["raw"]) -> _tuple2[np.ndarray]: ...

#
@overload  # workaround for microsoft/pyright#10232, compute_uv=True (default)
def svd(a: NDArray[Never], full_matrices: bool = True, compute_uv: L[True] = True, hermitian: bool = False) -> SVDResult: ...
@overload  # workaround for microsoft/pyright#10232, compute_uv=False (positional)
def svd(a: NDArray[Never], full_matrices: bool, compute_uv: L[False], hermitian: bool = False) -> np.ndarray: ...
@overload  # workaround for microsoft/pyright#10232, compute_uv=False (keyword)
def svd(a: NDArray[Never], full_matrices: bool = True, *, compute_uv: L[False], hermitian: bool = False) -> np.ndarray: ...
@overload  # ~inexact32, compute_uv=True (default)
def svd[ScalarT: _inexact32](
    a: _ArrayLike[ScalarT], full_matrices: bool = True, compute_uv: L[True] = True, hermitian: bool = False
) -> SVDResult[np.float32, ScalarT]: ...
@overload  # ~inexact32, compute_uv=False (positional)
def svd(a: _ArrayLike[_inexact32], full_matrices: bool, compute_uv: L[False], hermitian: bool = False) -> NDArray[np.float32]: ...
@overload  # ~inexact32, compute_uv=False (keyword)
def svd(
    a: _ArrayLike[_inexact32], full_matrices: bool = True, *, compute_uv: L[False], hermitian: bool = False
) -> NDArray[np.float32]: ...
@overload  # +float64, compute_uv=True (default)
def svd(
    a: _ToArrayF64, full_matrices: bool = True, compute_uv: L[True] = True, hermitian: bool = False
) -> SVDResult[np.float64, np.float64]: ...
@overload  # ~complex128, compute_uv=True (default)
def svd(
    a: _AsArrayC128, full_matrices: bool = True, compute_uv: L[True] = True, hermitian: bool = False
) -> SVDResult[np.float64, np.complex128]: ...
@overload  # +float64 | ~complex128, compute_uv=False (positional)
def svd(a: _ToArrayC128, full_matrices: bool, compute_uv: L[False], hermitian: bool = False) -> NDArray[np.float64]: ...
@overload  # +float64 | ~complex128, compute_uv=False (keyword)
def svd(a: _ToArrayC128, full_matrices: bool = True, *, compute_uv: L[False], hermitian: bool = False) -> NDArray[np.float64]: ...
@overload  # fallback, compute_uv=True (default)
def svd(
    a: _ArrayLikeComplex_co, full_matrices: bool = True, compute_uv: L[True] = True, hermitian: bool = False
) -> SVDResult: ...
@overload  # fallback, compute_uv=False (positional)
def svd(a: _ArrayLikeComplex_co, full_matrices: bool, compute_uv: L[False], hermitian: bool = False) -> np.ndarray: ...
@overload  # fallback, compute_uv=False (keyword)
def svd(a: _ArrayLikeComplex_co, full_matrices: bool = True, *, compute_uv: L[False], hermitian: bool = False) -> np.ndarray: ...

# NOTE: for real input the output dtype (floating/complexfloating) depends on the specific values
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
def eigvalsh(a: _ArrayLike[_inexact32], UPLO: _SideKind = "L") -> NDArray[np.float32]: ...
@overload  # +complex128
def eigvalsh(a: _ToArrayC128, UPLO: _SideKind = "L") -> NDArray[np.float64]: ...
@overload  # fallback
def eigvalsh(a: _ArrayLikeComplex_co, UPLO: _SideKind = "L") -> NDArray[np.floating]: ...

# keep in sync with eigvalsh
@overload  # abstract `inexact` (excluding concrete types)
def svdvals(a: NDArray[np.inexact[Never]], /) -> NDArray[np.floating]: ...
@overload  # ~inexact32
def svdvals(a: _ArrayLike[_inexact32], /) -> NDArray[np.float32]: ...
@overload  # +complex128
def svdvals(a: _ToArrayC128, /) -> NDArray[np.float64]: ...
@overload  # fallback
def svdvals(a: _ArrayLikeComplex_co, /) -> NDArray[np.floating]: ...

#
@overload  # workaround for microsoft/pyright#10232
def matrix_rank(
    A: np.ndarray[tuple[Never, ...], np.dtype[np.number]],
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> Any: ...
@overload  # <2d
def matrix_rank(
    A: complex | Sequence[complex] | np.ndarray[_AtMost1D, np.dtype[np.number]],
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> L[0, 1]: ...
@overload  # =2d
def matrix_rank(
    A: Sequence[Sequence[complex]] | _Array2D[np.number],
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> np.int_: ...
@overload  # =3d
def matrix_rank(
    A: Sequence[Sequence[Sequence[complex]]] | np.ndarray[tuple[int, int, int], np.dtype[np.number]],
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.int_]]: ...
@overload  # ≥4d
def matrix_rank(
    A: Sequence[Sequence[Sequence[_NestedSequence[complex]]]] | np.ndarray[_AtLeast4D, np.dtype[np.number]],
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> NDArray[np.int_]: ...
@overload  # ?d
def matrix_rank(
    A: _ArrayLikeComplex_co,
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> Any: ...

#
@overload  # workaround for microsoft/pyright#10232
def cond(x: np.ndarray[_JustAnyShape, np.dtype[np.number]], p: _OrderKind | None = None) -> Any: ...
@overload  # 2d ~inexact32
def cond(x: _Array2D[_inexact32], p: _OrderKind | None = None) -> np.float32: ...
@overload  # 2d +inexact64
def cond(x: _ToArrayC128_2d, p: _OrderKind | None = None) -> np.float64: ...
@overload  # 2d ~number
def cond(x: _Array2D[np.number], p: _OrderKind | None = None) -> np.floating: ...
@overload  # >2d ~inexact32
def cond(x: np.ndarray[_AtLeast3D, np.dtype[_inexact32]], p: _OrderKind | None = None) -> NDArray[np.float32]: ...
@overload  # >2d +inexact64
def cond(x: _ToArrayC128_3nd, p: _OrderKind | None = None) -> NDArray[np.float64]: ...
@overload  # >2d ~number
def cond(x: np.ndarray[_AtLeast3D, np.dtype[np.number]], p: _OrderKind | None = None) -> NDArray[np.floating]: ...
@overload  # fallback
def cond(x: _ArrayLikeComplex_co, p: _OrderKind | None = None) -> Any: ...

# keep in sync with `det`
@overload  # workaround for microsoft/pyright#10232
def slogdet(a: np.ndarray[_JustAnyShape, np.dtype[np.number]]) -> SlogdetResult: ...
@overload  # 2d ~inexact32
def slogdet[ScalarT: _inexact32](a: _Array2D[ScalarT]) -> SlogdetResult[np.float32, ScalarT]: ...
@overload  # >2d ~inexact32
def slogdet[ScalarT: _inexact32](a: _Array3ND[ScalarT]) -> SlogdetResult[NDArray[np.float32], NDArray[ScalarT]]: ...
@overload  # 2d +float64
def slogdet(a: _Array2D[_to_float64]) -> SlogdetResult[np.float64, np.float64]: ...
@overload  # >2d +float64
def slogdet(a: _Array3ND[_to_float64]) -> SlogdetResult[NDArray[np.float64], NDArray[np.float64]]: ...
@overload  # 2d ~complex128
def slogdet(a: _Array2D[np.complex128] | Sequence[list[complex]]) -> SlogdetResult[np.float64, np.complex128]: ...
@overload  # >2d ~complex128
def slogdet(
    a: _Array3ND[np.complex128] | _NestedSequence[Sequence[list[complex]]]
) -> SlogdetResult[NDArray[np.float64], NDArray[np.complex128]]: ...
@overload  # fallback
def slogdet(a: _ArrayLikeComplex_co) -> SlogdetResult: ...

# keep in sync with `slogdet`
@overload  # workaround for microsoft/pyright#10232
def det(a: np.ndarray[_JustAnyShape, np.dtype[np.number]]) -> Any: ...
@overload  # 2d ~inexact32
def det[ScalarT: _inexact32](a: _Array2D[ScalarT]) -> ScalarT: ...
@overload  # >2d ~inexact32
def det[ScalarT: _inexact32](a: _Array3ND[ScalarT]) -> NDArray[ScalarT]: ...
@overload  # 2d +float64
def det(a: _Array2D[_to_float64]) -> np.float64: ...
@overload  # >2d +float64
def det(a: _Array3ND[_to_float64]) -> NDArray[np.float64]: ...
@overload  # 2d ~complex128
def det(a: _Array2D[np.complex128] | Sequence[list[complex]]) -> np.complex128: ...
@overload  # >2d ~complex128
def det(a: _Array3ND[np.complex128] | _NestedSequence[Sequence[list[complex]]]) -> NDArray[np.complex128]: ...
@overload  # fallback
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
