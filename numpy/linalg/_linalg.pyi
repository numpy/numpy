from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Generic,
    Literal as L,
    NamedTuple,
    Never,
    Protocol,
    SupportsIndex,
    overload,
    type_check_only,
)
from typing_extensions import TypeVar

import numpy as np
from numpy import vecdot
from numpy._core.fromnumeric import matrix_transpose
from numpy._globals import _NoValue, _NoValueType
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _AnyShape,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeTD64_co,
    _ArrayLikeUInt_co,
    _DTypeLike,
    _NestedSequence,
    _Shape,
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
type _AtLeast2D = tuple[int, int, *tuple[int, ...]]
type _AtLeast3D = tuple[int, int, int, *tuple[int, ...]]
type _AtLeast4D = tuple[int, int, int, int, *tuple[int, ...]]
type _JustAnyShape = tuple[Never, ...]  # workaround for microsoft/pyright#10232

type _tuple2[T] = tuple[T, T]
type _Ax2 = SupportsIndex | _tuple2[SupportsIndex]

type _inexact32 = np.float32 | np.complex64
type _inexact80 = np.longdouble | np.clongdouble
type _to_integer = np.integer | np.bool
type _to_timedelta64 = np.timedelta64 | _to_integer
type _to_float64 = np.float64 | _to_integer
type _to_inexact64 = np.complex128 | _to_float64
type _to_inexact64_unsafe = _to_inexact64 | np.datetime64 | np.timedelta64 | np.character
type _to_complex = np.number | np.bool

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3ND[ScalarT: np.generic] = np.ndarray[_AtLeast3D, np.dtype[ScalarT]]

type _Sequence2D[T] = Sequence[Sequence[T]]
type _Sequence3D[T] = Sequence[_Sequence2D[T]]
type _Sequence2ND[T] = _NestedSequence[Sequence[T]]
type _Sequence3ND[T] = _NestedSequence[_Sequence2D[T]]
type _Sequence4ND[T] = _NestedSequence[_Sequence3D[T]]
type _Sequence0D1D[T] = T | Sequence[T]
type _Sequence1D2D[T] = Sequence[T] | _Sequence2D[T]

type _ArrayLike1D[ScalarT: np.generic] = _SupportsArray[tuple[int], np.dtype[ScalarT]] | Sequence[ScalarT]  # ==1d
type _ArrayLike2D[ScalarT: np.generic] = _SupportsArray[tuple[int, int], np.dtype[ScalarT]] | _Sequence2D[ScalarT]  # ==2d
type _ArrayLike1D2D[ScalarT: np.generic] = (  # 1d or 2d
    _SupportsArray[tuple[int] | tuple[int, int], np.dtype[ScalarT]] | _Sequence1D2D[ScalarT]
)
type _ArrayLike3D[ScalarT: np.generic] = _SupportsArray[tuple[int, int, int], np.dtype[ScalarT]] | _Sequence3D[ScalarT]  # ==3d
type _ArrayLike2ND[ScalarT: np.generic] = _SupportsArray[_AtLeast2D, np.dtype[ScalarT]] | _Sequence2ND[ScalarT]  # >=2d
type _ArrayLike3ND[ScalarT: np.generic] = _SupportsArray[_AtLeast3D, np.dtype[ScalarT]] | _Sequence3ND[ScalarT]  # >=3d
type _ArrayLike4ND[ScalarT: np.generic] = _SupportsArray[_AtLeast4D, np.dtype[ScalarT]] | _Sequence4ND[ScalarT]  # >=3d

# safe-castable array-likes
type _ToArrayBool_1d = _ArrayLike1D[np.bool_] | Sequence[bool]
type _ToArrayInt_1d = _ArrayLike1D[_to_integer] | Sequence[int]
type _ToArrayF64 = _ArrayLike[_to_float64] | _NestedSequence[float]
type _ToArrayF64_1d = _ArrayLike1D[_to_float64] | Sequence[float]
type _ToArrayF64_2d = _ArrayLike2D[_to_float64] | _Sequence2D[float]
type _ToArrayF64_3nd = _ArrayLike3ND[_to_float64] | _Sequence3ND[float]
type _ToArrayC128 = _ArrayLike[_to_inexact64] | _NestedSequence[complex]
type _ToArrayC128_3nd = _ArrayLike3ND[_to_inexact64] | _Sequence3ND[complex]
type _ToArrayComplex_1d = _ArrayLike1D[_to_complex] | Sequence[complex]
type _ToArrayComplex_2d = _ArrayLike2D[_to_complex] | _Sequence2D[complex]
type _ToArrayComplex_3d = _ArrayLike3D[_to_complex] | _Sequence3D[complex]
# the invariant `list` type avoids overlap with bool, int, etc
type _AsArrayI64 = _ArrayLike[np.int64] | list[int] | _NestedSequence[list[int]]
type _AsArrayI64_1d = _ArrayLike1D[np.int64] | list[int]
type _AsArrayF64 = _ArrayLike[np.float64] | list[float] | _NestedSequence[list[float]]
type _AsArrayF64_1d = _ArrayLike1D[np.float64] | list[float]
type _AsArrayC128 = _ArrayLike[np.complex128] | list[complex] | _NestedSequence[list[complex]]
type _AsArrayC128_1d = _ArrayLike1D[np.complex128] | list[complex]
type _AsArrayC128_2d = _ArrayLike2D[np.complex128] | Sequence[list[complex]]
type _AsArrayC128_3nd = _ArrayLike3ND[np.complex128] | _Sequence2ND[list[complex]]

type _OrderKind = L[1, -1, 2, -2, "fro", "nuc"] | float  # only accepts `-inf` and `inf` as `float`
type _SideKind = L["L", "U", "l", "u"]
type _NonNegInt = L[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
type _NegInt = L[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16]

type _LstSqResult[ShapeT: _Shape, InexactT: np.inexact, FloatingT: np.floating] = tuple[
    np.ndarray[ShapeT, np.dtype[InexactT]],  # least-squares solution
    _Array1D[FloatingT],  # residuals
    np.int32,  # rank
    _Array1D[FloatingT],  # singular values
]

_FloatingT_co = TypeVar("_FloatingT_co", bound=np.floating, default=Any, covariant=True)
_FloatingOrArrayT_co = TypeVar("_FloatingOrArrayT_co", bound=np.floating | NDArray[np.floating], default=Any, covariant=True)
_InexactT_co = TypeVar("_InexactT_co", bound=np.inexact, default=Any, covariant=True)
_InexactOrArrayT_co = TypeVar("_InexactOrArrayT_co", bound=np.inexact | NDArray[np.inexact], default=Any, covariant=True)

# shape-typed variant of numpy._typing._SupportsArray
@type_check_only
class _SupportsArray[ShapeT: _Shape, DTypeT: np.dtype](Protocol):
    def __array__(self, /) -> np.ndarray[ShapeT, DTypeT]: ...

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

# keep in sync with `solve`
@overload  # ~float64, +float64
def tensorsolve(a: _ToArrayF64, b: _ArrayLikeFloat_co, axes: Iterable[int] | None = None) -> NDArray[np.float64]: ...
@overload  # +float64, ~float64
def tensorsolve(a: _ArrayLikeFloat_co, b: _ToArrayF64, axes: Iterable[int] | None = None) -> NDArray[np.float64]: ...
@overload  # ~float32, ~float32
def tensorsolve(
    a: _ArrayLike[np.float32], b: _ArrayLike[np.float32], axes: Iterable[int] | None = None
) -> NDArray[np.float32]: ...
@overload  # +float, +float
def tensorsolve(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, axes: Iterable[int] | None = None) -> NDArray[np.float64 | Any]: ...
@overload  # ~complex128, +complex128
def tensorsolve(a: _AsArrayC128, b: _ArrayLikeComplex_co, axes: Iterable[int] | None = None) -> NDArray[np.complex128]: ...
@overload  # +complex128, ~complex128
def tensorsolve(a: _ArrayLikeComplex_co, b: _AsArrayC128, axes: Iterable[int] | None = None) -> NDArray[np.complex128]: ...
@overload  # ~complex64, +complex64
def tensorsolve(
    a: _ArrayLike[np.complex64], b: _ArrayLike[_inexact32], axes: Iterable[int] | None = None
) -> NDArray[np.complex64]: ...
@overload  # +complex64, ~complex64
def tensorsolve(
    a: _ArrayLike[_inexact32], b: _ArrayLike[np.complex64], axes: Iterable[int] | None = None
) -> NDArray[np.complex64]: ...
@overload  # +complex, +complex
def tensorsolve(
    a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, axes: Iterable[int] | None = None
) -> NDArray[np.complex128 | Any]: ...

# keep in sync with `tensorsolve`
@overload  # ~float64, +float64
def solve(a: _ToArrayF64, b: _ArrayLikeFloat_co) -> NDArray[np.float64]: ...
@overload  # +float64, ~float64
def solve(a: _ArrayLikeFloat_co, b: _ToArrayF64) -> NDArray[np.float64]: ...
@overload  # ~float32, ~float32
def solve(a: _ArrayLike[np.float32], b: _ArrayLike[np.float32]) -> NDArray[np.float32]: ...
@overload  # +float, +float
def solve(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co) -> NDArray[np.float64 | Any]: ...
@overload  # ~complex128, +complex128
def solve(a: _AsArrayC128, b: _ArrayLikeComplex_co) -> NDArray[np.complex128]: ...
@overload  # +complex128, ~complex128
def solve(a: _ArrayLikeComplex_co, b: _AsArrayC128) -> NDArray[np.complex128]: ...
@overload  # ~complex64, +complex64
def solve(a: _ArrayLike[np.complex64], b: _ArrayLike[_inexact32]) -> NDArray[np.complex64]: ...
@overload  # +complex64, ~complex64
def solve(a: _ArrayLike[_inexact32], b: _ArrayLike[np.complex64]) -> NDArray[np.complex64]: ...
@overload  # +complex, +complex
def solve(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co) -> NDArray[np.complex128 | Any]: ...

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
    A: np.ndarray[_JustAnyShape, np.dtype[_to_complex]],
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> Any: ...
@overload  # <2d
def matrix_rank(
    A: _SupportsArray[_AtMost1D, np.dtype[_to_complex]] | Sequence[complex | _to_complex] | complex | _to_complex,
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> L[0, 1]: ...
@overload  # =2d
def matrix_rank(
    A: _ToArrayComplex_2d,
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> np.int_: ...
@overload  # =3d
def matrix_rank(
    A: _ToArrayComplex_3d,
    tol: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | None = None,
) -> _Array1D[np.int_]: ...
@overload  # ≥4d
def matrix_rank(
    A: _ArrayLike4ND[_to_complex] | _Sequence4ND[complex],
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
def cond(x: np.ndarray[_JustAnyShape, np.dtype[_to_complex]], p: _OrderKind | None = None) -> Any: ...
@overload  # 2d ~inexact32
def cond(x: _ArrayLike2D[_inexact32], p: _OrderKind | None = None) -> np.float32: ...
@overload  # 2d +inexact64
def cond(x: _ArrayLike2D[_to_inexact64] | _Sequence2D[complex], p: _OrderKind | None = None) -> np.float64: ...
@overload  # 2d ~number
def cond(x: _ArrayLike2D[_to_complex], p: _OrderKind | None = None) -> np.floating: ...
@overload  # >2d ~inexact32
def cond(x: _ArrayLike3ND[_inexact32], p: _OrderKind | None = None) -> NDArray[np.float32]: ...
@overload  # >2d +inexact64
def cond(x: _ToArrayC128_3nd, p: _OrderKind | None = None) -> NDArray[np.float64]: ...
@overload  # >2d ~number
def cond(x: _ArrayLike3ND[_to_complex], p: _OrderKind | None = None) -> NDArray[np.floating]: ...
@overload  # fallback
def cond(x: _ArrayLikeComplex_co, p: _OrderKind | None = None) -> Any: ...

# keep in sync with `det`
@overload  # workaround for microsoft/pyright#10232
def slogdet(a: np.ndarray[_JustAnyShape, np.dtype[_to_complex]]) -> SlogdetResult: ...
@overload  # 2d ~inexact32
def slogdet[ScalarT: _inexact32](a: _ArrayLike2D[ScalarT]) -> SlogdetResult[np.float32, ScalarT]: ...
@overload  # >2d ~inexact32
def slogdet[ScalarT: _inexact32](a: _ArrayLike3ND[ScalarT]) -> SlogdetResult[NDArray[np.float32], NDArray[ScalarT]]: ...
@overload  # 2d +float64
def slogdet(a: _ArrayLike2D[_to_float64]) -> SlogdetResult[np.float64, np.float64]: ...
@overload  # >2d +float64
def slogdet(a: _ArrayLike3ND[_to_float64]) -> SlogdetResult[NDArray[np.float64], NDArray[np.float64]]: ...
@overload  # 2d ~complex128
def slogdet(a: _AsArrayC128_2d) -> SlogdetResult[np.float64, np.complex128]: ...
@overload  # >2d ~complex128
def slogdet(a: _AsArrayC128_3nd) -> SlogdetResult[NDArray[np.float64], NDArray[np.complex128]]: ...
@overload  # fallback
def slogdet(a: _ArrayLikeComplex_co) -> SlogdetResult: ...

# keep in sync with `slogdet`
@overload  # workaround for microsoft/pyright#10232
def det(a: np.ndarray[_JustAnyShape, np.dtype[_to_complex]]) -> Any: ...
@overload  # 2d ~inexact32
def det[ScalarT: _inexact32](a: _ArrayLike2D[ScalarT]) -> ScalarT: ...
@overload  # >2d ~inexact32
def det[ScalarT: _inexact32](a: _ArrayLike3ND[ScalarT]) -> NDArray[ScalarT]: ...
@overload  # 2d +float64
def det(a: _ArrayLike2D[_to_float64]) -> np.float64: ...
@overload  # >2d +float64
def det(a: _ArrayLike3ND[_to_float64]) -> NDArray[np.float64]: ...
@overload  # 2d ~complex128
def det(a: _AsArrayC128_2d) -> np.complex128: ...
@overload  # >2d ~complex128
def det(a: _AsArrayC128_3nd) -> NDArray[np.complex128]: ...
@overload  # fallback
def det(a: _ArrayLikeComplex_co) -> Any: ...

#
@overload  # +float64, ~float64, known shape
def lstsq[ShapeT: tuple[int] | tuple[int, int]](
    a: _ArrayLike2D[_to_float64] | _Sequence2D[float],
    b: _SupportsArray[ShapeT, np.dtype[np.floating | _to_integer]],
    rcond: float | None = None,
) -> _LstSqResult[ShapeT, np.float64, np.float64]: ...
@overload  # ~float64, +float64, known shape
def lstsq[ShapeT: tuple[int] | tuple[int, int]](
    a: _ArrayLike2D[np.floating | _to_integer] | _Sequence2D[float],
    b: _SupportsArray[ShapeT, np.dtype[_to_float64]],
    rcond: float | None = None,
) -> _LstSqResult[ShapeT, np.float64, np.float64]: ...
@overload  # +complex128, ~complex128, known shape
def lstsq[ShapeT: tuple[int] | tuple[int, int]](
    a: _ToArrayComplex_2d, b: _SupportsArray[ShapeT, np.dtype[np.complex128]], rcond: float | None = None
) -> _LstSqResult[ShapeT, np.complex128, np.float64]: ...
@overload  # ~complex128, +complex128, known shape
def lstsq[ShapeT: tuple[int] | tuple[int, int]](
    a: _AsArrayC128_2d, b: _SupportsArray[ShapeT, np.dtype[_to_complex]], rcond: float | None = None
) -> _LstSqResult[ShapeT, np.complex128, np.float64]: ...
@overload  # ~float32, ~float32, known shape
def lstsq[ShapeT: tuple[int] | tuple[int, int]](
    a: _ArrayLike2D[np.float32], b: _SupportsArray[ShapeT, np.dtype[np.float32]], rcond: float | None = None
) -> _LstSqResult[ShapeT, np.float32, np.float32]: ...
@overload  # +complex64, ~complex64, known shape
def lstsq[ShapeT: tuple[int] | tuple[int, int]](
    a: _ArrayLike2D[_inexact32], b: _SupportsArray[ShapeT, np.dtype[np.complex64]], rcond: float | None = None
) -> _LstSqResult[ShapeT, np.complex64, np.float32]: ...
@overload  # ~complex64, +complex64, known shape
def lstsq[ShapeT: tuple[int] | tuple[int, int]](
    a: _ArrayLike2D[np.complex64], b: _SupportsArray[ShapeT, np.dtype[_inexact32]], rcond: float | None = None
) -> _LstSqResult[ShapeT, np.complex64, np.float32]: ...
@overload  # +float64, +float64, unknown shape
def lstsq(
    a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, rcond: float | None = None
) -> _LstSqResult[_AnyShape, np.float64 | Any, np.float64 | Any]: ...
@overload  # +complex128, +complex128, unknown shape
def lstsq(
    a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, rcond: float | None = None
) -> _LstSqResult[_AnyShape, np.complex128 | Any, np.float64 | Any]: ...

# NOTE: This assumes that `axis` is only passed if `x` is >1d, and that `keepdims` is never passed positionally.
# keep in sync with `vector_norm`
@overload  # +inexact64 (unsafe casting), axis=None, keepdims=False
def norm(
    x: _ArrayLike[_to_inexact64_unsafe] | _NestedSequence[complex],
    ord: _OrderKind | None = None,
    axis: None = None,
    keepdims: L[False] = False,
) -> np.float64: ...
@overload  # +inexact64 (unsafe casting), axis=<given> (positional), keepdims=False
def norm(
    x: _ArrayLike[_to_inexact64_unsafe] | _NestedSequence[complex],
    ord: _OrderKind | None,
    axis: _Ax2,
    keepdims: L[False] = False,
) -> NDArray[np.float64]: ...
@overload  # +inexact64 (unsafe casting), axis=<given> (keyword), keepdims=False
def norm(
    x: _ArrayLike[_to_inexact64_unsafe] | _NestedSequence[complex],
    ord: _OrderKind | None = None,
    *,
    axis: _Ax2,
    keepdims: L[False] = False,
) -> NDArray[np.float64]: ...
@overload  # +inexact64 (unsafe casting), shape known, keepdims=True
def norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[_to_inexact64_unsafe]],
    ord: _OrderKind | None = None,
    axis: _Ax2 | None = None,
    *,
    keepdims: L[True],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # +inexact64 (unsafe casting), shape unknown, keepdims=True
def norm(
    x: _ArrayLike[_to_inexact64_unsafe] | _NestedSequence[complex],
    ord: _OrderKind | None = None,
    axis: _Ax2 | None = None,
    *,
    keepdims: L[True],
) -> NDArray[np.float64]: ...
@overload  # ~float16, axis=None, keepdims=False
def norm(
    x: _ArrayLike[np.float16], ord: _OrderKind | None = None, axis: None = None, keepdims: L[False] = False
) -> np.float16: ...
@overload  # ~float16, axis=<given> (positional), keepdims=False
def norm(x: _ArrayLike[np.float16], ord: _OrderKind | None, axis: _Ax2, keepdims: L[False] = False) -> NDArray[np.float16]: ...
@overload  # ~float16, axis=<given> (keyword), keepdims=False
def norm(
    x: _ArrayLike[np.float16], ord: _OrderKind | None = None, *, axis: _Ax2, keepdims: L[False] = False
) -> NDArray[np.float16]: ...
@overload  # ~float16, shape known, keepdims=True
def norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[np.float16]], ord: _OrderKind | None = None, axis: _Ax2 | None = None, *, keepdims: L[True]
) -> np.ndarray[ShapeT, np.dtype[np.float16]]: ...
@overload  # ~float16, shape unknown, keepdims=True
def norm(
    x: _ArrayLike[np.float16], ord: _OrderKind | None = None, axis: _Ax2 | None = None, *, keepdims: L[True]
) -> NDArray[np.float16]: ...
@overload  # ~inexact32, axis=None, keepdims=False
def norm(
    x: _ArrayLike[_inexact32], ord: _OrderKind | None = None, axis: None = None, keepdims: L[False] = False
) -> np.float32: ...
@overload  # ~inexact32, axis=<given> (positional), keepdims=False
def norm(x: _ArrayLike[_inexact32], ord: _OrderKind | None, axis: _Ax2, keepdims: L[False] = False) -> NDArray[np.float32]: ...
@overload  # ~inexact32, axis=<given> (keyword), keepdims=False
def norm(
    x: _ArrayLike[_inexact32], ord: _OrderKind | None = None, *, axis: _Ax2, keepdims: L[False] = False
) -> NDArray[np.float32]: ...
@overload  # ~inexact32, shape known, keepdims=True
def norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[_inexact32]], ord: _OrderKind | None = None, axis: _Ax2 | None = None, *, keepdims: L[True]
) -> np.ndarray[ShapeT, np.dtype[np.float32]]: ...
@overload  # ~inexact32, shape unknown, keepdims=True
def norm(
    x: _ArrayLike[_inexact32], ord: _OrderKind | None = None, axis: _Ax2 | None = None, *, keepdims: L[True]
) -> NDArray[np.float32]: ...
@overload  # ~inexact80, axis=None, keepdims=False
def norm(
    x: _ArrayLike[_inexact80], ord: _OrderKind | None = None, axis: None = None, keepdims: L[False] = False
) -> np.longdouble: ...
@overload  # ~inexact80, axis=<given> (positional), keepdims=False
def norm(x: _ArrayLike[_inexact80], ord: _OrderKind | None, axis: _Ax2, keepdims: L[False] = False) -> NDArray[np.longdouble]: ...
@overload  # ~inexact80, axis=<given> (keyword), keepdims=False
def norm(
    x: _ArrayLike[_inexact80], ord: _OrderKind | None = None, *, axis: _Ax2, keepdims: L[False] = False
) -> NDArray[np.longdouble]: ...
@overload  # ~inexact80, shape known, keepdims=True
def norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[_inexact80]], ord: _OrderKind | None = None, axis: _Ax2 | None = None, *, keepdims: L[True]
) -> np.ndarray[ShapeT, np.dtype[np.longdouble]]: ...
@overload  # ~inexact80, shape unknown, keepdims=True
def norm(
    x: _ArrayLike[_inexact80], ord: _OrderKind | None = None, axis: _Ax2 | None = None, *, keepdims: L[True]
) -> NDArray[np.longdouble]: ...
@overload  # fallback
def norm(x: ArrayLike, ord: _OrderKind | None = None, axis: _Ax2 | None = None, keepdims: bool = False) -> Any: ...

#
@overload  # +inexact64 (unsafe casting), ?d, keepdims=False
def matrix_norm(
    x: _SupportsArray[_JustAnyShape, np.dtype[_to_inexact64_unsafe]],
    /,
    *,
    ord: _OrderKind | None = "fro",
    keepdims: L[False] = False,
) -> NDArray[np.float64] | Any: ...
@overload  # +inexact64 (unsafe casting), 2d, keepdims=False
def matrix_norm(
    x: _ArrayLike2D[_to_inexact64_unsafe] | _Sequence2D[complex],
    /,
    *,
    ord: _OrderKind | None = "fro",
    keepdims: L[False] = False,
) -> np.float64: ...
@overload  # +inexact64 (unsafe casting), >2d, keepdims=False
def matrix_norm(
    x: _ArrayLike3ND[_to_inexact64_unsafe] | _Sequence3D[complex],
    /,
    *,
    ord: _OrderKind | None = "fro",
    keepdims: L[False] = False,
) -> NDArray[np.float64]: ...
@overload  # +inexact64 (unsafe casting), shape known, keepdims=True
def matrix_norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[_to_inexact64_unsafe]],
    /,
    *,
    ord: _OrderKind | None = "fro",
    keepdims: L[True],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # +inexact64 (unsafe casting), ?d, keepdims=True
def matrix_norm(
    x: _ArrayLike2ND[_to_inexact64_unsafe] | _Sequence2ND[complex], /, *, ord: _OrderKind | None = "fro", keepdims: L[True]
) -> NDArray[np.float64]: ...
@overload  # ~float16, ?d, keepdims=False
def matrix_norm(
    x: _SupportsArray[_JustAnyShape, np.dtype[np.float16]], /, *, ord: _OrderKind | None = "fro", keepdims: L[False] = False
) -> NDArray[np.float16] | Any: ...
@overload  # ~float16, 2d, keepdims=False
def matrix_norm(x: _ArrayLike2D[np.float16], /, *, ord: _OrderKind | None = "fro", keepdims: L[False] = False) -> np.float16: ...
@overload  # ~float16, >2d, keepdims=False
def matrix_norm(
    x: _ArrayLike3ND[np.float16], /, *, ord: _OrderKind | None = "fro", keepdims: L[False] = False
) -> NDArray[np.float16]: ...
@overload  # ~float16, shape known, keepdims=True
def matrix_norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[np.float16]], /, *, ord: _OrderKind | None = "fro", keepdims: L[True]
) -> np.ndarray[ShapeT, np.dtype[np.float16]]: ...
@overload  # ~float16, ?d, keepdims=True
def matrix_norm(x: _ArrayLike2ND[np.float16], /, *, ord: _OrderKind | None = "fro", keepdims: L[True]) -> NDArray[np.float16]: ...
@overload  # ~inexact32, ?d, keepdims=False
def matrix_norm(
    x: _SupportsArray[_JustAnyShape, np.dtype[_inexact32]], /, *, ord: _OrderKind | None = "fro", keepdims: L[False] = False
) -> NDArray[np.float32] | Any: ...
@overload  # ~inexact32, 2d, keepdims=False
def matrix_norm(x: _ArrayLike2D[_inexact32], /, *, ord: _OrderKind | None = "fro", keepdims: L[False] = False) -> np.float32: ...
@overload  # ~inexact32, >2d, keepdims=False
def matrix_norm(
    x: _ArrayLike3ND[_inexact32], /, *, ord: _OrderKind | None = "fro", keepdims: L[False] = False
) -> NDArray[np.float32]: ...
@overload  # ~inexact32, shape known, keepdims=True
def matrix_norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[_inexact32]], /, *, ord: _OrderKind | None = "fro", keepdims: L[True]
) -> np.ndarray[ShapeT, np.dtype[np.float32]]: ...
@overload  # ~inexact32, ?d, keepdims=True
def matrix_norm(x: _ArrayLike2ND[_inexact32], /, *, ord: _OrderKind | None = "fro", keepdims: L[True]) -> NDArray[np.float32]: ...
@overload  # ~inexact80, ?d, keepdims=False
def matrix_norm(
    x: _SupportsArray[_JustAnyShape, np.dtype[_inexact80]], /, *, ord: _OrderKind | None = "fro", keepdims: L[False] = False
) -> NDArray[np.longdouble] | Any: ...
@overload  # ~inexact80, 2d, keepdims=False
def matrix_norm(
    x: _ArrayLike2D[_inexact80], /, *, ord: _OrderKind | None = "fro", keepdims: L[False] = False
) -> np.longdouble: ...
@overload  # ~inexact80, >2d, keepdims=False
def matrix_norm(
    x: _ArrayLike3ND[_inexact80], /, *, ord: _OrderKind | None = "fro", keepdims: L[False] = False
) -> NDArray[np.longdouble]: ...
@overload  # ~inexact80, shape known, keepdims=True
def matrix_norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[_inexact80]], /, *, ord: _OrderKind | None = "fro", keepdims: L[True]
) -> np.ndarray[ShapeT, np.dtype[np.longdouble]]: ...
@overload  # ~inexact80, ?d, keepdims=True
def matrix_norm(
    x: _ArrayLike2ND[_inexact80], /, *, ord: _OrderKind | None = "fro", keepdims: L[True]
) -> NDArray[np.longdouble]: ...
@overload  # fallback
def matrix_norm(x: ArrayLike, /, *, ord: _OrderKind | None = "fro", keepdims: bool = False) -> Any: ...

# keep in sync with `norm`
@overload  # +inexact64 (unsafe casting), axis=None, keepdims=False
def vector_norm(
    x: _ArrayLike[_to_inexact64_unsafe] | _NestedSequence[complex],
    /,
    *,
    keepdims: L[False] = False,
    axis: None = None,
    ord: float | None = 2,
) -> np.float64: ...
@overload  # +inexact64 (unsafe casting), axis=<given>, keepdims=False
def vector_norm(
    x: _ArrayLike[_to_inexact64_unsafe] | _NestedSequence[complex],
    /,
    *,
    axis: _Ax2,
    keepdims: L[False] = False,
    ord: float | None = 2,
) -> NDArray[np.float64]: ...
@overload  # +inexact64 (unsafe casting), shape known, keepdims=True
def vector_norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[_to_inexact64_unsafe]],
    /,
    *,
    axis: _Ax2 | None = None,
    keepdims: L[True],
    ord: float | None = 2,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # +inexact64 (unsafe casting), shape unknown, keepdims=True
def vector_norm(
    x: _ArrayLike[_to_inexact64_unsafe] | _NestedSequence[complex],
    /,
    *,
    axis: _Ax2 | None = None,
    keepdims: L[True],
    ord: float | None = 2,
) -> NDArray[np.float64]: ...
@overload  # ~float16, axis=None, keepdims=False
def vector_norm(
    x: _ArrayLike[np.float16], /, *, axis: None = None, keepdims: L[False] = False, ord: float | None = 2
) -> np.float16: ...
@overload  # ~float16, axis=<given>  keepdims=False
def vector_norm(
    x: _ArrayLike[np.float16], /, *, axis: _Ax2, keepdims: L[False] = False, ord: float | None = 2
) -> NDArray[np.float16]: ...
@overload  # ~float16, shape known, keepdims=True
def vector_norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[np.float16]], /, *, axis: _Ax2 | None = None, keepdims: L[True], ord: float | None = 2
) -> np.ndarray[ShapeT, np.dtype[np.float16]]: ...
@overload  # ~float16, shape unknown, keepdims=True
def vector_norm(
    x: _ArrayLike[np.float16], /, *, axis: _Ax2 | None = None, keepdims: L[True], ord: float | None = 2
) -> NDArray[np.float16]: ...
@overload  # ~inexact32, axis=None, keepdims=False
def vector_norm(
    x: _ArrayLike[_inexact32], /, *, axis: None = None, keepdims: L[False] = False, ord: float | None = 2
) -> np.float32: ...
@overload  # ~inexact32, axis=<given>  keepdims=False
def vector_norm(
    x: _ArrayLike[_inexact32], /, *, axis: _Ax2, keepdims: L[False] = False, ord: float | None = 2
) -> NDArray[np.float32]: ...
@overload  # ~inexact32, shape known, keepdims=True
def vector_norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[_inexact32]], /, *, axis: _Ax2 | None = None, keepdims: L[True], ord: float | None = 2
) -> np.ndarray[ShapeT, np.dtype[np.float32]]: ...
@overload  # ~inexact32, shape unknown, keepdims=True
def vector_norm(
    x: _ArrayLike[_inexact32], /, *, axis: _Ax2 | None = None, keepdims: L[True], ord: float | None = 2
) -> NDArray[np.float32]: ...
@overload  # ~inexact80, axis=None, keepdims=False
def vector_norm(
    x: _ArrayLike[_inexact80], /, *, axis: None = None, keepdims: L[False] = False, ord: float | None = 2
) -> np.longdouble: ...
@overload  # ~inexact80, axis=<given>, keepdims=False
def vector_norm(
    x: _ArrayLike[_inexact80], /, *, axis: _Ax2, keepdims: L[False] = False, ord: float | None = 2
) -> NDArray[np.longdouble]: ...
@overload  # ~inexact80, shape known, keepdims=True
def vector_norm[ShapeT: _Shape](
    x: _SupportsArray[ShapeT, np.dtype[_inexact80]], /, *, axis: _Ax2 | None = None, keepdims: L[True], ord: float | None = 2
) -> np.ndarray[ShapeT, np.dtype[np.longdouble]]: ...
@overload  # ~inexact80, shape unknown, keepdims=True
def vector_norm(
    x: _ArrayLike[_inexact80], /, *, axis: _Ax2 | None = None, keepdims: L[True], ord: float | None = 2
) -> NDArray[np.longdouble]: ...
@overload  # fallback
def vector_norm(x: ArrayLike, /, *, axis: _Ax2 | None = None, keepdims: bool = False, ord: float | None = 2) -> Any: ...

# keep in sync with numpy._core.numeric.tensordot (ignoring `/, *`)
@overload
def tensordot[ScalarT: np.number | np.timedelta64 | np.object_](
    a: _ArrayLike[ScalarT], b: _ArrayLike[ScalarT], /, *, axes: int | tuple[_ShapeLike, _ShapeLike] = 2
) -> NDArray[ScalarT]: ...
@overload
def tensordot(
    a: _ArrayLikeBool_co, b: _ArrayLikeBool_co, /, *, axes: int | tuple[_ShapeLike, _ShapeLike] = 2
) -> NDArray[np.bool]: ...
@overload
def tensordot(
    a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, /, *, axes: int | tuple[_ShapeLike, _ShapeLike] = 2
) -> NDArray[np.int_ | Any]: ...
@overload
def tensordot(
    a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, /, *, axes: int | tuple[_ShapeLike, _ShapeLike] = 2
) -> NDArray[np.float64 | Any]: ...
@overload
def tensordot(
    a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, /, *, axes: int | tuple[_ShapeLike, _ShapeLike] = 2
) -> NDArray[np.complex128 | Any]: ...

# TODO: Returns a scalar or array
def multi_dot(
    arrays: Iterable[_ArrayLikeComplex_co | _ArrayLikeObject_co | _ArrayLikeTD64_co],
    *,
    out: NDArray[Any] | None = None,
) -> Any: ...

#
@overload  # workaround for microsoft/pyright#10232
def diagonal[DTypeT: np.dtype](
    x: _SupportsArray[_JustAnyShape, DTypeT], /, *, offset: SupportsIndex = 0
) -> np.ndarray[_AnyShape, DTypeT]: ...
@overload  # 2d, known dtype
def diagonal[DTypeT: np.dtype](
    x: _SupportsArray[tuple[int, int], DTypeT], /, *, offset: SupportsIndex = 0
) -> np.ndarray[tuple[int], DTypeT]: ...
@overload  # 3d, known dtype
def diagonal[DTypeT: np.dtype](
    x: _SupportsArray[tuple[int, int, int], DTypeT], /, *, offset: SupportsIndex = 0
) -> np.ndarray[tuple[int, int], DTypeT]: ...
@overload  # 4d, known dtype
def diagonal[DTypeT: np.dtype](
    x: _SupportsArray[tuple[int, int, int, int], DTypeT], /, *, offset: SupportsIndex = 0
) -> np.ndarray[tuple[int, int, int], DTypeT]: ...
@overload  # nd like ~bool
def diagonal(x: _NestedSequence[list[bool]], /, *, offset: SupportsIndex = 0) -> NDArray[np.bool]: ...
@overload  # nd like ~int
def diagonal(x: _NestedSequence[list[int]], /, *, offset: SupportsIndex = 0) -> NDArray[np.int_]: ...
@overload  # nd like ~float
def diagonal(x: _NestedSequence[list[float]], /, *, offset: SupportsIndex = 0) -> NDArray[np.float64]: ...
@overload  # nd like ~complex
def diagonal(x: _NestedSequence[list[complex]], /, *, offset: SupportsIndex = 0) -> NDArray[np.complex128]: ...
@overload  # nd like ~bytes
def diagonal(x: _NestedSequence[list[bytes]], /, *, offset: SupportsIndex = 0) -> NDArray[np.bytes_]: ...
@overload  # nd like ~str
def diagonal(x: _NestedSequence[list[str]], /, *, offset: SupportsIndex = 0) -> NDArray[np.str_]: ...
@overload  # fallback
def diagonal(x: ArrayLike, /, *, offset: SupportsIndex = 0) -> np.ndarray: ...

#
@overload  # workaround for microsoft/pyright#10232
def trace(
    x: _SupportsArray[_JustAnyShape, np.dtype[_to_complex]], /, *, offset: SupportsIndex = 0, dtype: DTypeLike | None = None
) -> Any: ...
@overload  # 2d known dtype, dtype=None
def trace[ScalarT: _to_complex](x: _ArrayLike2D[ScalarT], /, *, offset: SupportsIndex = 0, dtype: None = None) -> ScalarT: ...
@overload  # 2d, dtype=<given>
def trace[ScalarT: _to_complex](
    x: _ToArrayComplex_2d, /, *, offset: SupportsIndex = 0, dtype: _DTypeLike[ScalarT]
) -> ScalarT: ...
@overload  # 2d bool
def trace(x: _Sequence2D[bool], /, *, offset: SupportsIndex = 0, dtype: None = None) -> np.bool: ...
@overload  # 2d int
def trace(x: Sequence[list[int]], /, *, offset: SupportsIndex = 0, dtype: None = None) -> np.int_: ...
@overload  # 2d float
def trace(x: Sequence[list[float]], /, *, offset: SupportsIndex = 0, dtype: None = None) -> np.float64: ...
@overload  # 2d complex
def trace(x: Sequence[list[complex]], /, *, offset: SupportsIndex = 0, dtype: None = None) -> np.complex128: ...
@overload  # 3d known dtype, dtype=None
def trace[DTypeT: np.dtype[_to_complex]](
    x: _SupportsArray[tuple[int, int, int], DTypeT], /, *, offset: SupportsIndex = 0, dtype: None = None
) -> np.ndarray[tuple[int], DTypeT]: ...
@overload  # 3d, dtype=<given>
def trace[ScalarT: _to_complex](
    x: _ToArrayComplex_3d, /, *, offset: SupportsIndex = 0, dtype: _DTypeLike[ScalarT]
) -> _Array1D[ScalarT]: ...
@overload  # 3d+ known dtype, dtype=None
def trace[DTypeT: np.dtype[_to_complex]](
    x: _SupportsArray[_AtLeast3D, DTypeT], /, *, offset: SupportsIndex = 0, dtype: None = None
) -> np.ndarray[tuple[int, *tuple[Any, ...]], DTypeT]: ...
@overload  # 3d+, dtype=<given>
def trace[ScalarT: _to_complex](
    x: _ArrayLike3ND[_to_complex] | _Sequence3ND[complex], /, *, offset: SupportsIndex = 0, dtype: _DTypeLike[ScalarT]
) -> np.ndarray[tuple[int, *tuple[Any, ...]], np.dtype[ScalarT]]: ...
@overload  # 3d+ bool
def trace(x: _Sequence3ND[bool], /, *, offset: SupportsIndex = 0, dtype: None = None) -> NDArray[np.bool]: ...
@overload  # 3d+ int
def trace(x: _Sequence2ND[list[int]], /, *, offset: SupportsIndex = 0, dtype: None = None) -> NDArray[np.int_]: ...
@overload  # 3d+ float
def trace(x: _Sequence2ND[list[float]], /, *, offset: SupportsIndex = 0, dtype: None = None) -> NDArray[np.float64]: ...
@overload  # 3d+ complex
def trace(x: _Sequence2ND[list[complex]], /, *, offset: SupportsIndex = 0, dtype: None = None) -> NDArray[np.complex128]: ...
@overload  # fallback
def trace(x: _ArrayLikeComplex_co, /, *, offset: SupportsIndex = 0, dtype: DTypeLike | None = None) -> Any: ...

#
@overload  # workaround for microsoft/pyright#10232
def outer(x1: NDArray[Never], x2: NDArray[Never], /) -> _Array2D[Any]: ...
@overload  # +bool, +bool
def outer(x1: _ToArrayBool_1d, x2: _ToArrayBool_1d, /) -> _Array2D[np.bool]: ...
@overload  # ~int64, +int64
def outer(x1: _AsArrayI64_1d, x2: _ToArrayInt_1d, /) -> _Array2D[np.int64]: ...
@overload  # +int64, ~int64
def outer(x1: _ToArrayInt_1d, x2: _AsArrayI64_1d, /) -> _Array2D[np.int64]: ...
@overload  # ~timedelta64, +timedelta64
def outer(x1: _ArrayLike1D[np.timedelta64], x2: _ArrayLike1D[_to_timedelta64], /) -> _Array2D[np.timedelta64]: ...
@overload  # +timedelta64, ~timedelta64
def outer(x1: _ArrayLike1D[_to_timedelta64], x2: _ArrayLike1D[np.timedelta64], /) -> _Array2D[np.timedelta64]: ...
@overload  # ~float64, +float64
def outer(x1: _AsArrayF64_1d, x2: _ToArrayF64_1d, /) -> _Array2D[np.float64]: ...
@overload  # +float64, ~float64
def outer(x1: _ToArrayF64_1d, x2: _AsArrayF64_1d, /) -> _Array2D[np.float64]: ...
@overload  # ~complex128, +complex128
def outer(x1: _AsArrayC128_1d, x2: _ToArrayComplex_1d, /) -> _Array2D[np.complex128]: ...
@overload  # +complex128, ~complex128
def outer(x1: _ToArrayComplex_1d, x2: _AsArrayC128_1d, /) -> _Array2D[np.complex128]: ...
@overload  # ~ScalarT, ~ScalarT
def outer[ScalarT: np.number | np.object_](x1: _ArrayLike1D[ScalarT], x2: _ArrayLike1D[ScalarT], /) -> _Array2D[ScalarT]: ...
@overload  # fallback
def outer(x1: _ToArrayComplex_1d, x2: _ToArrayComplex_1d, /) -> _Array2D[Any]: ...

#
@overload  # ~T, ~T  (we use constraints instead of a `: np.number` bound to prevent joins/unions)
def cross[
    AnyScalarT: (  # int64, float64, and complex128 are handled separately
        np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.uint64,
        np.float16, np.float32, np.longdouble, np.complex64, np.clongdouble,
    ),
](
    x1: _ArrayLike1D2D[AnyScalarT],
    x2: _ArrayLike1D2D[AnyScalarT],
    /,
    *,
    axis: SupportsIndex = -1,
) -> NDArray[AnyScalarT]: ...  # fmt: skip
@overload  # ~int64, +int64
def cross(
    x1: _ArrayLike1D2D[np.int64] | _Sequence1D2D[int],
    x2: _ArrayLike1D2D[np.integer] | _Sequence1D2D[int],
    /,
    *,
    axis: SupportsIndex = -1,
) -> NDArray[np.int64]: ...
@overload  # +int64, ~int64
def cross(
    x1: _ArrayLike1D2D[np.integer],
    x2: _ArrayLike1D2D[np.int64],
    /,
    *,
    axis: SupportsIndex = -1,
) -> NDArray[np.int64]: ...
@overload  # ~float64, +float64
def cross(
    x1: _ArrayLike1D2D[np.float64] | _Sequence0D1D[list[float]],
    x2: _ArrayLike1D2D[np.floating | np.integer] | _Sequence1D2D[float],
    /,
    *,
    axis: SupportsIndex = -1,
) -> NDArray[np.float64]: ...
@overload  # +float64, ~float64
def cross(
    x1: _ArrayLike1D2D[np.floating | np.integer] | _Sequence1D2D[float],
    x2: _ArrayLike1D2D[np.float64] | _Sequence0D1D[list[float]],
    /,
    *,
    axis: SupportsIndex = -1,
) -> NDArray[np.float64]: ...
@overload  # ~complex128, +complex128
def cross(
    x1: _ArrayLike1D2D[np.complex128] | _Sequence0D1D[list[complex]],
    x2: _ArrayLike1D2D[np.number] | _Sequence1D2D[complex],
    /,
    *,
    axis: SupportsIndex = -1,
) -> NDArray[np.complex128]: ...
@overload  # +complex128, ~complex128
def cross(
    x1: _ArrayLike1D2D[np.number] | _Sequence1D2D[complex],
    x2: _ArrayLike1D2D[np.complex128] | _Sequence0D1D[list[complex]],
    /,
    *,
    axis: SupportsIndex = -1,
) -> NDArray[np.complex128]: ...
@overload  # ~object_, +object_
def cross(
    x1: _SupportsArray[tuple[int] | tuple[int, int], np.dtype[np.object_]],
    x2: _ArrayLike1D2D[np.number | np.object_] | _Sequence1D2D[complex],
    /,
    *,
    axis: SupportsIndex = -1,
) -> NDArray[np.object_]: ...
@overload  # +object_, ~object_
def cross(
    x1: _ArrayLike1D2D[np.number | np.object_] | _Sequence1D2D[complex],
    x2: _SupportsArray[tuple[int] | tuple[int, int], np.dtype[np.object_]],
    /,
    *,
    axis: SupportsIndex = -1,
) -> NDArray[np.object_]: ...
@overload  # fallback
def cross[ScalarT: np.number](
    x1: _ArrayLike1D2D[ScalarT],
    x2: _ArrayLike1D2D[ScalarT],
    /,
    *,
    axis: SupportsIndex = -1,
) -> NDArray[ScalarT]: ...

# TODO: narrow return types
@overload
def matmul[ScalarT: np.number](x1: _ArrayLike[ScalarT], x2: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload
def matmul(x1: _ArrayLikeUInt_co, x2: _ArrayLikeUInt_co, /) -> NDArray[np.unsignedinteger]: ...
@overload
def matmul(x1: _ArrayLikeInt_co, x2: _ArrayLikeInt_co, /) -> NDArray[np.signedinteger]: ...
@overload
def matmul(x1: _ArrayLikeFloat_co, x2: _ArrayLikeFloat_co, /) -> NDArray[np.floating]: ...
@overload
def matmul(x1: _ArrayLikeComplex_co, x2: _ArrayLikeComplex_co, /) -> NDArray[np.complexfloating]: ...
