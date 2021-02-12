import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union, overload

from numpy import double, dtype, float32, float64, int64, integer, ndarray, single
from numpy.random import BitGenerator, SeedSequence
from numpy.typing import (
    ArrayLike,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _DoubleCodes,
    _DTypeLikeBool,
    _DTypeLikeInt,
    _DTypeLikeUInt,
    _Float32Codes,
    _Float64Codes,
    _ShapeLike,
    _SingleCodes,
    _SupportsDType,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_DTypeLikeFloat32 = Union[
    dtype[single],
    dtype[float32],
    _SupportsDType[dtype[single]],
    _SupportsDType[dtype[float32]],
    Type[float32],
    Type[single],
    _Float32Codes,
    _SingleCodes,
]

_DTypeLikeFloat64 = Union[
    dtype[double],
    dtype[float64],
    _SupportsDType[dtype[double]],
    _SupportsDType[dtype[float64]],
    Type[double],
    Type[float],
    Type[float64],
    _Float64Codes,
    _DoubleCodes,
]

class Generator:
    # COMPLETE
    def __init__(self, bit_generator: BitGenerator) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    # Pickling support:
    def __getstate__(self) -> Dict[str, Any]: ...
    def __setstate__(self, state: Dict[str, Any]) -> None: ...
    def __reduce__(self) -> Tuple[Callable[[str], BitGenerator], Tuple[str], Dict[str, Any]]: ...
    @property
    def bit_generator(self) -> BitGenerator: ...
    def bytes(self, length: int) -> str: ...
    @overload
    def standard_normal(  # type: ignore[misc]
        self,
        size: None = ...,
        dtype: Union[_DTypeLikeFloat32, _DTypeLikeFloat64] = ...,
        out: None = ...,
    ) -> float: ...
    @overload
    def standard_normal(  # type: ignore[misc]
        self,
        size: _ShapeLike = ...,
        dtype: Union[_DTypeLikeFloat32, _DTypeLikeFloat64] = ...,
        out: Optional[ndarray[Any, dtype[Union[float32, float64]]]] = ...,
    ) -> ndarray[Any, dtype[Union[float32, float64]]]: ...
    @overload
    def permutation(self, x: int, axis: int = ...) -> ndarray[Any, dtype[int64]]: ...
    @overload
    def permutation(self, x: ArrayLike, axis: int = ...) -> ndarray[Any, Any]: ...
    # TODO: Need overloading
    def standard_cauchy(
        self, size: Optional[_ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def standard_exponential(
        self,
        size: Optional[_ShapeLike] = ...,
        dtype: Union[_DTypeLikeFloat32, _DTypeLikeFloat64] = ...,
        method: Literal["zig", "inv"] = ...,
        out: Optional[ndarray[Any, dtype[Union[float32, float64]]]] = ...,
    ) -> Union[float, ndarray[Any, Any]]: ...
    def random(
        self,
        size: Optional[_ShapeLike] = ...,
        dtype: Union[_DTypeLikeFloat32, _DTypeLikeFloat64] = ...,
        out: Optional[ndarray[Any, dtype[Union[float32, float64]]]] = ...,
    ): ...
    def beta(
        self, a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ): ...
    def exponential(self, scale: _ArrayLikeFloat_co = ..., size: Optional[_ShapeLike] = ...): ...
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: Optional[_ArrayLikeInt_co] = ...,
        size: Optional[_ShapeLike] = ...,
        dtype: Union[_DTypeLikeBool, _DTypeLikeInt, _DTypeLikeUInt] = ...,
        endpoint: bool = ...,
    ) -> ndarray[Any, dtype[integer]]: ...
    # TODO: Use a TypeVar _T here to get away from Any output?  Should be int->ndarray[Any,dtype[int64]], ArrayLike[_T] -> Union[_T, ndarray[Any,Any]]
    def choice(
        self,
        a: ArrayLike,
        size: Optional[_ShapeLike] = ...,
        replace: bool = ...,
        p: Optional[_ArrayLikeFloat_co] = ...,
        axis: Optional[int] = ...,
        shuffle: bool = ...,
    ) -> Any: ...
    def uniform(
        self,
        low: _ArrayLikeFloat_co = ...,
        high: _ArrayLikeFloat_co = ...,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def normal(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def standard_gamma(
        self,
        shape,
        size: Optional[_ShapeLike] = ...,
        dtype: Union[_DTypeLikeFloat32, _DTypeLikeFloat64] = ...,
        out: Optional[ndarray[Any, dtype[Union[float32, float64]]]] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def gamma(
        self,
        shape: _ArrayLikeFloat_co,
        scale: _ArrayLikeFloat_co = ...,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def f(
        self,
        dfnum: _ArrayLikeFloat_co,
        dfden: _ArrayLikeFloat_co,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def noncentral_f(
        self,
        dfnum: _ArrayLikeFloat_co,
        dfden: _ArrayLikeFloat_co,
        nonc: _ArrayLikeFloat_co,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def chisquare(
        self, df: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def noncentral_chisquare(
        self, df: _ArrayLikeFloat_co, nonc: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def standard_t(
        self, df: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def vonmises(
        self, mu: _ArrayLikeFloat_co, kappa: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def pareto(
        self, a: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def weibull(
        self, a: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def power(
        self, a: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def laplace(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def gumbel(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def logistic(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def lognormal(
        self,
        mean: _ArrayLikeFloat_co = ...,
        sigma: _ArrayLikeFloat_co = ...,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def rayleigh(
        self, scale: _ArrayLikeFloat_co = ..., size: Optional[_ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def wald(
        self,
        mean: _ArrayLikeFloat_co,
        scale: _ArrayLikeFloat_co,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def triangular(
        self,
        left: _ArrayLikeFloat_co,
        mode: _ArrayLikeFloat_co,
        right: _ArrayLikeFloat_co,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    # Complicated, discrete distributions:
    def binomial(
        self, n: _ArrayLikeInt_co, p: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def negative_binomial(
        self, n: _ArrayLikeFloat_co, p: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def poisson(
        self, lam: _ArrayLikeFloat_co = ..., size: Optional[_ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def zipf(
        self, a: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def geometric(
        self, p: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def hypergeometric(
        self,
        ngood: _ArrayLikeInt_co,
        nbad: _ArrayLikeInt_co,
        nsample: _ArrayLikeInt_co,
        size: Optional[_ShapeLike] = ...,
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def logseries(
        self, p: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    # Multivariate distributions:
    def multivariate_normal(
        self,
        mean: _ArrayLikeFloat_co,
        cov: _ArrayLikeFloat_co,
        size: Optional[_ShapeLike] = ...,
        check_valid: Literal["warn", "raise", "ignore"] = ...,
        tol: float = ...,
        *,
        method: Literal["svd", "eigh", "cholesky"] = ...
    ): ...
    def multinomial(
        self, n: _ArrayLikeInt_co, pvals: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ): ...
    def multivariate_hypergeometric(
        self,
        colors: _ArrayLikeInt_co,
        nsample: int,
        size: Optional[_ShapeLike] = ...,
        method: Literal["marginals", "count"] = ...,
    ): ...
    def dirichlet(
        self, alpha: _ArrayLikeFloat_co, size: Optional[_ShapeLike] = ...
    ) -> ndarray[Any, dtype[float64]]: ...
    def permuted(
        self, x: ArrayLike, *, axis: Optional[int] = ..., out: Optional[ndarray[Any, Any]] = ...
    ) -> ndarray[Any, Any]: ...
    def shuffle(self, x: ArrayLike, axis: int = ...) -> Sequence[Any]: ...

def default_rng(seed: Union[None, _ArrayLikeInt_co, SeedSequence] = ...) -> Generator: ...
