import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, overload

from numpy import dtype, float32, float64, int64, integer, ndarray
from numpy.random import BitGenerator
from numpy.typing import ArrayLike, DTypeLike, _ArrayLikeFloat_co, _ArrayLikeInt_co, _ShapeLike, _DoubleCodes, _SingleCodes

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

class Generator:
    # COMPLETE
    _bit_generator: BitGenerator
    _poisson_lam_max: float
    def __init__(self, bit_generator: BitGenerator) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    # Pickling support:
    def __getstate__(self) -> Dict[str, Any]: ...
    def __setstate__(self, state: Dict[str, Any]): ...
    def __reduce__(self) -> Tuple[Callable[[str], BitGenerator], Tuple[str], Dict[str, Any]]: ...
    @property
    def bit_generator(self) -> BitGenerator: ...
    def bytes(self, length: int) -> str: ...
    # TODO: Needs overloading
    def standard_cauchy(
        self, size: Union[None, _ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    # TODO: Needs overloading and specific dtypes
    def standard_exponential(
        self,
        size: Optional[Union[_ShapeLike]] = ...,
        dtype: DTypeLike = ...,
        method: Literal["zig", "inv"] = ...,
        out=Union[None, ndarray[Any, dtype[float32]], ndarray[Any, dtype[float64]]],
    ) -> Union[float, ndarray[Any, Any]]: ...
    # TODO: Needs typing
    def random(
        self,
        size: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: Union[None, ndarray[Any, dtype[float32]], ndarray[Any, dtype[float64]]] = ...,
    ): ...
    def beta(
        self, a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ): ...
    def exponential(self, scale: _ArrayLikeFloat_co = ..., size: Union[None, _ShapeLike] = ...): ...
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: Optional[_ArrayLikeInt_co] = ...,
        size: Union[None, _ShapeLike] = ...,
        dtype=...,
        endpoint: bool = ...,
    ) -> ndarray[Any, dtype[integer]]: ...
    # TODO: Use a TypeVar _T here to get away from Any output?  Should be int->ndarray[Any,dtype[int64]], ArrayLike[_T] -> Union[_T, ndarray[Any,Any]]
    def choice(
        self,
        a: ArrayLike,
        size: Union[None, _ShapeLike] = ...,
        replace: bool = ...,
        p=...,
        axis=...,
        shuffle: bool = ...,
    ) -> Any: ...
    def uniform(
        self,
        low: _ArrayLikeFloat_co = ...,
        high: _ArrayLikeFloat_co = ...,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    @overload
    def standard_normal(
        self,
        size: None = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> float: ...
    # TODO: How to literal dtype?
    @overload
    def standard_normal(
        self,
        size: _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: Union[None, ndarray[Any, dtype[float32]], ndarray[Any, dtype[float64]]] = ...,
    ) -> Union[ndarray[Any, dtype[float32]], ndarray[Any, dtype[float64]]]: ...
    def normal(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def standard_gamma(
        self,
        shape,
        size: Union[None, _ShapeLike] = ...,
        dtype=...,
        out: Union[None, ndarray[Any, dtype[float32]], ndarray[Any, dtype[float64]]] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def gamma(
        self,
        shape: _ArrayLikeFloat_co,
        scale: _ArrayLikeFloat_co = ...,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def f(
        self,
        dfnum: _ArrayLikeFloat_co,
        dfden: _ArrayLikeFloat_co,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def noncentral_f(
        self,
        dfnum: _ArrayLikeFloat_co,
        dfden: _ArrayLikeFloat_co,
        nonc: _ArrayLikeFloat_co,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def chisquare(
        self, df: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def noncentral_chisquare(
        self, df: _ArrayLikeFloat_co, nonc: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def standard_t(
        self, df: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def vonmises(
        self, mu: _ArrayLikeFloat_co, kappa: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def pareto(
        self, a: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def weibull(
        self, a: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def power(
        self, a: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def laplace(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def gumbel(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def logistic(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def lognormal(
        self,
        mean: _ArrayLikeFloat_co = ...,
        sigma: _ArrayLikeFloat_co = ...,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def rayleigh(
        self, scale: _ArrayLikeFloat_co = ..., size: Union[None, _ShapeLike] = ...
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def wald(
        self,
        mean: _ArrayLikeFloat_co,
        scale: _ArrayLikeFloat_co,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    def triangular(
        self,
        left: _ArrayLikeFloat_co,
        mode: _ArrayLikeFloat_co,
        right: _ArrayLikeFloat_co,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[float, ndarray[Any, dtype[float64]]]: ...
    # Complicated, discrete distributions:
    def binomial(
        self, n: _ArrayLikeInt_co, p: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def negative_binomial(
        self, n: _ArrayLikeFloat_co, p: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def poisson(
        self, lam: _ArrayLikeFloat_co = ..., size: Union[None, _ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def zipf(
        self, a: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def geometric(
        self, p: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def hypergeometric(
        self,
        ngood: _ArrayLikeInt_co,
        nbad: _ArrayLikeInt_co,
        nsample: _ArrayLikeInt_co,
        size: Union[None, _ShapeLike] = ...,
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    def logseries(
        self, p: _ArrayLikeFloat_co, size: Union[None, _ShapeLike] = ...
    ) -> Union[int, ndarray[Any, dtype[int64]]]: ...
    # Multivariate distributions:
    # TODO: Really need  1-d array like floating and 2-d array-like floating. Using Sequence[float] ??
    def multivariate_normal(
        self,
        mean: Sequence[float],
        cov: Sequence[Sequence[float]],
        size: Union[None, _ShapeLike] = ...,
        check_valid: Literal["warn", "raise", "ignore"] = ...,
        tol: float = ...,
        *,
        method: Literal["svd", "eigh", "cholesky"] = ...
    ): ...
    # TODO: Need 1-d array like floating. Using Sequence[float] ??
    def multinomial(
        self, n: _ArrayLikeInt_co, pvals: Sequence[float], size: Union[None, _ShapeLike] = ...
    ): ...
    # TODO: Need 1-d array like integers. Using Sequence[int] ??
    def multivariate_hypergeometric(
        self,
        colors: Sequence[int],
        nsample: int,
        size: Union[None, _ShapeLike] = ...,
        method: Literal["marginals", "count"] = ...,
    ): ...
    # TODO: Need 1-d array like floating. Using Sequence[float] ??
    def dirichlet(
        self, alpha: Sequence[float], size: Union[None, _ShapeLike] = ...
    ) -> ndarray[Any, dtype[float64]]: ...
    def permuted(
        self, x: ArrayLike, *, axis: Optional[int] = ..., out: Optional[ndarray[Any, Any]] = ...
    ) -> ndarray[Any, Any]: ...
    def shuffle(self, x: ArrayLike, axis: int = ...) -> Sequence[Any]: ...
    @overload
    def permutation(self, x: int, axis: int = ...) -> ndarray[Any, dtype[int64]]: ...
    @overload
    def permutation(self, x: ArrayLike, axis: int = ...) -> ndarray[Any, Any]: ...

def default_rng(seed=None) -> Generator: ...
