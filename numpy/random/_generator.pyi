from collections.abc import Callable, MutableSequence
from typing import Any, Literal, Self, overload

import numpy as np
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _DTypeLike,
    _Float32Codes,
    _Float64Codes,
    _FloatLike_co,
    _Int64Codes,
    _NestedSequence,
    _ShapeLike,
)

from .bit_generator import BitGenerator, SeedSequence
from .mtrand import RandomState

type _ArrayF32 = NDArray[np.float32]
type _ArrayF64 = NDArray[np.float64]

type _DTypeLikeI64 = _DTypeLike[np.int64] | _Int64Codes
type _DTypeLikeF32 = _DTypeLike[np.float32] | _Float32Codes
type _DTypeLikeF64 = type[float] | _DTypeLike[np.float64] | _Float64Codes
# we use `str` to avoid type-checker performance issues because of the many `Literal` variants
type _DTypeLikeFloat = type[float] | _DTypeLike[np.float32 | np.float64] | str

# Similar to `_ArrayLike{}_co`, but rejects scalars
type _NDArrayLikeInt = NDArray[np.generic[int]] | _NestedSequence[int]
type _NDArrayLikeFloat = NDArray[np.generic[float]] | _NestedSequence[float]

type _MethodExp = Literal["zig", "inv"]

###

class Generator:
    def __init__(self, bit_generator: BitGenerator) -> None: ...
    def __setstate__(self, state: dict[str, Any] | None) -> None: ...
    def __reduce__(self) -> tuple[Callable[[BitGenerator], Generator], tuple[BitGenerator], None]: ...

    #
    @property
    def bit_generator(self) -> BitGenerator: ...
    def spawn(self, n_children: int) -> list[Self]: ...
    def bytes(self, length: int) -> bytes: ...

    # continuous distributions

    #
    @overload
    def standard_cauchy(self, size: None = None) -> float: ...
    @overload
    def standard_cauchy(self, size: _ShapeLike) -> _ArrayF64: ...

    #
    @overload  # size=None (default);  NOTE: dtype is ignored
    def random(self, size: None = None, dtype: _DTypeLikeFloat = ..., out: None = None) -> float: ...
    @overload  # size=<given>, dtype=f64 (default)
    def random(self, size: _ShapeLike, dtype: _DTypeLikeF64 = ..., out: None = None) -> _ArrayF64: ...
    @overload  # size=<given>, dtype=f32
    def random(self, size: _ShapeLike, dtype: _DTypeLikeF32, out: None = None) -> _ArrayF32: ...
    @overload  # out: f64 array  (keyword)
    def random[ArrayT: _ArrayF64](self, size: _ShapeLike | None = None, dtype: _DTypeLikeF64 = ..., *, out: ArrayT) -> ArrayT: ...
    @overload  # dtype: f32 (keyword), out: f64 array
    def random[ArrayT: _ArrayF32](self, size: _ShapeLike | None = None, *, dtype: _DTypeLikeF32, out: ArrayT) -> ArrayT: ...
    @overload  # out: f64 array  (positional)
    def random[ArrayT: _ArrayF64](self, size: _ShapeLike | None, dtype: _DTypeLikeF64, out: ArrayT) -> ArrayT: ...
    @overload  # dtype: f32 (positional), out: f32 array
    def random[ArrayT: _ArrayF32](self, size: _ShapeLike | None, dtype: _DTypeLikeF32, out: ArrayT) -> ArrayT: ...

    #
    @overload  # size=None (default);  NOTE: dtype is ignored
    def standard_normal(self, size: None = None, dtype: _DTypeLikeFloat = ..., out: None = None) -> float: ...
    @overload  # size=<given>, dtype: f64 (default)
    def standard_normal(self, size: _ShapeLike, dtype: _DTypeLikeF64 = ..., out: None = None) -> _ArrayF64: ...
    @overload  # size=<given>, dtype: f32
    def standard_normal(self, size: _ShapeLike, dtype: _DTypeLikeF32, *, out: None = None) -> _ArrayF32: ...
    @overload  # dtype: f64 (default), out: f64 array (keyword)
    def standard_normal[ArrayT: _ArrayF64](
        self, size: _ShapeLike | None = None, dtype: _DTypeLikeF64 = ..., *, out: ArrayT
    ) -> ArrayT: ...
    @overload  # dtype: f32 (keyword), out: f32 array
    def standard_normal[ArrayT: _ArrayF32](
        self, size: _ShapeLike | None = None, *, dtype: _DTypeLikeF32, out: ArrayT
    ) -> ArrayT: ...
    @overload  # dtype: f32 (positional), out: f32 array
    def standard_normal[ArrayT: _ArrayF32](self, size: _ShapeLike | None, dtype: _DTypeLikeF32, out: ArrayT) -> ArrayT: ...

    #
    @overload  # size=None (default);  NOTE: dtype is ignored
    def standard_exponential(
        self, size: None = None, dtype: _DTypeLikeFloat = ..., method: _MethodExp = "zig", out: None = None
    ) -> float: ...
    @overload  # size=<given>, dtype: f64 (default)
    def standard_exponential(
        self, size: _ShapeLike, dtype: _DTypeLikeF64 = ..., method: _MethodExp = "zig", out: None = None
    ) -> _ArrayF64: ...
    @overload  # size=<given>, dtype: f32 (default)
    def standard_exponential(
        self, size: _ShapeLike, dtype: _DTypeLikeF32, method: _MethodExp = "zig", out: None = None
    ) -> _ArrayF32: ...
    @overload  # dtype: f64 (default), out: f64 array (keyword)
    def standard_exponential[ArrayT: _ArrayF64](
        self, size: _ShapeLike | None = None, dtype: _DTypeLikeF64 = ..., method: _MethodExp = "zig", *, out: ArrayT
    ) -> ArrayT: ...
    @overload  # dtype: f32 (keyword), out: f32 array
    def standard_exponential[ArrayT: _ArrayF32](
        self, size: _ShapeLike | None = None, *, dtype: _DTypeLikeF32, method: _MethodExp = "zig", out: ArrayT
    ) -> ArrayT: ...
    @overload  # dtype: f32 (positional), out: f32 array (keyword)
    def standard_exponential[ArrayT: _ArrayF32](
        self, size: _ShapeLike | None, dtype: _DTypeLikeF32, method: _MethodExp = "zig", *, out: ArrayT
    ) -> ArrayT: ...

    #
    @overload  # 0d, size=None (default);  NOTE: dtype is ignored
    def standard_gamma(
        self, shape: _FloatLike_co, size: None = None, dtype: _DTypeLikeFloat = ..., out: None = None
    ) -> float: ...
    @overload  # >0d, dtype: f64 (default)
    def standard_gamma(
        self, shape: _NDArrayLikeFloat, size: None = None, dtype: _DTypeLikeF64 = ..., out: None = None
    ) -> _ArrayF64: ...
    @overload  # >0d, dtype: f32 (keyword)
    def standard_gamma(
        self, shape: _NDArrayLikeFloat, size: None = None, *, dtype: _DTypeLikeF32, out: None = None
    ) -> _ArrayF32: ...
    @overload  # >=0d, dtype: f64 (default)
    def standard_gamma(
        self, shape: _ArrayLikeFloat_co, size: None = None, dtype: _DTypeLikeF64 = ..., out: None = None
    ) -> _ArrayF64 | Any: ...
    @overload  # >=0d, dtype: f32 (keyword)
    def standard_gamma(
        self, shape: _ArrayLikeFloat_co, size: None = None, *, dtype: _DTypeLikeF32, out: None = None
    ) -> _ArrayF32 | Any: ...
    @overload  # >=0d, size=<given>, dtype: f64 (default)
    def standard_gamma(
        self, shape: _ArrayLikeFloat_co, size: _ShapeLike, dtype: _DTypeLikeF64 = ..., out: None = None
    ) -> _ArrayF64: ...
    @overload  # >=0d, size=<given>, dtype: f32
    def standard_gamma(
        self, shape: _ArrayLikeFloat_co, size: _ShapeLike, dtype: _DTypeLikeF32, *, out: None = None
    ) -> _ArrayF32: ...
    @overload  # >=0d, dtype: f64 (default), out: f64 array (keyword)
    def standard_gamma[ArrayT: _ArrayF64](
        self, shape: _ArrayLikeFloat_co, size: _ShapeLike | None = None, dtype: _DTypeLikeF64 = ..., *, out: ArrayT
    ) -> ArrayT: ...
    @overload  # >=0d, dtype: f32 (keyword), out: f32 array
    def standard_gamma[ArrayT: _ArrayF32](
        self, shape: _ArrayLikeFloat_co, size: _ShapeLike | None = None, *, dtype: _DTypeLikeF32, out: ArrayT
    ) -> ArrayT: ...
    @overload  # >=0d, dtype: f32 (positional), out: f32 array
    def standard_gamma[ArrayT: _ArrayF32](
        self, shape: _ArrayLikeFloat_co, size: _ShapeLike | None, dtype: _DTypeLikeF32, out: ArrayT
    ) -> ArrayT: ...

    #
    @overload  # 0d
    def power(self, /, a: _FloatLike_co, size: None = None) -> float: ...
    @overload  # size=<given>
    def power(self, /, a: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >0d
    def power(self, /, a: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d
    def power(self, /, a: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d
    def pareto(self, /, a: _FloatLike_co, size: None = None) -> float: ...
    @overload  # size=<given>
    def pareto(self, /, a: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >0d
    def pareto(self, /, a: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d
    def pareto(self, /, a: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d
    def weibull(self, /, a: _FloatLike_co, size: None = None) -> float: ...
    @overload  # size=<given>
    def weibull(self, /, a: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >0d
    def weibull(self, /, a: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d
    def weibull(self, /, a: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d
    def standard_t(self, /, df: _FloatLike_co, size: None = None) -> float: ...
    @overload  # size=<given>
    def standard_t(self, /, df: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >0d
    def standard_t(self, /, df: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d
    def standard_t(self, /, df: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d
    def chisquare(self, /, df: _FloatLike_co, size: None = None) -> float: ...
    @overload  # size=<given>
    def chisquare(self, /, df: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >0d
    def chisquare(self, /, df: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d
    def chisquare(self, /, df: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d (default)
    def exponential(self, /, scale: _FloatLike_co = 1.0, size: None = None) -> float: ...
    @overload  # size=<given> (keyword)
    def exponential(self, /, scale: _ArrayLikeFloat_co = 1.0, *, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # size=<given> (positional)
    def exponential(self, /, scale: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >0d
    def exponential(self, /, scale: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d
    def exponential(self, /, scale: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d (default)
    def rayleigh(self, /, scale: _FloatLike_co = 1.0, size: None = None) -> float: ...
    @overload  # size=<given> (keyword)
    def rayleigh(self, /, scale: _ArrayLikeFloat_co = 1.0, *, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # size=<given> (positional)
    def rayleigh(self, /, scale: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >0d
    def rayleigh(self, /, scale: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d
    def rayleigh(self, /, scale: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d, 0d
    def noncentral_chisquare(self, /, df: _FloatLike_co, nonc: _FloatLike_co, size: None = None) -> float: ...
    @overload  # size=<given>
    def noncentral_chisquare(self, /, df: _ArrayLikeFloat_co, nonc: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d
    def noncentral_chisquare(self, /, df: _ArrayLikeFloat_co, nonc: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def noncentral_chisquare(self, /, df: _NDArrayLikeFloat, nonc: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d
    def noncentral_chisquare(self, /, df: _ArrayLikeFloat_co, nonc: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d, 0d
    def f(self, /, dfnum: _FloatLike_co, dfden: _FloatLike_co, size: None = None) -> float: ...
    @overload  # size=<given>
    def f(self, /, dfnum: _ArrayLikeFloat_co, dfden: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d
    def f(self, /, dfnum: _ArrayLikeFloat_co, dfden: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def f(self, /, dfnum: _NDArrayLikeFloat, dfden: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d (fallback)
    def f(self, /, dfnum: _ArrayLikeFloat_co, dfden: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d, 0d
    def vonmises(self, /, mu: _FloatLike_co, kappa: _FloatLike_co, size: None = None) -> float: ...
    @overload  # size=<given>
    def vonmises(self, /, mu: _ArrayLikeFloat_co, kappa: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d
    def vonmises(self, /, mu: _ArrayLikeFloat_co, kappa: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def vonmises(self, /, mu: _NDArrayLikeFloat, kappa: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d (fallback)
    def vonmises(self, /, mu: _ArrayLikeFloat_co, kappa: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d, 0d
    def wald(self, /, mean: _FloatLike_co, scale: _FloatLike_co, size: None = None) -> float: ...
    @overload  # size=<given>
    def wald(self, /, mean: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d
    def wald(self, /, mean: _ArrayLikeFloat_co, scale: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def wald(self, /, mean: _NDArrayLikeFloat, scale: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d (fallback)
    def wald(self, /, mean: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d, 0d
    def beta(self, /, a: _FloatLike_co, b: _FloatLike_co, size: None = None) -> float: ...
    @overload  # size=<given>
    def beta(self, /, a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d
    def beta(self, /, a: _ArrayLikeFloat_co, b: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def beta(self, /, a: _NDArrayLikeFloat, b: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d (fallback)
    def beta(self, /, a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d, 0d (default)
    def gamma(self, /, shape: _FloatLike_co, scale: _FloatLike_co = 1.0, size: None = None) -> float: ...
    @overload  # size=<given> (positional)
    def gamma(self, /, shape: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # size=<given> (keyword)
    def gamma(self, /, shape: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co = 1.0, *, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d
    def gamma(self, /, shape: _ArrayLikeFloat_co, scale: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def gamma(self, /, shape: _NDArrayLikeFloat, scale: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d (fallback)
    def gamma(self, /, shape: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d (default), 0d (default)
    def uniform(self, /, low: _FloatLike_co = 0.0, high: _FloatLike_co = 1.0, size: None = None) -> float: ...
    @overload  # >=0d, >=0d, size=<given> (positional)
    def uniform(self, /, low: _ArrayLikeFloat_co, high: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (positional)
    def uniform(self, /, low: _ArrayLikeFloat_co, high: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d, size=<given> (keyword)
    def uniform(self, /, low: _ArrayLikeFloat_co = 0.0, high: _ArrayLikeFloat_co = 1.0, *, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (keyword)
    def uniform(self, /, low: _ArrayLikeFloat_co = 0.0, *, high: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def uniform(self, /, low: _NDArrayLikeFloat, high: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d (fallback)
    def uniform(self, /, low: _ArrayLikeFloat_co = 0.0, high: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d (default), 0d (default)
    def normal(self, /, loc: _FloatLike_co = 0.0, scale: _FloatLike_co = 1.0, size: None = None) -> float: ...
    @overload  # size=<given> (positional)
    def normal(self, /, loc: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (positional)
    def normal(self, /, loc: _ArrayLikeFloat_co, scale: _NDArrayLikeFloat, size: None) -> _ArrayF64: ...
    @overload  # size=<given> (keyword)
    def normal(self, /, loc: _ArrayLikeFloat_co = 0.0, scale: _ArrayLikeFloat_co = 1.0, *, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (keyword)
    def normal(self, /, loc: _ArrayLikeFloat_co = 0.0, *, scale: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def normal(self, /, loc: _NDArrayLikeFloat, scale: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d
    def normal(self, /, loc: _ArrayLikeFloat_co = 0.0, scale: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d (default), 0d (default)
    def gumbel(self, /, loc: _FloatLike_co = 0.0, scale: _FloatLike_co = 1.0, size: None = None) -> float: ...
    @overload  # size=<given> (positional)
    def gumbel(self, /, loc: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (positional)
    def gumbel(self, /, loc: _ArrayLikeFloat_co, scale: _NDArrayLikeFloat, size: None) -> _ArrayF64: ...
    @overload  # size=<given> (keyword)
    def gumbel(self, /, loc: _ArrayLikeFloat_co = 0.0, scale: _ArrayLikeFloat_co = 1.0, *, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (keyword)
    def gumbel(self, /, loc: _ArrayLikeFloat_co = 0.0, *, scale: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def gumbel(self, /, loc: _NDArrayLikeFloat, scale: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d
    def gumbel(self, /, loc: _ArrayLikeFloat_co = 0.0, scale: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d (default), 0d (default)
    def logistic(self, /, loc: _FloatLike_co = 0.0, scale: _FloatLike_co = 1.0, size: None = None) -> float: ...
    @overload  # size=<given> (positional)
    def logistic(self, /, loc: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (positional)
    def logistic(self, /, loc: _ArrayLikeFloat_co, scale: _NDArrayLikeFloat, size: None) -> _ArrayF64: ...
    @overload  # size=<given> (keyword)
    def logistic(self, /, loc: _ArrayLikeFloat_co = 0.0, scale: _ArrayLikeFloat_co = 1.0, *, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (keyword)
    def logistic(self, /, loc: _ArrayLikeFloat_co = 0.0, *, scale: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def logistic(self, /, loc: _NDArrayLikeFloat, scale: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d
    def logistic(
        self, /, loc: _ArrayLikeFloat_co = 0.0, scale: _ArrayLikeFloat_co = 1.0, size: None = None
    ) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d (default), 0d (default)
    def laplace(self, /, loc: _FloatLike_co = 0.0, scale: _FloatLike_co = 1.0, size: None = None) -> float: ...
    @overload  # size=<given> (positional)
    def laplace(self, /, loc: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (positional)
    def laplace(self, /, loc: _ArrayLikeFloat_co, scale: _NDArrayLikeFloat, size: None) -> _ArrayF64: ...
    @overload  # size=<given> (keyword)
    def laplace(self, /, loc: _ArrayLikeFloat_co = 0.0, scale: _ArrayLikeFloat_co = 1.0, *, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (keyword)
    def laplace(self, /, loc: _ArrayLikeFloat_co = 0.0, *, scale: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def laplace(self, /, loc: _NDArrayLikeFloat, scale: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d
    def laplace(
        self, /, loc: _ArrayLikeFloat_co = 0.0, scale: _ArrayLikeFloat_co = 1.0, size: None = None
    ) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d (default), 0d (default)
    def lognormal(self, /, mean: _FloatLike_co = 0.0, sigma: _FloatLike_co = 1.0, size: None = None) -> float: ...
    @overload  # size=<given> (positional)
    def lognormal(self, /, mean: _ArrayLikeFloat_co, sigma: _ArrayLikeFloat_co, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (positional)
    def lognormal(self, /, mean: _ArrayLikeFloat_co, sigma: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # size=<given> (keyword)
    def lognormal(self, /, mean: _ArrayLikeFloat_co = 0.0, sigma: _ArrayLikeFloat_co = 1.0, *, size: _ShapeLike) -> _ArrayF64: ...
    @overload  # >=0d, >0d (keyword)
    def lognormal(self, /, mean: _ArrayLikeFloat_co = 0.0, *, sigma: _NDArrayLikeFloat, size: None = None) -> _ArrayF64: ...
    @overload  # >0d, >=0d
    def lognormal(self, /, mean: _NDArrayLikeFloat, sigma: _ArrayLikeFloat_co = 1.0, size: None = None) -> _ArrayF64: ...
    @overload  # >=0d, >=0d
    def lognormal(
        self, /, mean: _ArrayLikeFloat_co = 0.0, sigma: _ArrayLikeFloat_co = 1.0, size: None = None
    ) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d, 0d, 0d
    def triangular(self, /, left: _FloatLike_co, mode: _FloatLike_co, right: _FloatLike_co, size: None = None) -> float: ...
    @overload  # >=0d, >=0d, >=0d, size=<given>
    def triangular(
        self, /, left: _ArrayLikeFloat_co, mode: _ArrayLikeFloat_co, right: _ArrayLikeFloat_co, size: _ShapeLike
    ) -> _ArrayF64: ...
    @overload  # >=0d, >=0d, >0d
    def triangular(
        self, /, left: _ArrayLikeFloat_co, mode: _ArrayLikeFloat_co, right: _NDArrayLikeFloat, size: None = None
    ) -> _ArrayF64: ...
    @overload  # >=0d, >0d, >=0d
    def triangular(
        self, /, left: _ArrayLikeFloat_co, mode: _NDArrayLikeFloat, right: _ArrayLikeFloat_co, size: None = None
    ) -> _ArrayF64: ...
    @overload  # >0d, >=0d, >=0d
    def triangular(
        self, /, left: _NDArrayLikeFloat, mode: _ArrayLikeFloat_co, right: _ArrayLikeFloat_co, size: None = None
    ) -> _ArrayF64: ...
    @overload  # >=0d, >=0d, >=0d (fallback)
    def triangular(
        self, /, left: _ArrayLikeFloat_co, mode: _ArrayLikeFloat_co, right: _ArrayLikeFloat_co, size: None = None
    ) -> _ArrayF64 | Any: ...

    #
    @overload  # 0d, 0d, 0d
    def noncentral_f(self, /, dfnum: _FloatLike_co, dfden: _FloatLike_co, nonc: _FloatLike_co, size: None = None) -> float: ...
    @overload  # >=0d, >=0d, >=0d, size=<given>
    def noncentral_f(
        self, /, dfnum: _ArrayLikeFloat_co, dfden: _ArrayLikeFloat_co, nonc: _ArrayLikeFloat_co, size: _ShapeLike
    ) -> _ArrayF64: ...
    @overload  # >=0d, >=0d, >0d
    def noncentral_f(
        self, /, dfnum: _ArrayLikeFloat_co, dfden: _ArrayLikeFloat_co, nonc: _NDArrayLikeFloat, size: None = None
    ) -> _ArrayF64: ...
    @overload  # >=0d, >0d, >=0d
    def noncentral_f(
        self, /, dfnum: _ArrayLikeFloat_co, dfden: _NDArrayLikeFloat, nonc: _ArrayLikeFloat_co, size: None = None
    ) -> _ArrayF64: ...
    @overload  # >0d, >=0d, >=0d
    def noncentral_f(
        self, /, dfnum: _NDArrayLikeFloat, dfden: _ArrayLikeFloat_co, nonc: _ArrayLikeFloat_co, size: None = None
    ) -> _ArrayF64: ...
    @overload  # >=0d, >=0d, >=0d (fallback)
    def noncentral_f(
        self, /, dfnum: _ArrayLikeFloat_co, dfden: _ArrayLikeFloat_co, nonc: _ArrayLikeFloat_co, size: None = None
    ) -> _ArrayF64 | Any: ...

    ###
    # discrete

    #
    @overload  # 0d bool | int
    def integers[AnyIntT: (bool, int)](
        self, low: int, high: int | None = None, size: None = None, *, dtype: type[AnyIntT], endpoint: bool = False
    ) -> AnyIntT: ...
    @overload  # 0d integer dtype
    def integers[ScalarT: np.integer | np.bool](
        self, low: int, high: int | None = None, size: None = None, *, dtype: _DTypeLike[ScalarT], endpoint: bool = False
    ) -> ScalarT: ...
    @overload  # 0d int64 (default)
    def integers(
        self, low: int, high: int | None = None, size: None = None, dtype: _DTypeLikeI64 = ..., endpoint: bool = False
    ) -> np.int64: ...
    @overload  # 0d unknown
    def integers(
        self, low: int, high: int | None = None, size: None = None, dtype: DTypeLike | None = ..., endpoint: bool = False
    ) -> Any: ...
    @overload  # integer dtype, size=<given>
    def integers[ScalarT: np.integer | np.bool](
        self,
        low: _ArrayLikeInt_co,
        high: _ArrayLikeInt_co | None = None,
        *,
        size: _ShapeLike,
        dtype: _DTypeLike[ScalarT],
        endpoint: bool = False,
    ) -> NDArray[ScalarT]: ...
    @overload  # int64 (default), size=<given>
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: _ArrayLikeInt_co | None = None,
        *,
        size: _ShapeLike,
        dtype: _DTypeLikeI64 = ...,
        endpoint: bool = False,
    ) -> NDArray[np.int64]: ...
    @overload  # unknown, size=<given>
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: _ArrayLikeInt_co | None = None,
        *,
        size: _ShapeLike,
        dtype: DTypeLike | None = ...,
        endpoint: bool = False,
    ) -> np.ndarray: ...
    @overload  # >=0d, integer dtype
    def integers[ScalarT: np.integer | np.bool](
        self,
        low: _ArrayLikeInt_co,
        high: _ArrayLikeInt_co | None = None,
        size: _ShapeLike | None = None,
        *,
        dtype: _DTypeLike[ScalarT],
        endpoint: bool = False,
    ) -> NDArray[ScalarT] | Any: ...
    @overload  # >=0d, int64 (default)
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: _ArrayLikeInt_co | None = None,
        size: _ShapeLike | None = None,
        dtype: _DTypeLikeI64 = ...,
        endpoint: bool = False,
    ) -> NDArray[np.int64] | Any: ...
    @overload  # >=0d, unknown
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: _ArrayLikeInt_co | None = None,
        size: _ShapeLike | None = None,
        dtype: DTypeLike | None = ...,
        endpoint: bool = False,
    ) -> np.ndarray | Any: ...

    #
    @overload  # 0d
    def zipf(self, /, a: _FloatLike_co, size: None = None) -> int: ...
    @overload  # size=<given>
    def zipf(self, /, a: _ArrayLikeFloat_co, size: _ShapeLike) -> NDArray[np.int64]: ...
    @overload  # >0d
    def zipf(self, /, a: _NDArrayLikeFloat, size: None = None) -> NDArray[np.int64]: ...
    @overload  # >=0d
    def zipf(self, /, a: _ArrayLikeFloat_co, size: None = None) -> NDArray[np.int64] | Any: ...

    #
    @overload  # 0d
    def geometric(self, /, p: _FloatLike_co, size: None = None) -> int: ...
    @overload  # size=<given>
    def geometric(self, /, p: _ArrayLikeFloat_co, size: _ShapeLike) -> NDArray[np.int64]: ...
    @overload  # >0d
    def geometric(self, /, p: _NDArrayLikeFloat, size: None = None) -> NDArray[np.int64]: ...
    @overload  # >=0d
    def geometric(self, /, p: _ArrayLikeFloat_co, size: None = None) -> NDArray[np.int64] | Any: ...

    #
    @overload  # 0d
    def logseries(self, /, p: _FloatLike_co, size: None = None) -> int: ...
    @overload  # size=<given>
    def logseries(self, /, p: _ArrayLikeFloat_co, size: _ShapeLike) -> NDArray[np.int64]: ...
    @overload  # >0d
    def logseries(self, /, p: _NDArrayLikeFloat, size: None = None) -> NDArray[np.int64]: ...
    @overload  # >=0d
    def logseries(self, /, p: _ArrayLikeFloat_co, size: None = None) -> NDArray[np.int64] | Any: ...

    #
    @overload  # 0d (default)
    def poisson(self, /, lam: _FloatLike_co = 1.0, size: None = None) -> int: ...
    @overload  # size=<given> (keyword)
    def poisson(self, /, lam: _ArrayLikeFloat_co = 1.0, *, size: _ShapeLike) -> NDArray[np.int64]: ...
    @overload  # size=<given> (positional)
    def poisson(self, /, lam: _ArrayLikeFloat_co, size: _ShapeLike) -> NDArray[np.int64]: ...
    @overload  # >0d
    def poisson(self, /, lam: _NDArrayLikeFloat, size: None = None) -> NDArray[np.int64]: ...
    @overload  # >=0d
    def poisson(self, /, lam: _ArrayLikeFloat_co, size: None = None) -> NDArray[np.int64] | Any: ...

    #
    @overload  # 0d, 0d
    def binomial(self, /, n: int, p: _FloatLike_co, size: None = None) -> int: ...
    @overload  # size=<given>
    def binomial(self, /, n: _ArrayLikeInt_co, p: _ArrayLikeFloat_co, size: _ShapeLike) -> NDArray[np.int64]: ...
    @overload  # >=0d, >0d
    def binomial(self, /, n: _ArrayLikeInt_co, p: _NDArrayLikeFloat, size: None = None) -> NDArray[np.int64]: ...
    @overload  # >0d, >=0d
    def binomial(self, /, n: _NDArrayLikeInt, p: _ArrayLikeFloat_co, size: None = None) -> NDArray[np.int64]: ...
    @overload  # >=0d, >=0d
    def binomial(self, /, n: _ArrayLikeInt_co, p: _ArrayLikeFloat_co, size: None = None) -> NDArray[np.int64] | Any: ...

    #
    @overload  # 0d, 0d
    def negative_binomial(self, /, n: _FloatLike_co, p: _FloatLike_co, size: None = None) -> int: ...
    @overload  # size=<given>
    def negative_binomial(self, /, n: _ArrayLikeFloat_co, p: _ArrayLikeFloat_co, size: _ShapeLike) -> NDArray[np.int64]: ...
    @overload  # >=0d, >0d
    def negative_binomial(self, /, n: _ArrayLikeFloat_co, p: _NDArrayLikeFloat, size: None = None) -> NDArray[np.int64]: ...
    @overload  # >0d, >=0d
    def negative_binomial(self, /, n: _NDArrayLikeFloat, p: _ArrayLikeFloat_co, size: None = None) -> NDArray[np.int64]: ...
    @overload  # >=0d, >=0d
    def negative_binomial(
        self, /, n: _ArrayLikeFloat_co, p: _ArrayLikeFloat_co, size: None = None
    ) -> NDArray[np.int64] | Any: ...

    #
    @overload  # 0d, 0d, 0d
    def hypergeometric(self, /, ngood: int, nbad: int, nsample: int, size: None = None) -> int: ...
    @overload  # size=<given>
    def hypergeometric(
        self, /, ngood: _ArrayLikeInt_co, nbad: _ArrayLikeInt_co, nsample: _ArrayLikeInt_co, size: _ShapeLike
    ) -> NDArray[np.int64]: ...
    @overload  # >=0d, >=0d, >0d
    def hypergeometric(
        self, /, ngood: _ArrayLikeInt_co, nbad: _ArrayLikeInt_co, nsample: _NDArrayLikeInt, size: None = None
    ) -> NDArray[np.int64] | Any: ...
    @overload  # >=0d, >0d, >=0d
    def hypergeometric(
        self, /, ngood: _ArrayLikeInt_co, nbad: _NDArrayLikeInt, nsample: _ArrayLikeInt_co, size: None = None
    ) -> NDArray[np.int64] | Any: ...
    @overload  # >0d, >=0d, >=0d
    def hypergeometric(
        self, /, ngood: _NDArrayLikeInt, nbad: _ArrayLikeInt_co, nsample: _ArrayLikeInt_co, size: None = None
    ) -> NDArray[np.int64] | Any: ...
    @overload  # >=0d, >=0d, >=0d
    def hypergeometric(
        self, /, ngood: _ArrayLikeInt_co, nbad: _ArrayLikeInt_co, nsample: _ArrayLikeInt_co, size: None = None
    ) -> NDArray[np.int64] | Any: ...

    ###
    # multivariate

    #
    def dirichlet(self, /, alpha: _ArrayLikeFloat_co, size: _ShapeLike | None = None) -> _ArrayF64: ...

    #
    def multivariate_normal(
        self,
        /,
        mean: _ArrayLikeFloat_co,
        cov: _ArrayLikeFloat_co,
        size: _ShapeLike | None = None,
        check_valid: Literal["warn", "raise", "ignore"] = "warn",
        tol: float = 1e-8,
        *,
        method: Literal["svd", "eigh", "cholesky"] = "svd",
    ) -> _ArrayF64: ...

    #
    def multinomial(
        self, /, n: _ArrayLikeInt_co, pvals: _ArrayLikeFloat_co, size: _ShapeLike | None = None
    ) -> NDArray[np.int64]: ...

    #
    def multivariate_hypergeometric(
        self,
        /,
        colors: _ArrayLikeInt_co,
        nsample: int,
        size: _ShapeLike | None = None,
        method: Literal["marginals", "count"] = "marginals",
    ) -> NDArray[np.int64]: ...

    ###
    # resampling

    # axis must be 0 for MutableSequence
    @overload
    def shuffle(self, /, x: np.ndarray, axis: int = 0) -> None: ...
    @overload
    def shuffle(self, /, x: MutableSequence[Any], axis: Literal[0] = 0) -> None: ...

    #
    @overload
    def permutation(self, /, x: int, axis: int = 0) -> NDArray[np.int64]: ...
    @overload
    def permutation(self, /, x: ArrayLike, axis: int = 0) -> np.ndarray: ...

    #
    @overload
    def permuted[ArrayT: np.ndarray](self, /, x: ArrayT, *, axis: int | None = None, out: None = None) -> ArrayT: ...
    @overload
    def permuted(self, /, x: ArrayLike, *, axis: int | None = None, out: None = None) -> np.ndarray: ...
    @overload
    def permuted[ArrayT: np.ndarray](self, /, x: ArrayLike, *, axis: int | None = None, out: ArrayT) -> ArrayT: ...

    #
    @overload  # >=0d int, size=None (default)
    def choice(
        self,
        /,
        a: int | _NestedSequence[int],
        size: None = None,
        replace: bool = True,
        p: _ArrayLikeFloat_co | None = None,
        axis: int = 0,
        shuffle: bool = True,
    ) -> int: ...
    @overload  # >=0d known, size=None (default)
    def choice[ScalarT: np.generic](
        self,
        /,
        a: _ArrayLike[ScalarT],
        size: None = None,
        replace: bool = True,
        p: _ArrayLikeFloat_co | None = None,
        axis: int = 0,
        shuffle: bool = True,
    ) -> ScalarT: ...
    @overload  # >=0d unknown, size=None (default)
    def choice(
        self,
        /,
        a: ArrayLike,
        size: None = None,
        replace: bool = True,
        p: _ArrayLikeFloat_co | None = None,
        axis: int = 0,
        shuffle: bool = True,
    ) -> Any: ...
    @overload  # >=0d int, size=<given>
    def choice(
        self,
        /,
        a: int | _NestedSequence[int],
        size: _ShapeLike,
        replace: bool = True,
        p: _ArrayLikeFloat_co | None = None,
        axis: int = 0,
        shuffle: bool = True,
    ) -> NDArray[np.int64]: ...
    @overload  # >=0d known, size=<given>
    def choice[ScalarT: np.generic](
        self,
        /,
        a: _ArrayLike[ScalarT],
        size: _ShapeLike,
        replace: bool = True,
        p: _ArrayLikeFloat_co | None = None,
        axis: int = 0,
        shuffle: bool = True,
    ) -> NDArray[ScalarT]: ...
    @overload  # >=0d unknown, size=<given>
    def choice(
        self,
        /,
        a: ArrayLike,
        size: _ShapeLike,
        replace: bool = True,
        p: _ArrayLikeFloat_co | None = None,
        axis: int = 0,
        shuffle: bool = True,
    ) -> np.ndarray: ...

def default_rng(seed: _ArrayLikeInt_co | SeedSequence | BitGenerator | Generator | RandomState | None = None) -> Generator: ...
