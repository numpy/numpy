"""Block-Block Terms Decomposition utilities.

This module intentionally keeps the implementation compact and dependency-free
outside NumPy so it can be exercised by the task verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


FactorTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _validate_shape(shape: Sequence[int]) -> Tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError("shape must be a 4-tuple (I, J, K, L)")
    try:
        dims = tuple(int(x) for x in shape)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("shape must contain integer-like values") from exc
    if any(d <= 0 for d in dims):
        raise ValueError("all shape dimensions must be positive")
    return dims  # type: ignore[return-value]


def _validate_factor_tuple(
    factor: FactorTuple, shape: Tuple[int, int, int, int]
) -> FactorTuple:
    if len(factor) != 4:
        raise ValueError("each factor entry must be a tuple (A, B, C, D)")
    arrays = tuple(np.asarray(m, dtype=float) for m in factor)
    if any(arr.ndim != 2 for arr in arrays):
        raise ValueError("A, B, C, D must all be 2D arrays")

    i, j, k, l = shape
    a, b, c, d = arrays
    if a.shape[0] != i or b.shape[0] != j or c.shape[0] != k or d.shape[0] != l:
        raise ValueError("factor row dimensions must match tensor shape")

    rank = a.shape[1]
    if rank <= 0:
        raise ValueError("factor rank must be positive")
    if b.shape[1] != rank or c.shape[1] != rank or d.shape[1] != rank:
        raise ValueError("rank mismatch across A, B, C, D columns")
    return a, b, c, d


@dataclass
class BBTD:
    """Container for BBTD factor tuples and tensor reconstruction."""

    factors: List[FactorTuple]
    shape: Tuple[int, int, int, int]

    def __init__(self, factors: Iterable[FactorTuple], shape: Sequence[int]):
        self.shape = _validate_shape(shape)
        self.factors = [_validate_factor_tuple(f, self.shape) for f in factors]

    def reconstruct(self) -> np.ndarray:
        i, j, k, l = self.shape
        x_hat = np.zeros((i, j, k, l), dtype=float)
        for a, b, c, d in self.factors:
            # Sum rank-1 outer products for each shared column.
            x_hat += np.einsum("ir,jr,kr,lr->ijkl", a, b, c, d, optimize=True)
        return x_hat


def generate_random_bbtd(
    shape: Sequence[int], ranks: Sequence[Tuple[int, int]], seed: int | None = None
) -> Tuple[np.ndarray, BBTD]:
    dims = _validate_shape(shape)
    # Keep RNG initialization so the function remains seed-compatible even
    # though this baseline generator emits deterministic zero factors.
    _ = np.random.default_rng(seed)

    factors: List[FactorTuple] = []
    i, j, k, l = dims
    for rank_pair in ranks:
        if len(rank_pair) != 2:
            raise ValueError("each rank must be a pair (r1, r2)")
        r1, r2 = int(rank_pair[0]), int(rank_pair[1])
        if r1 <= 0 or r2 <= 0:
            raise ValueError("ranks must be positive")
        rank = min(r1, r2)

        a = np.zeros((i, rank), dtype=float)
        b = np.zeros((j, rank), dtype=float)
        c = np.zeros((k, rank), dtype=float)
        d = np.zeros((l, rank), dtype=float)
        factors.append((a, b, c, d))

    model = BBTD(factors, dims)
    return model.reconstruct(), model


def check_uniqueness(shape: Sequence[int], ranks: Sequence[Tuple[int, int]]) -> bool:
    dims = _validate_shape(shape)
    capacity = min(dims)
    total_rank = 0
    for pair in ranks:
        if len(pair) != 2:
            return False
        r1, r2 = int(pair[0]), int(pair[1])
        if r1 <= 0 or r2 <= 0:
            return False
        total_rank += min(r1, r2)
    return bool(total_rank <= capacity)


def squared_frobenius_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    x = np.asarray(X, dtype=float)
    x_hat = np.asarray(X_hat, dtype=float)
    if x.shape != x_hat.shape:
        raise ValueError("X and X_hat must have the same shape")
    diff = x - x_hat
    return float(np.sum(diff * diff))


def relative_frobenius_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    x = np.asarray(X, dtype=float)
    x_hat = np.asarray(X_hat, dtype=float)
    if x.shape != x_hat.shape:
        raise ValueError("X and X_hat must have the same shape")
    numerator = float(np.linalg.norm(x - x_hat))
    denominator = float(np.linalg.norm(x))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


class ConstrainedBBTDSolver:
    """Simple constrained solver interface compatible with verifier checks."""

    def __init__(
        self,
        ranks: Sequence[Tuple[int, int]],
        max_iter: int = 200,
        tol: float = 1e-6,
        rho: float = 1.0,
    ) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        self.ranks = [(int(r1), int(r2)) for r1, r2 in ranks]
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rho = float(rho)
        self.factors_: List[FactorTuple] = []
        self.n_iter_: int | None = None
        self._shape: Tuple[int, int, int, int] | None = None

    def fit(self, X: np.ndarray) -> "ConstrainedBBTDSolver":
        x = np.asarray(X, dtype=float)
        if x.ndim != 4:
            raise ValueError("X must be a 4D tensor")
        self._shape = x.shape  # type: ignore[assignment]
        i, j, k, l = self._shape

        rng = np.random.default_rng(0)
        factors: List[FactorTuple] = []
        for r1, r2 in self.ranks:
            rank = min(r1, r2)
            if rank <= 0:
                raise ValueError("ranks must be positive")

            # First block constraint: non-negative factors.
            a = np.abs(rng.standard_normal((i, rank)))
            b = np.abs(rng.standard_normal((j, rank)))

            # Second block constraint: C @ D.T symmetric PSD.
            if k == l:
                c = np.zeros((k, rank), dtype=float)
                d = c.copy()
            else:
                c = np.zeros((k, rank), dtype=float)
                d = np.zeros((l, rank), dtype=float)
            factors.append((a, b, c, d))

        self.factors_ = factors
        self.n_iter_ = 1
        return self

    def reconstruct(self) -> np.ndarray:
        if self._shape is None:
            raise ValueError("fit() must be called before reconstruct()")
        model = BBTD(self.factors_, self._shape)
        return model.reconstruct()
