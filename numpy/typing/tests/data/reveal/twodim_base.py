from typing import Any, List, TypeVar

import numpy as np
import numpy.typing as npt

_SCT = TypeVar("_SCT", bound=np.generic)


def func1(ar: npt.NDArray[_SCT], a: int) -> npt.NDArray[_SCT]:
    pass


def func2(ar: npt.NDArray[np.number[Any]], a: str) -> npt.NDArray[np.float64]:
    pass


AR_b: npt.NDArray[np.bool_]
AR_u: npt.NDArray[np.uint64]
AR_i: npt.NDArray[np.int64]
AR_f: npt.NDArray[np.float64]
AR_c: npt.NDArray[np.complex128]
AR_O: npt.NDArray[np.object_]

AR_LIKE_b: List[bool]

reveal_type(np.fliplr(AR_b))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.fliplr(AR_LIKE_b))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.flipud(AR_b))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.flipud(AR_LIKE_b))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.eye(10))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(
    np.eye(10, M=20, dtype=np.int64)
)  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(np.eye(10, k=2, dtype=int))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.diag(AR_b))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.diag(AR_LIKE_b, k=0))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.diagflat(AR_b))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.diagflat(AR_LIKE_b, k=0))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.tri(10))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(
    np.tri(10, M=20, dtype=np.int64)
)  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(np.tri(10, k=2, dtype=int))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.tril(AR_b))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.tril(AR_LIKE_b, k=0))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.triu(AR_b))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.triu(AR_LIKE_b, k=0))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(
    np.vander(AR_b)
)  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[Any]]]
reveal_type(
    np.vander(AR_u)
)  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[Any]]]
reveal_type(
    np.vander(AR_i, N=2)
)  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[Any]]]
reveal_type(
    np.vander(AR_f, increasing=True)
)  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]
reveal_type(
    np.vander(AR_c)
)  # E: numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]]
reveal_type(np.vander(AR_O))  # E: numpy.ndarray[Any, numpy.dtype[numpy.object_]]

reveal_type(
    np.histogram2d(AR_i, AR_b)
)  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]], numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]]
reveal_type(
    np.histogram2d(AR_f, AR_f)
)  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]], numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]]
reveal_type(
    np.histogram2d(AR_f, AR_c, weights=AR_LIKE_b)
)  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]], numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]]]

reveal_type(
    np.mask_indices(10, func1)
)  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(
    np.mask_indices(8, func2, "0")
)  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]

reveal_type(
    np.tril_indices(10)
)  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{int_}]], numpy.ndarray[Any, numpy.dtype[{int_}]]]

reveal_type(
    np.tril_indices_from(AR_b)
)  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{int_}]], numpy.ndarray[Any, numpy.dtype[{int_}]]]

reveal_type(
    np.triu_indices(10)
)  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{int_}]], numpy.ndarray[Any, numpy.dtype[{int_}]]]

reveal_type(
    np.triu_indices_from(AR_b)
)  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{int_}]], numpy.ndarray[Any, numpy.dtype[{int_}]]]
