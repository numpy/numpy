from types import EllipsisType
from typing import Any, Literal, assert_type

import numpy as np
import numpy.typing as npt

AR_LIKE_b: list[bool]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_c: list[complex]
AR_LIKE_S: list[bytes]
AR_LIKE_U: list[str]
AR_LIKE_O: list[object]

AR_i8: npt.NDArray[np.int64]
AR_i8_2d: np.ndarray[tuple[int, int], np.dtype[np.int64]]
AR_i8_3d: np.ndarray[tuple[int, int, int], np.dtype[np.int64]]
AR_i8_4d: np.ndarray[tuple[int, int, int, int], np.dtype[np.int64]]
AR_i8_5d: np.ndarray[tuple[int, int, int, int, int], np.dtype[np.int64]]
AR_f4: npt.NDArray[np.float32]
AR_O: npt.NDArray[np.object_]

type _Int1D = np.ndarray[tuple[int], np.dtype[np.intp]]

assert_type(np.ndenumerate(AR_i8), np.ndenumerate[np.int64])
assert_type(np.ndenumerate(AR_LIKE_f), np.ndenumerate[np.float64])
assert_type(np.ndenumerate(AR_LIKE_U), np.ndenumerate[np.str_])
assert_type(np.ndenumerate(AR_LIKE_O), np.ndenumerate[Any])

assert_type(next(np.ndenumerate(AR_i8)), tuple[tuple[Any, ...], np.int64])
assert_type(next(np.ndenumerate(AR_LIKE_f)), tuple[tuple[Any, ...], np.float64])
assert_type(next(np.ndenumerate(AR_LIKE_U)), tuple[tuple[Any, ...], np.str_])
assert_type(next(np.ndenumerate(AR_LIKE_O)), tuple[tuple[Any, ...], Any])

assert_type(iter(np.ndenumerate(AR_i8)), np.ndenumerate[np.int64])
assert_type(iter(np.ndenumerate(AR_LIKE_f)), np.ndenumerate[np.float64])
assert_type(iter(np.ndenumerate(AR_LIKE_U)), np.ndenumerate[np.str_])
assert_type(iter(np.ndenumerate(AR_LIKE_O)), np.ndenumerate[Any])

assert_type(np.ndindex(1, 2, 3), np.ndindex)
assert_type(np.ndindex((1, 2, 3)), np.ndindex)
assert_type(iter(np.ndindex(1, 2, 3)), np.ndindex)
assert_type(next(np.ndindex(1, 2, 3)), tuple[Any, ...])

assert_type(np.unravel_index([22, 41, 37], (7, 6)), tuple[npt.NDArray[np.intp], ...])
assert_type(np.unravel_index([31, 41, 13], (7, 6), order="F"), tuple[npt.NDArray[np.intp], ...])
assert_type(np.unravel_index(1621, (6, 7, 8, 9)), tuple[np.intp, ...])

assert_type(np.ravel_multi_index([[1]], (7, 6)), npt.NDArray[np.intp])
assert_type(np.ravel_multi_index(AR_LIKE_i, (7, 6)), np.intp)
assert_type(np.ravel_multi_index(AR_LIKE_i, (7, 6), order="F"), np.intp)
assert_type(np.ravel_multi_index(AR_LIKE_i, (4, 6), mode="clip"), np.intp)
assert_type(np.ravel_multi_index(AR_LIKE_i, (4, 4), mode=("clip", "wrap")), np.intp)
assert_type(np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9)), np.intp)

assert_type(np.mgrid[1:1:2], npt.NDArray[Any])
assert_type(np.mgrid[1:1:2, None:10], npt.NDArray[Any])

assert_type(np.ogrid[1:1:2], tuple[npt.NDArray[Any], ...])
assert_type(np.ogrid[1:1:2, None:10], tuple[npt.NDArray[Any], ...])

assert_type(np.index_exp[0:1], tuple[slice[int, int, None]])
assert_type(np.index_exp[0:1, None:3], tuple[slice[int, int, None], slice[None, int, None]])
assert_type(np.index_exp[0, 0:1, ..., [0, 1, 3]], tuple[Literal[0], slice[int, int, None], EllipsisType, list[int]])

assert_type(np.s_[0:1], slice[int, int, None])
assert_type(np.s_[0:1, None:3], tuple[slice[int, int, None], slice[None, int, None]])
assert_type(np.s_[0, 0:1, ..., [0, 1, 3]], tuple[Literal[0], slice[int, int, None], EllipsisType, list[int]])

assert_type(np.ix_(AR_LIKE_b), tuple[np.ndarray[tuple[int], np.dtype[np.int_]]])
assert_type(np.ix_(AR_LIKE_i), tuple[np.ndarray[tuple[int], np.dtype[np.int_]]])
assert_type(np.ix_(AR_f4), tuple[np.ndarray[tuple[int], np.dtype[np.float32]]])
assert_type(
    np.ix_(AR_LIKE_b, AR_LIKE_b),
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.int_]],
        np.ndarray[tuple[int, int], np.dtype[np.int_]],
    ],
)
assert_type(
    np.ix_(AR_LIKE_i, AR_LIKE_i),
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.int_]],
        np.ndarray[tuple[int, int], np.dtype[np.int_]],
    ],
)
assert_type(
    np.ix_(AR_f4, AR_f4),
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
    ],
)
assert_type(
    np.ix_(AR_LIKE_b, AR_LIKE_b, AR_LIKE_b),
    tuple[
        np.ndarray[tuple[int, int, int], np.dtype[np.int_]],
        np.ndarray[tuple[int, int, int], np.dtype[np.int_]],
        np.ndarray[tuple[int, int, int], np.dtype[np.int_]],
    ],
)
assert_type(
    np.ix_(AR_LIKE_i, AR_LIKE_i, AR_LIKE_i),
    tuple[
        np.ndarray[tuple[int, int, int], np.dtype[np.int_]],
        np.ndarray[tuple[int, int, int], np.dtype[np.int_]],
        np.ndarray[tuple[int, int, int], np.dtype[np.int_]],
    ],
)
assert_type(
    np.ix_(AR_f4, AR_f4, AR_f4),
    tuple[
        np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    ],
)
assert_type(
    np.ix_(AR_LIKE_b, AR_LIKE_b, AR_LIKE_b, AR_LIKE_b),
    tuple[npt.NDArray[np.int_], ...],
)
assert_type(
    np.ix_(AR_LIKE_i, AR_LIKE_i, AR_LIKE_i, AR_LIKE_b),
    tuple[npt.NDArray[np.int_], ...],
)
assert_type(
    np.ix_(AR_f4, AR_f4, AR_f4, AR_f4),
    tuple[npt.NDArray[np.float32], ...],
)
assert_type(np.ix_(AR_LIKE_f), tuple[npt.NDArray[np.float64], ...])
assert_type(np.ix_(AR_LIKE_c), tuple[npt.NDArray[np.complex128], ...])
assert_type(np.ix_(AR_LIKE_S), tuple[npt.NDArray[np.bytes_], ...])
assert_type(np.ix_(AR_LIKE_U), tuple[npt.NDArray[np.str_], ...])

assert_type(np.fill_diagonal(AR_i8, 5), None)

assert_type(np.diag_indices(4), tuple[_Int1D, _Int1D])
assert_type(np.diag_indices(4, 0), tuple[()])
assert_type(np.diag_indices(4, 1), tuple[_Int1D])
assert_type(np.diag_indices(4, 2), tuple[_Int1D, _Int1D])
assert_type(np.diag_indices(4, 3), tuple[_Int1D, _Int1D, _Int1D])
assert_type(np.diag_indices(4, 4), tuple[_Int1D, ...])

assert_type(np.diag_indices_from(AR_i8), tuple[_Int1D, _Int1D, *tuple[_Int1D, ...]])
assert_type(np.diag_indices_from(AR_i8_2d), tuple[_Int1D, _Int1D])
assert_type(np.diag_indices_from(AR_i8_3d), tuple[_Int1D, _Int1D, _Int1D])
assert_type(np.diag_indices_from(AR_i8_4d), tuple[_Int1D, _Int1D, _Int1D, _Int1D])
assert_type(np.diag_indices_from(AR_i8_5d), tuple[_Int1D, _Int1D, *tuple[_Int1D, ...]])
