import numpy as np
import numpy.typing as npt

AR_b: npt.NDArray[np.bool_]
AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_M: npt.NDArray[np.datetime64]
AR_O: npt.NDArray[np.object_]

AR_LIKE_f8: list[float]

reveal_type(np.ediff1d(AR_b))  # E: numpy.ndarray[Any, numpy.dtype[{int8}]]
reveal_type(np.ediff1d(AR_i8, to_end=[1, 2, 3]))  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(np.ediff1d(AR_M))  # E: numpy.ndarray[Any, numpy.dtype[numpy.timedelta64]]
reveal_type(np.ediff1d(AR_O))  # E: numpy.ndarray[Any, numpy.dtype[numpy.object_]]
reveal_type(np.ediff1d(AR_LIKE_f8, to_begin=[1, 1.5]))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.intersect1d(AR_i8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(np.intersect1d(AR_M, AR_M, assume_unique=True))  # E: numpy.ndarray[Any, numpy.dtype[numpy.datetime64]]
reveal_type(np.intersect1d(AR_f8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[Any]]
reveal_type(np.intersect1d(AR_f8, AR_f8, return_indices=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]

reveal_type(np.setxor1d(AR_i8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(np.setxor1d(AR_M, AR_M, assume_unique=True))  # E: numpy.ndarray[Any, numpy.dtype[numpy.datetime64]]
reveal_type(np.setxor1d(AR_f8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.in1d(AR_i8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.in1d(AR_M, AR_M, assume_unique=True))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.in1d(AR_f8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.in1d(AR_f8, AR_LIKE_f8, invert=True))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(np.isin(AR_i8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isin(AR_M, AR_M, assume_unique=True))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isin(AR_f8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isin(AR_f8, AR_LIKE_f8, invert=True))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(np.union1d(AR_i8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(np.union1d(AR_M, AR_M))  # E: numpy.ndarray[Any, numpy.dtype[numpy.datetime64]]
reveal_type(np.union1d(AR_f8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.setdiff1d(AR_i8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(np.setdiff1d(AR_M, AR_M, assume_unique=True))  # E: numpy.ndarray[Any, numpy.dtype[numpy.datetime64]]
reveal_type(np.setdiff1d(AR_f8, AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.unique(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(np.unique(AR_LIKE_f8, axis=0))  # E: numpy.ndarray[Any, numpy.dtype[Any]]
reveal_type(np.unique(AR_f8, return_index=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_index=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[Any]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_inverse=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_inverse=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[Any]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_counts=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_counts=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[Any]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_index=True, return_inverse=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_index=True, return_inverse=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[Any]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_index=True, return_counts=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_index=True, return_counts=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[Any]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_inverse=True, return_counts=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_inverse=True, return_counts=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[Any]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_index=True, return_inverse=True, return_counts=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[{float64}]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_index=True, return_inverse=True, return_counts=True))  # E: Tuple[numpy.ndarray[Any, numpy.dtype[Any]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]], numpy.ndarray[Any, numpy.dtype[{intp}]]]
