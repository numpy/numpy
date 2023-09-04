import numpy as np
import numpy.typing as npt

AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]

reveal_type(np.histogram_bin_edges(AR_i8, bins="auto"))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.histogram_bin_edges(AR_i8, bins="rice", range=(0, 3)))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.histogram_bin_edges(AR_i8, bins="scott", weights=AR_f8))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.histogram(AR_i8, bins="auto"))  # E: tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]
reveal_type(np.histogram(AR_i8, bins="rice", range=(0, 3)))  # E: tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]
reveal_type(np.histogram(AR_i8, bins="scott", weights=AR_f8))  # E: tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]
reveal_type(np.histogram(AR_f8, bins=1, density=True))  # E: tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]

reveal_type(np.histogramdd(AR_i8, bins=[1]))  # E: tuple[ndarray[Any, dtype[Any]], builtins.list[ndarray[Any, dtype[Any]]]]
reveal_type(np.histogramdd(AR_i8, range=[(0, 3)]))  # E: tuple[ndarray[Any, dtype[Any]], builtins.list[ndarray[Any, dtype[Any]]]]
reveal_type(np.histogramdd(AR_i8, weights=AR_f8))  # E: tuple[ndarray[Any, dtype[Any]], builtins.list[ndarray[Any, dtype[Any]]]]
reveal_type(np.histogramdd(AR_f8, density=True))  # E: tuple[ndarray[Any, dtype[Any]], builtins.list[ndarray[Any, dtype[Any]]]]
