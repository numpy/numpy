from typing import Any, List
import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]
AR_i8: npt.NDArray[np.int64]
AR_u1: npt.NDArray[np.uint8]

AR_LIKE_f: List[float]
AR_LIKE_i: List[int]

b_f8 = np.broadcast(AR_f8)
b_i8_f8_f8 = np.broadcast(AR_i8, AR_f8, AR_f8)

reveal_type(next(b_f8))  # E: tuple[Any]
reveal_type(b_f8.reset())  # E: None
reveal_type(b_f8.index)  # E: int
reveal_type(b_f8.iters)  # E: tuple[numpy.flatiter[Any]]
reveal_type(b_f8.nd)  # E: int
reveal_type(b_f8.ndim)  # E: int
reveal_type(b_f8.numiter)  # E: int
reveal_type(b_f8.shape)  # E: tuple[builtins.int]
reveal_type(b_f8.size)  # E: int

reveal_type(next(b_i8_f8_f8))  # E: tuple[Any]
reveal_type(b_i8_f8_f8.reset())  # E: None
reveal_type(b_i8_f8_f8.index)  # E: int
reveal_type(b_i8_f8_f8.iters)  # E: tuple[numpy.flatiter[Any]]
reveal_type(b_i8_f8_f8.nd)  # E: int
reveal_type(b_i8_f8_f8.ndim)  # E: int
reveal_type(b_i8_f8_f8.numiter)  # E: int
reveal_type(b_i8_f8_f8.shape)  # E: tuple[builtins.int]
reveal_type(b_i8_f8_f8.size)  # E: int

reveal_type(np.inner(AR_f8, AR_i8))  # E: Any

reveal_type(np.where([True, True, False]))  # E: tuple[numpy.ndarray[Any, numpy.dtype[{intp}]]]
reveal_type(np.where([True, True, False], 1, 0))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.lexsort([0, 1, 2]))  # E: Any

reveal_type(np.can_cast(np.dtype("i8"), int))  # E: bool
reveal_type(np.can_cast(AR_f8, "f8"))  # E: bool
reveal_type(np.can_cast(AR_f8, np.complex128, casting="unsafe"))  # E: bool

reveal_type(np.min_scalar_type([1]))  # E: numpy.dtype[Any]
reveal_type(np.min_scalar_type(AR_f8))  # E: numpy.dtype[Any]

reveal_type(np.result_type(int, [1]))  # E: numpy.dtype[Any]
reveal_type(np.result_type(AR_f8, AR_u1))  # E: numpy.dtype[Any]
reveal_type(np.result_type(AR_f8, np.complex128))  # E: numpy.dtype[Any]

reveal_type(np.dot(AR_LIKE_f, AR_i8))  # E: Any
reveal_type(np.dot(AR_u1, 1))  # E: Any
reveal_type(np.dot(1.5j, 1))  # E: Any
reveal_type(np.dot(AR_u1, 1, out=AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]

reveal_type(np.vdot(AR_LIKE_f, AR_i8))  # E: numpy.floating[Any]
reveal_type(np.vdot(AR_u1, 1))  # E: numpy.signedinteger[Any]
reveal_type(np.vdot(1.5j, 1))  # E: numpy.complexfloating[Any, Any]

reveal_type(np.bincount(AR_i8))  # E: numpy.ndarray[Any, numpy.dtype[{intp}]]

reveal_type(np.copyto(AR_f8, [1., 1.5, 1.6]))  # E: None

reveal_type(np.putmask(AR_f8, [True, True, False], 1.5))  # E: None

reveal_type(np.packbits(AR_i8))  # numpy.ndarray[Any, numpy.dtype[{uint8}]]
reveal_type(np.packbits(AR_u1))  # numpy.ndarray[Any, numpy.dtype[{uint8}]]

reveal_type(np.unpackbits(AR_u1))  # numpy.ndarray[Any, numpy.dtype[{uint8}]]

reveal_type(np.shares_memory(1, 2))  # E: bool
reveal_type(np.shares_memory(AR_f8, AR_f8, max_work=1))  # E: bool

reveal_type(np.may_share_memory(1, 2))  # E: bool
reveal_type(np.may_share_memory(AR_f8, AR_f8, max_work=1))  # E: bool
