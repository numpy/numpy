from typing import Literal, assert_type

import numpy as np
import numpy.typing as npt
from numpy._typing import _AnyShape

type AR_T_alias = np.ndarray[_AnyShape, np.dtypes.StringDType]
type AR_TU_alias = AR_T_alias | npt.NDArray[np.str_]

AR_U: npt.NDArray[np.str_]
AR_S: npt.NDArray[np.bytes_]
AR_T: AR_T_alias

assert_type(np.strings.equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.equal(AR_S, AR_S), npt.NDArray[np.bool])
assert_type(np.strings.equal(AR_T, AR_T), npt.NDArray[np.bool])

assert_type(np.strings.not_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.not_equal(AR_S, AR_S), npt.NDArray[np.bool])
assert_type(np.strings.not_equal(AR_T, AR_T), npt.NDArray[np.bool])

assert_type(np.strings.greater_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.greater_equal(AR_S, AR_S), npt.NDArray[np.bool])
assert_type(np.strings.greater_equal(AR_T, AR_T), npt.NDArray[np.bool])

assert_type(np.strings.less_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.less_equal(AR_S, AR_S), npt.NDArray[np.bool])
assert_type(np.strings.less_equal(AR_T, AR_T), npt.NDArray[np.bool])

assert_type(np.strings.greater(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.greater(AR_S, AR_S), npt.NDArray[np.bool])
assert_type(np.strings.greater(AR_T, AR_T), npt.NDArray[np.bool])

assert_type(np.strings.less(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.less(AR_S, AR_S), npt.NDArray[np.bool])
assert_type(np.strings.less(AR_T, AR_T), npt.NDArray[np.bool])

assert_type(np.strings.add(AR_U, AR_U), np.ndarray)
assert_type(np.strings.add(AR_S, AR_S), np.ndarray)
assert_type(np.strings.add(AR_T, AR_T), np.ndarray)

assert_type(np.strings.multiply(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.multiply(AR_S, [5, 4, 3]), npt.NDArray[np.bytes_])
assert_type(np.strings.multiply(AR_T, 5), AR_T_alias)

assert_type(np.strings.mod(AR_U, "test"), npt.NDArray[np.str_])
assert_type(np.strings.mod(AR_S, "test"), npt.NDArray[np.bytes_])
assert_type(np.strings.mod(AR_T, "test"), AR_T_alias)

assert_type(np.strings.capitalize(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.capitalize(AR_S), npt.NDArray[np.bytes_])
assert_type(np.strings.capitalize(AR_T), AR_T_alias)

assert_type(np.strings.center(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.center(AR_S, [2, 3, 4], b"a"), npt.NDArray[np.bytes_])
assert_type(np.strings.center(AR_T, 5), AR_T_alias)

assert_type(np.strings.encode(AR_U), npt.NDArray[np.bytes_])
assert_type(np.strings.encode(AR_T), npt.NDArray[np.bytes_])
assert_type(np.strings.decode(AR_S), npt.NDArray[np.str_])

assert_type(np.strings.expandtabs(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.expandtabs(AR_S, tabsize=4), npt.NDArray[np.bytes_])
assert_type(np.strings.expandtabs(AR_T), AR_T_alias)

assert_type(np.strings.ljust(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.ljust(AR_S, [4, 3, 1], fillchar=[b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.strings.ljust(AR_T, 5), AR_T_alias)
assert_type(np.strings.ljust(AR_T, [4, 2, 1], fillchar=["a", "b", "c"]), AR_T_alias)

assert_type(np.strings.rjust(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.rjust(AR_S, [4, 3, 1], fillchar=[b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.strings.rjust(AR_T, 5), AR_T_alias)
assert_type(np.strings.rjust(AR_T, [4, 2, 1], fillchar=["a", "b", "c"]), AR_T_alias)

assert_type(np.strings.lstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.lstrip(AR_S, b"_"), npt.NDArray[np.bytes_])
assert_type(np.strings.lstrip(AR_T), AR_T_alias)
assert_type(np.strings.lstrip(AR_T, "_"), AR_T_alias)

assert_type(np.strings.rstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.rstrip(AR_S, b"_"), npt.NDArray[np.bytes_])
assert_type(np.strings.rstrip(AR_T), AR_T_alias)
assert_type(np.strings.rstrip(AR_T, "_"), AR_T_alias)

assert_type(np.strings.strip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.strip(AR_S, b"_"), npt.NDArray[np.bytes_])
assert_type(np.strings.strip(AR_T), AR_T_alias)
assert_type(np.strings.strip(AR_T, "_"), AR_T_alias)

assert_type(np.strings.count(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.count(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.count(AR_T, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.count(AR_T, ["a", "b", "c"], end=9), npt.NDArray[np.int_])

type _tuple3[T] = tuple[T, T, T]

assert_type(np.strings.partition(AR_U, "\n"), _tuple3[npt.NDArray[np.str_]])
assert_type(np.strings.partition(AR_S, [b"a", b"b", b"c"]), _tuple3[npt.NDArray[np.bytes_]])
assert_type(np.strings.partition(AR_T, "\n"), _tuple3[AR_TU_alias])

assert_type(np.strings.rpartition(AR_U, "\n"), _tuple3[npt.NDArray[np.str_]])
assert_type(np.strings.rpartition(AR_S, [b"a", b"b", b"c"]), _tuple3[npt.NDArray[np.bytes_]])
assert_type(np.strings.rpartition(AR_T, "\n"), _tuple3[AR_TU_alias])

assert_type(np.strings.replace(AR_U, "_", "-"), npt.NDArray[np.str_])
assert_type(np.strings.replace(AR_S, [b"_", b""], [b"a", b"b"]), npt.NDArray[np.bytes_])
assert_type(np.strings.replace(AR_T, "_", "_"), AR_TU_alias)

assert_type(np.strings.lower(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.lower(AR_S), npt.NDArray[np.bytes_])
assert_type(np.strings.lower(AR_T), AR_T_alias)

assert_type(np.strings.upper(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.upper(AR_S), npt.NDArray[np.bytes_])
assert_type(np.strings.upper(AR_T), AR_T_alias)

assert_type(np.strings.swapcase(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.swapcase(AR_S), npt.NDArray[np.bytes_])
assert_type(np.strings.swapcase(AR_T), AR_T_alias)

assert_type(np.strings.title(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.title(AR_S), npt.NDArray[np.bytes_])
assert_type(np.strings.title(AR_T), AR_T_alias)

assert_type(np.strings.zfill(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.zfill(AR_S, [2, 3, 4]), npt.NDArray[np.bytes_])
assert_type(np.strings.zfill(AR_T, 5), AR_T_alias)

assert_type(np.strings.endswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool])
assert_type(np.strings.endswith(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])
assert_type(np.strings.endswith(AR_T, "a", start=[1, 2, 3]), npt.NDArray[np.bool])

assert_type(np.strings.startswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool])
assert_type(np.strings.startswith(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])
assert_type(np.strings.startswith(AR_T, "a", start=[1, 2, 3]), npt.NDArray[np.bool])

assert_type(np.strings.find(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.find(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.find(AR_T, "a", start=[1, 2, 3]), npt.NDArray[np.int_])

assert_type(np.strings.rfind(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.rfind(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.rfind(AR_T, "a", start=[1, 2, 3]), npt.NDArray[np.int_])

assert_type(np.strings.index(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.index(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.index(AR_T, "a", start=[1, 2, 3]), npt.NDArray[np.int_])

assert_type(np.strings.rindex(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.rindex(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.rindex(AR_T, "a", start=[1, 2, 3]), npt.NDArray[np.int_])

assert_type(np.strings.translate(AR_U, ""), npt.NDArray[np.str_])
assert_type(np.strings.translate(AR_S, ""), npt.NDArray[np.bytes_])
assert_type(np.strings.translate(AR_T, ""), AR_T_alias)

assert_type(np.strings.slice(AR_U, 1, 5, 2), npt.NDArray[np.str_])
assert_type(np.strings.slice(AR_S, 1, 5, 2), npt.NDArray[np.bytes_])
assert_type(np.strings.slice(AR_T, 1, 5, 2), AR_T_alias)

###

_py_s_0d: bytes
_py_s_1d: list[bytes]
_py_s_2d: list[list[bytes]]
_py_u_0d: str
_py_u_1d: list[str]
_py_u_2d: list[list[str]]

_s_0d: np.bytes_
_s_1d: np.ndarray[tuple[int], np.dtype[np.bytes_]]
_s_2d: np.ndarray[tuple[int, int], np.dtype[np.bytes_]]
_s_nd: np.ndarray[_AnyShape, np.dtype[np.bytes_]]
_u_0d: np.str_
_u_1d: np.ndarray[tuple[int], np.dtype[np.str_]]
_u_2d: np.ndarray[tuple[int, int], np.dtype[np.str_]]
_u_nd: np.ndarray[_AnyShape, np.dtype[np.str_]]
_t_1d: np.ndarray[tuple[int], np.dtypes.StringDType]
_t_2d: np.ndarray[tuple[int, int], np.dtypes.StringDType]
_t_nd: np.ndarray[_AnyShape, np.dtypes.StringDType]

_b_1d: np.ndarray[tuple[int], np.dtype[np.bool]]

# _ufunc_11_ut_b
# (isdecimal, isnumeric)

assert_type(np.strings.isdecimal.identity, Literal[False])

assert_type(np.strings.isdecimal(_py_u_0d), np.bool)
assert_type(np.strings.isdecimal(_py_u_0d, out=...), np.ndarray[tuple[()], np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_py_u_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_py_u_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])

assert_type(np.strings.isdecimal(_u_0d), np.bool)
assert_type(np.strings.isdecimal(_u_0d, out=...), np.ndarray[tuple[()], np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_u_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_t_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_u_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_t_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_u_nd), np.ndarray[_AnyShape, np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_t_nd), np.ndarray[_AnyShape, np.dtype[np.bool]])

assert_type(np.strings.isdecimal(_py_u_1d, dtype=bool), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_py_u_1d, dtype="?"), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_py_u_1d, dtype="b1"), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdecimal(_py_u_1d, dtype=np.bool), np.ndarray[tuple[int], np.dtype[np.bool]])

assert_type(np.strings.isdecimal(_py_u_0d, out=_b_1d), np.ndarray[tuple[int], np.dtype[np.bool_]])
assert_type(np.strings.isdecimal(_py_u_1d, out=_t_1d), np.ndarray[tuple[int], np.dtypes.StringDType])

assert_type(np.strings.isdecimal.at(_u_1d, 1), None)
assert_type(np.strings.isdecimal.at(_u_1d, (1, 1)), None)
assert_type(np.strings.isdecimal.at(_t_1d, 1), None)
assert_type(np.strings.isdecimal.at(_t_1d, (1, 1)), None)

# _ufunc_11_sut_b
# (isalnum, isalpha, isdigit, islower, isspace, istitle, isupper)

assert_type(np.strings.isdigit.identity, Literal[False])

assert_type(np.strings.isdigit(_py_s_0d), np.bool)
assert_type(np.strings.isdigit(_py_u_0d), np.bool)
assert_type(np.strings.isdigit(_py_s_0d, out=...), np.ndarray[tuple[()], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_py_u_0d, out=...), np.ndarray[tuple[()], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_py_s_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_py_u_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_py_s_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_py_u_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])

assert_type(np.strings.isdigit(_s_0d), np.bool)
assert_type(np.strings.isdigit(_u_0d), np.bool)
assert_type(np.strings.isdigit(_s_0d, out=...), np.ndarray[tuple[()], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_u_0d, out=...), np.ndarray[tuple[()], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_s_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_u_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_t_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_s_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_u_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_t_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_s_nd), np.ndarray[_AnyShape, np.dtype[np.bool]])
assert_type(np.strings.isdigit(_u_nd), np.ndarray[_AnyShape, np.dtype[np.bool]])
assert_type(np.strings.isdigit(_t_nd), np.ndarray[_AnyShape, np.dtype[np.bool]])

assert_type(np.strings.isdigit(_py_s_1d, dtype=bool), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_py_s_1d, dtype="?"), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_py_s_1d, dtype="b1"), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.strings.isdigit(_py_s_1d, dtype=np.bool), np.ndarray[tuple[int], np.dtype[np.bool]])

assert_type(np.strings.isdigit(_py_s_0d, out=_b_1d), np.ndarray[tuple[int], np.dtype[np.bool_]])
assert_type(np.strings.isdigit(_py_u_1d, out=_t_1d), np.ndarray[tuple[int], np.dtypes.StringDType])

assert_type(np.strings.isdigit.at(_s_1d, 1), None)
assert_type(np.strings.isdigit.at(_s_1d, (1, 1)), None)
assert_type(np.strings.isdigit.at(_u_1d, 1), None)
assert_type(np.strings.isdigit.at(_u_1d, (1, 1)), None)
assert_type(np.strings.isdigit.at(_t_1d, 1), None)
assert_type(np.strings.isdigit.at(_t_1d, (1, 1)), None)

# _ufunc_11_sut_i
# (str_len)

assert_type(np.strings.str_len.identity, Literal[0])

assert_type(np.strings.str_len(_py_s_0d), np.int_)
assert_type(np.strings.str_len(_py_u_0d), np.int_)
assert_type(np.strings.str_len(_py_s_0d, out=...), np.ndarray[tuple[()], np.dtype[np.int_]])
assert_type(np.strings.str_len(_py_u_0d, out=...), np.ndarray[tuple[()], np.dtype[np.int_]])
assert_type(np.strings.str_len(_py_s_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.strings.str_len(_py_u_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.strings.str_len(_py_s_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])
assert_type(np.strings.str_len(_py_u_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])

assert_type(np.strings.str_len(_s_0d), np.int_)
assert_type(np.strings.str_len(_u_0d), np.int_)
assert_type(np.strings.str_len(_s_0d, out=...), np.ndarray[tuple[()], np.dtype[np.int_]])
assert_type(np.strings.str_len(_u_0d, out=...), np.ndarray[tuple[()], np.dtype[np.int_]])
assert_type(np.strings.str_len(_s_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.strings.str_len(_u_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.strings.str_len(_t_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.strings.str_len(_s_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])
assert_type(np.strings.str_len(_u_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])
assert_type(np.strings.str_len(_t_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])
assert_type(np.strings.str_len(_s_nd), np.ndarray[_AnyShape, np.dtype[np.int_]])
assert_type(np.strings.str_len(_u_nd), np.ndarray[_AnyShape, np.dtype[np.int_]])
assert_type(np.strings.str_len(_t_nd), np.ndarray[_AnyShape, np.dtype[np.int_]])

assert_type(np.strings.str_len(_py_s_0d, out=_b_1d), np.ndarray[tuple[int], np.dtype[np.bool_]])
assert_type(np.strings.str_len(_py_u_1d, out=_t_1d), np.ndarray[tuple[int], np.dtypes.StringDType])

assert_type(np.strings.str_len.at(_s_1d, 1), None)
assert_type(np.strings.str_len.at(_s_1d, (1, 1)), None)
assert_type(np.strings.str_len.at(_u_1d, 1), None)
assert_type(np.strings.str_len.at(_u_1d, (1, 1)), None)
assert_type(np.strings.str_len.at(_t_1d, 1), None)
assert_type(np.strings.str_len.at(_t_1d, (1, 1)), None)
