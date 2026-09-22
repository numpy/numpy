from typing import Any, Literal, assert_type

import numpy as np
import numpy.typing as npt
from numpy._typing import _AnyShape

type AR_T_alias = np.ndarray[_AnyShape, np.dtypes.StringDType]
type AR_TU_alias = AR_T_alias | npt.NDArray[np.str_]

type _Array0D[ScalarT: np.generic] = np.ndarray[tuple[()], np.dtype[ScalarT]]
type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

type _tuple3[T] = tuple[T, T, T]

AR_U: npt.NDArray[np.str_]
AR_S: npt.NDArray[np.bytes_]
AR_T: AR_T_alias

_U_2d: _Array2D[np.str_]
_S_2d: _Array2D[np.bytes_]
_T_2d: np.ndarray[tuple[int, int], np.dtypes.StringDType]

###

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

assert_type(np.strings.add(AR_U, AR_U), npt.NDArray[np.str_])
assert_type(np.strings.add(AR_S, AR_S), npt.NDArray[np.bytes_])
assert_type(np.strings.add(AR_T, AR_T), AR_T_alias)

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

assert_type(np.strings.lstrip(_U_2d), _Array2D[np.str_])
assert_type(np.strings.lstrip(_T_2d, "_"), np.ndarray[tuple[int, int], np.dtypes.StringDType])
assert_type(np.strings.lstrip(_S_2d, b"_"), _Array2D[np.bytes_])
assert_type(np.strings.lstrip("_"), np.str_)
assert_type(np.strings.lstrip(b"_", b"_"), np.bytes_)
assert_type(np.strings.lstrip(["_"]), _Array1D[np.str_])
assert_type(np.strings.lstrip([b"_"], b"_"), _Array1D[np.bytes_])
assert_type(np.strings.lstrip([["_"]]), _Array2D[np.str_])
assert_type(np.strings.lstrip([[b"_"]], b"_"), _Array2D[np.bytes_])
assert_type(np.strings.lstrip(AR_U, AR_U), npt.NDArray[np.str_] | Any)
assert_type(np.strings.lstrip(AR_S, [b"_"]), npt.NDArray[np.bytes_] | Any)
assert_type(np.strings.lstrip(AR_T, AR_T), AR_T_alias)
assert_type(np.strings.lstrip("_", AR_T), AR_TU_alias | Any)

assert_type(np.strings.rstrip(_U_2d), _Array2D[np.str_])
assert_type(np.strings.rstrip(_T_2d, "_"), np.ndarray[tuple[int, int], np.dtypes.StringDType])
assert_type(np.strings.rstrip(_S_2d, b"_"), _Array2D[np.bytes_])
assert_type(np.strings.rstrip("_"), np.str_)
assert_type(np.strings.rstrip(b"_", b"_"), np.bytes_)
assert_type(np.strings.rstrip(["_"]), _Array1D[np.str_])
assert_type(np.strings.rstrip([b"_"], b"_"), _Array1D[np.bytes_])
assert_type(np.strings.rstrip([["_"]]), _Array2D[np.str_])
assert_type(np.strings.rstrip([[b"_"]], b"_"), _Array2D[np.bytes_])
assert_type(np.strings.rstrip(AR_U, AR_U), npt.NDArray[np.str_] | Any)
assert_type(np.strings.rstrip(AR_S, [b"_"]), npt.NDArray[np.bytes_] | Any)
assert_type(np.strings.rstrip(AR_T, AR_T), AR_T_alias)
assert_type(np.strings.rstrip("_", AR_T), AR_TU_alias | Any)

assert_type(np.strings.strip(_U_2d), _Array2D[np.str_])
assert_type(np.strings.strip(_T_2d, "_"), np.ndarray[tuple[int, int], np.dtypes.StringDType])
assert_type(np.strings.strip(_S_2d, b"_"), _Array2D[np.bytes_])
assert_type(np.strings.strip("_"), np.str_)
assert_type(np.strings.strip(b"_", b"_"), np.bytes_)
assert_type(np.strings.strip(["_"]), _Array1D[np.str_])
assert_type(np.strings.strip([b"_"], b"_"), _Array1D[np.bytes_])
assert_type(np.strings.strip([["_"]]), _Array2D[np.str_])
assert_type(np.strings.strip([[b"_"]], b"_"), _Array2D[np.bytes_])
assert_type(np.strings.strip(AR_U, AR_U), npt.NDArray[np.str_] | Any)
assert_type(np.strings.strip(AR_S, [b"_"]), npt.NDArray[np.bytes_] | Any)
assert_type(np.strings.strip(AR_T, AR_T), AR_T_alias)
assert_type(np.strings.strip("_", AR_T), AR_TU_alias | Any)

assert_type(np.strings.count(_U_2d, "_"), _Array2D[np.int_])
assert_type(np.strings.count(_T_2d, "_", 1), _Array2D[np.int_])
assert_type(np.strings.count(_S_2d, b"_", 1, 2), _Array2D[np.int_])
assert_type(np.strings.count("_", "_"), np.int_)
assert_type(np.strings.count(b"_", b"_", 1, 2), np.int_)
assert_type(np.strings.count(["_"], "_"), _Array1D[np.int_])
assert_type(np.strings.count([b"_"], b"_", 1, 2), _Array1D[np.int_])
assert_type(np.strings.count([["_"]], "_"), _Array2D[np.int_])
assert_type(np.strings.count([[b"_"]], b"_", 1, 2), _Array2D[np.int_])
assert_type(np.strings.count(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_] | Any)
assert_type(np.strings.count(AR_U, AR_T), npt.NDArray[np.int_] | Any)
assert_type(np.strings.count(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_] | Any)

assert_type(np.strings.partition(_U_2d, "_"), _tuple3[_Array2D[np.str_]])
assert_type(np.strings.partition(_S_2d, b"_"), _tuple3[_Array2D[np.bytes_]])
assert_type(np.strings.partition(AR_S, "_"), _tuple3[npt.NDArray[np.bytes_]])
assert_type(np.strings.partition(_T_2d, "_"), _tuple3[np.ndarray[tuple[int, int], np.dtypes.StringDType]])
assert_type(np.strings.partition("_", "_"), _tuple3[_Array0D[np.str_]])
assert_type(np.strings.partition("_", b"_"), _tuple3[_Array0D[np.str_]])
assert_type(np.strings.partition(b"_", b"_"), _tuple3[_Array0D[np.bytes_]])
assert_type(np.strings.partition(["_"], "_"), _tuple3[_Array1D[np.str_]])
assert_type(np.strings.partition([b"_"], b"_"), _tuple3[_Array1D[np.bytes_]])
assert_type(np.strings.partition([["_"]], "_"), _tuple3[_Array2D[np.str_]])
assert_type(np.strings.partition([[b"_"]], b"_"), _tuple3[_Array2D[np.bytes_]])
assert_type(np.strings.partition(AR_U, AR_U), _tuple3[npt.NDArray[np.str_]])
assert_type(np.strings.partition(AR_U, AR_S), _tuple3[npt.NDArray[np.str_]])
assert_type(np.strings.partition(AR_S, [b"a", b"b", b"c"]), _tuple3[npt.NDArray[np.bytes_]])
assert_type(np.strings.partition(AR_T, AR_T), _tuple3[AR_T_alias])
assert_type(np.strings.partition("_", AR_T), _tuple3[AR_TU_alias])

assert_type(np.strings.rpartition(_U_2d, "_"), _tuple3[_Array2D[np.str_]])
assert_type(np.strings.rpartition(_S_2d, b"_"), _tuple3[_Array2D[np.bytes_]])
assert_type(np.strings.rpartition(AR_S, "_"), _tuple3[npt.NDArray[np.bytes_]])
assert_type(np.strings.rpartition(_T_2d, "_"), _tuple3[np.ndarray[tuple[int, int], np.dtypes.StringDType]])
assert_type(np.strings.rpartition("_", "_"), _tuple3[_Array0D[np.str_]])
assert_type(np.strings.rpartition("_", b"_"), _tuple3[_Array0D[np.str_]])
assert_type(np.strings.rpartition(b"_", b"_"), _tuple3[_Array0D[np.bytes_]])
assert_type(np.strings.rpartition(["_"], "_"), _tuple3[_Array1D[np.str_]])
assert_type(np.strings.rpartition([b"_"], b"_"), _tuple3[_Array1D[np.bytes_]])
assert_type(np.strings.rpartition([["_"]], "_"), _tuple3[_Array2D[np.str_]])
assert_type(np.strings.rpartition([[b"_"]], b"_"), _tuple3[_Array2D[np.bytes_]])
assert_type(np.strings.rpartition(AR_U, AR_U), _tuple3[npt.NDArray[np.str_]])
assert_type(np.strings.rpartition(AR_U, AR_S), _tuple3[npt.NDArray[np.str_]])
assert_type(np.strings.rpartition(AR_S, [b"a", b"b", b"c"]), _tuple3[npt.NDArray[np.bytes_]])
assert_type(np.strings.rpartition(AR_T, AR_T), _tuple3[AR_T_alias])
assert_type(np.strings.rpartition("_", AR_T), _tuple3[AR_TU_alias])

assert_type(np.strings.replace(_U_2d, "_", "-"), _Array2D[np.str_])
assert_type(np.strings.replace(_S_2d, b"_", b"-"), _Array2D[np.bytes_])
assert_type(np.strings.replace(AR_S, "_", "-", 1), npt.NDArray[np.bytes_])
assert_type(np.strings.replace(_T_2d, "_", "_"), np.ndarray[tuple[int, int], np.dtypes.StringDType])
assert_type(np.strings.replace("_", "_", "-"), _Array0D[np.str_])
assert_type(np.strings.replace("_", b"_", b"-"), _Array0D[np.str_])
assert_type(np.strings.replace(b"_", b"_", b"-"), _Array0D[np.bytes_])
assert_type(np.strings.replace(["_"], "_", "-"), _Array1D[np.str_])
assert_type(np.strings.replace([b"_"], b"_", b"-"), _Array1D[np.bytes_])
assert_type(np.strings.replace([["_"]], "_", "-"), _Array2D[np.str_])
assert_type(np.strings.replace([[b"_"]], b"_", b"-"), _Array2D[np.bytes_])
assert_type(np.strings.replace(AR_U, AR_U, "-"), npt.NDArray[np.str_])
assert_type(np.strings.replace(AR_U, [b"_"], "-"), npt.NDArray[np.str_])
assert_type(np.strings.replace(AR_S, [b"_", b""], [b"a", b"b"]), npt.NDArray[np.bytes_])
assert_type(np.strings.replace(AR_T, AR_T, AR_T), AR_T_alias)
assert_type(np.strings.replace(AR_T, "_", AR_T), AR_TU_alias)

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

assert_type(np.strings.startswith(_U_2d, "_"), _Array2D[np.bool])
assert_type(np.strings.startswith(_T_2d, "_", 1), _Array2D[np.bool])
assert_type(np.strings.startswith(_S_2d, b"_", 1, 2), _Array2D[np.bool])
assert_type(np.strings.startswith("_", "_"), np.bool)
assert_type(np.strings.startswith(b"_", b"_", 1, 2), np.bool)
assert_type(np.strings.startswith(["_"], "_"), _Array1D[np.bool])
assert_type(np.strings.startswith([b"_"], b"_", 1, 2), _Array1D[np.bool])
assert_type(np.strings.startswith([["_"]], "_"), _Array2D[np.bool])
assert_type(np.strings.startswith([[b"_"]], b"_", 1, 2), _Array2D[np.bool])
assert_type(np.strings.startswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool] | Any)
assert_type(np.strings.startswith(AR_U, AR_T), npt.NDArray[np.bool] | Any)
assert_type(np.strings.startswith(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool] | Any)

assert_type(np.strings.endswith(_U_2d, "_"), _Array2D[np.bool])
assert_type(np.strings.endswith(_T_2d, "_", 1), _Array2D[np.bool])
assert_type(np.strings.endswith(_S_2d, b"_", 1, 2), _Array2D[np.bool])
assert_type(np.strings.endswith("_", "_"), np.bool)
assert_type(np.strings.endswith(b"_", b"_", 1, 2), np.bool)
assert_type(np.strings.endswith(["_"], "_"), _Array1D[np.bool])
assert_type(np.strings.endswith([b"_"], b"_", 1, 2), _Array1D[np.bool])
assert_type(np.strings.endswith([["_"]], "_"), _Array2D[np.bool])
assert_type(np.strings.endswith([[b"_"]], b"_", 1, 2), _Array2D[np.bool])
assert_type(np.strings.endswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool] | Any)
assert_type(np.strings.endswith(AR_U, AR_T), npt.NDArray[np.bool] | Any)
assert_type(np.strings.endswith(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool] | Any)

assert_type(np.strings.find(_U_2d, "_"), _Array2D[np.int_])
assert_type(np.strings.find(_T_2d, "_", 1), _Array2D[np.int_])
assert_type(np.strings.find(_S_2d, b"_", 1, 2), _Array2D[np.int_])
assert_type(np.strings.find("_", "_"), np.int_)
assert_type(np.strings.find(b"_", b"_", 1, 2), np.int_)
assert_type(np.strings.find(["_"], "_"), _Array1D[np.int_])
assert_type(np.strings.find([b"_"], b"_", 1, 2), _Array1D[np.int_])
assert_type(np.strings.find([["_"]], "_"), _Array2D[np.int_])
assert_type(np.strings.find([[b"_"]], b"_", 1, 2), _Array2D[np.int_])
assert_type(np.strings.find(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_] | Any)
assert_type(np.strings.find(AR_U, AR_T), npt.NDArray[np.int_] | Any)
assert_type(np.strings.find(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_] | Any)

assert_type(np.strings.rfind(_U_2d, "_"), _Array2D[np.int_])
assert_type(np.strings.rfind(_T_2d, "_", 1), _Array2D[np.int_])
assert_type(np.strings.rfind(_S_2d, b"_", 1, 2), _Array2D[np.int_])
assert_type(np.strings.rfind("_", "_"), np.int_)
assert_type(np.strings.rfind(b"_", b"_", 1, 2), np.int_)
assert_type(np.strings.rfind(["_"], "_"), _Array1D[np.int_])
assert_type(np.strings.rfind([b"_"], b"_", 1, 2), _Array1D[np.int_])
assert_type(np.strings.rfind([["_"]], "_"), _Array2D[np.int_])
assert_type(np.strings.rfind([[b"_"]], b"_", 1, 2), _Array2D[np.int_])
assert_type(np.strings.rfind(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_] | Any)
assert_type(np.strings.rfind(AR_U, AR_T), npt.NDArray[np.int_] | Any)
assert_type(np.strings.rfind(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_] | Any)

assert_type(np.strings.index(_U_2d, "_"), _Array2D[np.int_])
assert_type(np.strings.index(_T_2d, "_", 1), _Array2D[np.int_])
assert_type(np.strings.index(_S_2d, b"_", 1, 2), _Array2D[np.int_])
assert_type(np.strings.index("_", "_"), np.int_)
assert_type(np.strings.index(b"_", b"_", 1, 2), np.int_)
assert_type(np.strings.index(["_"], "_"), _Array1D[np.int_])
assert_type(np.strings.index([b"_"], b"_", 1, 2), _Array1D[np.int_])
assert_type(np.strings.index([["_"]], "_"), _Array2D[np.int_])
assert_type(np.strings.index([[b"_"]], b"_", 1, 2), _Array2D[np.int_])
assert_type(np.strings.index(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_] | Any)
assert_type(np.strings.index(AR_U, AR_T), npt.NDArray[np.int_] | Any)
assert_type(np.strings.index(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_] | Any)

assert_type(np.strings.rindex(_U_2d, "_"), _Array2D[np.int_])
assert_type(np.strings.rindex(_T_2d, "_", 1), _Array2D[np.int_])
assert_type(np.strings.rindex(_S_2d, b"_", 1, 2), _Array2D[np.int_])
assert_type(np.strings.rindex("_", "_"), np.int_)
assert_type(np.strings.rindex(b"_", b"_", 1, 2), np.int_)
assert_type(np.strings.rindex(["_"], "_"), _Array1D[np.int_])
assert_type(np.strings.rindex([b"_"], b"_", 1, 2), _Array1D[np.int_])
assert_type(np.strings.rindex([["_"]], "_"), _Array2D[np.int_])
assert_type(np.strings.rindex([[b"_"]], b"_", 1, 2), _Array2D[np.int_])
assert_type(np.strings.rindex(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_] | Any)
assert_type(np.strings.rindex(AR_U, AR_T), npt.NDArray[np.int_] | Any)
assert_type(np.strings.rindex(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_] | Any)

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
