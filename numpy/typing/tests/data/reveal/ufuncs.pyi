from typing import Any, Literal, NoReturn, assert_type

import numpy as np
import numpy.typing as npt

i8: np.int64
f8: np.float64
AR_f8: npt.NDArray[np.float64]
AR_i8: npt.NDArray[np.int64]

assert_type(np.add.__name__, Literal["add"])
assert_type(np.add.__qualname__, Literal["add"])
assert_type(np.add.ntypes, Literal[22])
assert_type(np.add.identity, Literal[0])
assert_type(np.add.nin, Literal[2])
assert_type(np.add.nout, Literal[1])
assert_type(np.add.nargs, Literal[3])
assert_type(np.add.signature, None)
assert_type(np.add(f8, f8), Any)
assert_type(np.add(AR_f8, f8), npt.NDArray[Any])
assert_type(np.add.at(AR_f8, AR_i8, f8), None)
assert_type(np.add.reduce(AR_f8, axis=0), Any)
assert_type(np.add.accumulate(AR_f8), npt.NDArray[Any])
assert_type(np.add.reduceat(AR_f8, AR_i8), npt.NDArray[Any])
assert_type(np.add.outer(f8, f8), Any)
assert_type(np.add.outer(AR_f8, f8), npt.NDArray[Any])

assert_type(np.divmod.__name__, Literal["divmod"])
assert_type(np.divmod.__qualname__, Literal["divmod"])
assert_type(np.divmod.ntypes, Literal[15])
assert_type(np.divmod.identity, None)
assert_type(np.divmod.nin, Literal[2])
assert_type(np.divmod.nout, Literal[2])
assert_type(np.divmod.nargs, Literal[4])
assert_type(np.divmod.signature, None)
assert_type(np.divmod(f8, f8), tuple[Any, Any])
assert_type(np.divmod(AR_f8, f8), tuple[npt.NDArray[Any], npt.NDArray[Any]])

assert_type(np.matmul.__name__, Literal["matmul"])
assert_type(np.matmul.__qualname__, Literal["matmul"])
assert_type(np.matmul.ntypes, Literal[19])
assert_type(np.matmul.identity, None)
assert_type(np.matmul.nin, Literal[2])
assert_type(np.matmul.nout, Literal[1])
assert_type(np.matmul.nargs, Literal[3])
assert_type(np.matmul.signature, Literal["(n?,k),(k,m?)->(n?,m?)"])
assert_type(np.matmul.identity, None)
assert_type(np.matmul(AR_f8, AR_f8), Any)
assert_type(np.matmul(AR_f8, AR_f8, axes=[(0, 1), (0, 1), (0, 1)]), Any)

assert_type(np.vecdot.__name__, Literal["vecdot"])
assert_type(np.vecdot.__qualname__, Literal["vecdot"])
assert_type(np.vecdot.ntypes, Literal[19])
assert_type(np.vecdot.identity, None)
assert_type(np.vecdot.nin, Literal[2])
assert_type(np.vecdot.nout, Literal[1])
assert_type(np.vecdot.nargs, Literal[3])
assert_type(np.vecdot.signature, Literal["(n),(n)->()"])
assert_type(np.vecdot.identity, None)
assert_type(np.vecdot(AR_f8, AR_f8), Any)

def test_absolute_outer_invalid() -> None:
    assert_type(np.absolute.outer(AR_f8, AR_f8), NoReturn)  # type: ignore[arg-type]
def test_frexp_outer_invalid() -> None:
    assert_type(np.frexp.outer(AR_f8, AR_f8), NoReturn)  # type: ignore[arg-type]
def test_divmod_outer_invalid() -> None:
    assert_type(np.divmod.outer(AR_f8, AR_f8), NoReturn)  # type: ignore[arg-type]
def test_matmul_outer_invalid() -> None:
    assert_type(np.matmul.outer(AR_f8, AR_f8), NoReturn)  # type: ignore[arg-type]

def test_absolute_reduceat_invalid() -> None:
    assert_type(np.absolute.reduceat(AR_f8, AR_i8), NoReturn)  # type: ignore[arg-type]
def test_frexp_reduceat_invalid() -> None:
    assert_type(np.frexp.reduceat(AR_f8, AR_i8), NoReturn)  # type: ignore[arg-type]
def test_divmod_reduceat_invalid() -> None:
    assert_type(np.divmod.reduceat(AR_f8, AR_i8), NoReturn)  # type: ignore[arg-type]
def test_matmul_reduceat_invalid() -> None:
    assert_type(np.matmul.reduceat(AR_f8, AR_i8), NoReturn)  # type: ignore[arg-type]

def test_absolute_reduce_invalid() -> None:
    assert_type(np.absolute.reduce(AR_f8), NoReturn)  # type: ignore[arg-type]
def test_frexp_reduce_invalid() -> None:
    assert_type(np.frexp.reduce(AR_f8), NoReturn)  # type: ignore[arg-type]
def test_divmod_reduce_invalid() -> None:
    assert_type(np.divmod.reduce(AR_f8), NoReturn)  # type: ignore[arg-type]
def test_matmul_reduce_invalid() -> None:
    assert_type(np.matmul.reduce(AR_f8), NoReturn)  # type: ignore[arg-type]

def test_absolute_accumulate_invalid() -> None:
    assert_type(np.absolute.accumulate(AR_f8), NoReturn)  # type: ignore[arg-type]
def test_frexp_accumulate_invalid() -> None:
    assert_type(np.frexp.accumulate(AR_f8), NoReturn)  # type: ignore[arg-type]
def test_divmod_accumulate_invalid() -> None:
    assert_type(np.divmod.accumulate(AR_f8), NoReturn)  # type: ignore[arg-type]
def test_matmul_accumulate_invalid() -> None:
    assert_type(np.matmul.accumulate(AR_f8), NoReturn)  # type: ignore[arg-type]

def test_frexp_at_invalid() -> None:
    assert_type(np.frexp.at(AR_f8, i8), NoReturn)  # type: ignore[arg-type]
def test_divmod_at_invalid() -> None:
    assert_type(np.divmod.at(AR_f8, i8, AR_f8), NoReturn)  # type: ignore[arg-type]
def test_matmul_at_invalid() -> None:
    assert_type(np.matmul.at(AR_f8, i8, AR_f8), NoReturn)  # type: ignore[arg-type]

###

_py_b_0d: bool
_py_b_1d: list[bool]
_py_b_2d: list[list[bool]]
_py_i_0d: int
_py_i_1d: list[int]
_py_i_2d: list[list[int]]
_py_f_0d: float
_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_c_0d: complex
_py_c_1d: list[complex]
_py_c_2d: list[list[complex]]

_bool_0d: np.bool
_bool_1d: np.ndarray[tuple[int], np.dtype[np.bool]]
_bool_2d: np.ndarray[tuple[int, int], np.dtype[np.bool]]
_bool_nd: npt.NDArray[np.bool]
_u8_0d: np.uint8
_u8_1d: np.ndarray[tuple[int], np.dtype[np.uint8]]
_u8_2d: np.ndarray[tuple[int, int], np.dtype[np.uint8]]
_u8_nd: npt.NDArray[np.uint8]
_i16_0d: np.int16
_i16_1d: np.ndarray[tuple[int], np.dtype[np.int16]]
_i16_2d: np.ndarray[tuple[int, int], np.dtype[np.int16]]
_i16_nd: npt.NDArray[np.int16]
_f32_0d: np.float32
_f32_1d: np.ndarray[tuple[int], np.dtype[np.float32]]
_f32_2d: np.ndarray[tuple[int, int], np.dtype[np.float32]]
_f32_nd: npt.NDArray[np.float32]
_c64_0d: np.complex64
_c64_1d: np.ndarray[tuple[int], np.dtype[np.complex64]]
_c64_2d: np.ndarray[tuple[int, int], np.dtype[np.complex64]]
_c64_nd: npt.NDArray[np.complex64]
_dt_ns_0d: np.datetime64[int]
_dt_ns_1d: np.ndarray[tuple[int], np.dtype[np.datetime64[int]]]
_dt_ns_2d: np.ndarray[tuple[int, int], np.dtype[np.datetime64[int]]]
_dt_ns_nd: npt.NDArray[np.datetime64[int]]
_td_ns_0d: np.timedelta64[int]
_td_ns_1d: np.ndarray[tuple[int], np.dtype[np.timedelta64[int]]]
_td_ns_2d: np.ndarray[tuple[int, int], np.dtype[np.timedelta64[int]]]
_td_ns_nd: npt.NDArray[np.timedelta64[int]]
_obj_1d: np.ndarray[tuple[int], np.dtype[np.object_]]
_obj_2d: np.ndarray[tuple[int, int], np.dtype[np.object_]]
_obj_nd: npt.NDArray[np.object_]

# _ufunc_11_m_b
# (isnat)

assert_type(np.isnat(_dt_ns_0d), np.bool)
assert_type(np.isnat(_dt_ns_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isnat(_dt_ns_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isnat(_dt_ns_nd), npt.NDArray[np.bool])
assert_type(np.isnat(_td_ns_0d), np.bool)
assert_type(np.isnat(_td_ns_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isnat(_td_ns_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isnat(_td_ns_nd), npt.NDArray[np.bool])

assert_type(np.isnat(_dt_ns_2d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# _ufunc_11_f_b
# (signbit)

assert_type(np.signbit(_py_b_0d), np.bool)
assert_type(np.signbit(_py_b_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.signbit(_py_b_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.signbit(_py_i_0d), np.bool)
assert_type(np.signbit(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.signbit(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.signbit(_py_f_0d), np.bool)
assert_type(np.signbit(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.signbit(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])

assert_type(np.signbit(_i16_0d), np.bool)
assert_type(np.signbit(_i16_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.signbit(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.signbit(_i16_nd), npt.NDArray[np.bool])
assert_type(np.signbit(_f32_0d), np.bool)
assert_type(np.signbit(_f32_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.signbit(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.signbit(_f32_nd), npt.NDArray[np.bool])

assert_type(np.signbit(_py_i_2d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# _ufunc_11_bifgco_bo
# (logical_not)

assert_type(np.logical_not(_py_b_0d), np.bool)
assert_type(np.logical_not(_py_b_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.logical_not(_py_b_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.logical_not(_py_i_0d), np.bool)
assert_type(np.logical_not(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.logical_not(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.logical_not(_py_f_0d), np.bool)
assert_type(np.logical_not(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.logical_not(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.logical_not(_py_c_0d), np.bool)
assert_type(np.logical_not(_py_c_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.logical_not(_py_c_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])

assert_type(np.logical_not(_bool_0d), np.bool)
assert_type(np.logical_not(_bool_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.logical_not(_bool_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.logical_not(_bool_nd), npt.NDArray[np.bool])
assert_type(np.logical_not(_i16_0d), np.bool)
assert_type(np.logical_not(_i16_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.logical_not(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.logical_not(_i16_nd), npt.NDArray[np.bool])
assert_type(np.logical_not(_f32_0d), np.bool)
assert_type(np.logical_not(_f32_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.logical_not(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.logical_not(_f32_nd), npt.NDArray[np.bool])
assert_type(np.logical_not(_c64_0d), np.bool)
assert_type(np.logical_not(_c64_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.logical_not(_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.logical_not(_c64_nd), npt.NDArray[np.bool])
assert_type(np.logical_not(_obj_1d), np.ndarray[tuple[int], np.dtype[np.object_]])
assert_type(np.logical_not(_obj_2d), np.ndarray[tuple[int, int], np.dtype[np.object_]])
assert_type(np.logical_not(_obj_nd), npt.NDArray[np.object_])

assert_type(np.logical_not(_py_i_2d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# _ufunc_11_bifgcm_b
# (isfinite, isinf, isnan)

assert_type(np.isinf(_py_b_0d), np.bool)
assert_type(np.isinf(_py_b_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isinf(_py_b_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isinf(_py_i_0d), np.bool)
assert_type(np.isinf(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isinf(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isinf(_py_f_0d), np.bool)
assert_type(np.isinf(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isinf(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isinf(_py_c_0d), np.bool)
assert_type(np.isinf(_py_c_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isinf(_py_c_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])

assert_type(np.isinf(_bool_0d), np.bool)
assert_type(np.isinf(_bool_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isinf(_bool_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isinf(_bool_nd), npt.NDArray[np.bool])
assert_type(np.isinf(_i16_0d), np.bool)
assert_type(np.isinf(_i16_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isinf(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isinf(_i16_nd), npt.NDArray[np.bool])
assert_type(np.isinf(_f32_0d), np.bool)
assert_type(np.isinf(_f32_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isinf(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isinf(_f32_nd), npt.NDArray[np.bool])
assert_type(np.isinf(_c64_0d), np.bool)
assert_type(np.isinf(_c64_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isinf(_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isinf(_c64_nd), npt.NDArray[np.bool])
assert_type(np.isinf(_dt_ns_0d), np.bool)
assert_type(np.isinf(_dt_ns_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isinf(_dt_ns_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isinf(_dt_ns_nd), npt.NDArray[np.bool])
assert_type(np.isinf(_td_ns_0d), np.bool)
assert_type(np.isinf(_td_ns_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isinf(_td_ns_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isinf(_td_ns_nd), npt.NDArray[np.bool])

assert_type(np.isinf(_py_i_2d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# _ufunc_11_io
# (bitwise_count)

assert_type(np.bitwise_count(_py_b_0d), np.uint8)
assert_type(np.bitwise_count(_py_b_1d), np.ndarray[tuple[int], np.dtype[np.uint8]])
assert_type(np.bitwise_count(_py_b_2d), np.ndarray[tuple[int, int], np.dtype[np.uint8]])
assert_type(np.bitwise_count(_py_i_0d), np.uint8)
assert_type(np.bitwise_count(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.uint8]])
assert_type(np.bitwise_count(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.uint8]])

assert_type(np.bitwise_count(_bool_0d), np.uint8)
assert_type(np.bitwise_count(_bool_1d), np.ndarray[tuple[int], np.dtype[np.uint8]])
assert_type(np.bitwise_count(_bool_2d), np.ndarray[tuple[int, int], np.dtype[np.uint8]])
assert_type(np.bitwise_count(_bool_nd), npt.NDArray[np.uint8])
assert_type(np.bitwise_count(_i16_0d), np.uint8)
assert_type(np.bitwise_count(_i16_1d), np.ndarray[tuple[int], np.dtype[np.uint8]])
assert_type(np.bitwise_count(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.uint8]])
assert_type(np.bitwise_count(_i16_nd), npt.NDArray[np.uint8])

assert_type(np.bitwise_count(_py_i_2d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# _ufunc_11_f
# (spacing)

assert_type(np.spacing(_py_i_0d), np.float64)
assert_type(np.spacing(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.spacing(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.spacing(_py_f_0d), np.float64)
assert_type(np.spacing(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.spacing(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])

assert_type(np.spacing(_i16_0d), np.float64)
assert_type(np.spacing(_i16_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.spacing(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.spacing(_i16_nd), npt.NDArray[np.float64])
assert_type(np.spacing(_f32_0d), np.float32)
assert_type(np.spacing(_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(np.spacing(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.spacing(_f32_nd), npt.NDArray[np.float32])

assert_type(np.spacing(_py_i_0d, dtype=np.float32), np.float32)
assert_type(np.spacing(_py_i_1d, dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.spacing(_i16_2d, dtype=np.float32), np.ndarray[tuple[int, int], np.dtype[np.float32]])

assert_type(np.spacing(_py_i_0d, dtype="f4"), Any)
assert_type(np.spacing(_py_i_1d, dtype="f4"), np.ndarray)
assert_type(np.spacing(_i16_2d, dtype="f4"), np.ndarray[tuple[int, int]])

assert_type(np.spacing(_py_i_2d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# _ufunc_11_fo
# (cbrt, deg2rad, degrees, fabs, rad2deg, radians)

assert_type(np.cbrt(_py_i_0d), np.float64)
assert_type(np.cbrt(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.cbrt(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.cbrt(_py_f_0d), np.float64)
assert_type(np.cbrt(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.cbrt(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])

assert_type(np.cbrt(_i16_0d), np.float64)
assert_type(np.cbrt(_i16_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.cbrt(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.cbrt(_i16_nd), npt.NDArray[np.float64])
assert_type(np.cbrt(_f32_0d), np.float32)
assert_type(np.cbrt(_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(np.cbrt(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.cbrt(_f32_nd), npt.NDArray[np.float32])
assert_type(np.cbrt(_obj_1d), np.ndarray[tuple[int], np.dtype[np.object_]])
assert_type(np.cbrt(_obj_2d), np.ndarray[tuple[int, int], np.dtype[np.object_]])
assert_type(np.cbrt(_obj_nd), npt.NDArray[np.object_])

assert_type(np.cbrt(_py_i_0d, dtype=np.float32), np.float32)
assert_type(np.cbrt(_py_i_1d, dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.cbrt(_i16_2d, dtype=np.float32), np.ndarray[tuple[int, int], np.dtype[np.float32]])

assert_type(np.cbrt(_py_i_0d, dtype="f4"), Any)
assert_type(np.cbrt(_py_i_1d, dtype="f4"), np.ndarray)
assert_type(np.cbrt(_i16_2d, dtype="f4"), np.ndarray[tuple[int, int]])

assert_type(np.cbrt(_py_i_2d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# _ufunc_11_fco

assert_type(np.sin(_py_i_0d), np.float64)
assert_type(np.sin(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.sin(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.sin(_py_f_0d), np.float64)
assert_type(np.sin(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.sin(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.sin(_py_c_0d), np.complex128 | Any)  # `complex` overlaps with `float`, hence the `Any`
assert_type(np.sin(_py_c_1d), np.ndarray[tuple[int], np.dtype[np.complex128]])
assert_type(np.sin(_py_c_2d), np.ndarray[tuple[int, int], np.dtype[np.complex128]])

assert_type(np.sin(_i16_0d), np.float64)
assert_type(np.sin(_i16_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.sin(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.sin(_i16_nd), npt.NDArray[np.float64])
assert_type(np.sin(_f32_0d), np.float32)
assert_type(np.sin(_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(np.sin(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.sin(_f32_nd), npt.NDArray[np.float32])
assert_type(np.sin(_c64_0d), np.complex64)
assert_type(np.sin(_c64_1d), np.ndarray[tuple[int], np.dtype[np.complex64]])
assert_type(np.sin(_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.complex64]])
assert_type(np.sin(_c64_nd), npt.NDArray[np.complex64])
assert_type(np.sin(_obj_1d), np.ndarray[tuple[int], np.dtype[np.object_]])
assert_type(np.sin(_obj_2d), np.ndarray[tuple[int, int], np.dtype[np.object_]])
assert_type(np.sin(_obj_nd), npt.NDArray[np.object_])

assert_type(np.sin(_py_i_0d, dtype=np.float32), np.float32)
assert_type(np.sin(_py_i_1d, dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.sin(_i16_2d, dtype=np.float32), np.ndarray[tuple[int, int], np.dtype[np.float32]])

assert_type(np.sin(_py_i_0d, dtype="f4"), Any)
assert_type(np.sin(_py_i_1d, dtype="f4"), np.ndarray)
assert_type(np.sin(_i16_2d, dtype="f4"), np.ndarray[tuple[int, int]])

assert_type(np.sin(_py_i_2d, out=_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.complex64]])

# _ufunc_11_ifco
# (conj[ugate], reciprocal, square)

assert_type(np.square(_py_b_0d), np.int8)
assert_type(np.square(_py_b_1d), np.ndarray[tuple[int], np.dtype[np.int8]])
assert_type(np.square(_py_b_2d), np.ndarray[tuple[int, int], np.dtype[np.int8]])
assert_type(np.square(_py_i_0d), np.int_ | Any)  # `int` overlaps with `bool`
assert_type(np.square(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.square(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])
assert_type(np.square(_py_f_0d), np.float64 | Any)  # `complex` overlaps with `int`
assert_type(np.square(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.square(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.square(_py_c_0d), np.complex128 | Any)  # `complex` overlaps with `float`
assert_type(np.square(_py_c_1d), np.ndarray[tuple[int], np.dtype[np.complex128]])
assert_type(np.square(_py_c_2d), np.ndarray[tuple[int, int], np.dtype[np.complex128]])

assert_type(np.square(_bool_0d), np.int8)
assert_type(np.square(_bool_1d), np.ndarray[tuple[int], np.dtype[np.int8]])
assert_type(np.square(_bool_2d), np.ndarray[tuple[int, int], np.dtype[np.int8]])
assert_type(np.square(_bool_nd), npt.NDArray[np.int8])
assert_type(np.square(_i16_0d), np.int16)
assert_type(np.square(_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.square(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.int16]])
assert_type(np.square(_i16_nd), npt.NDArray[np.int16])
assert_type(np.square(_f32_0d), np.float32)
assert_type(np.square(_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(np.square(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.square(_f32_nd), npt.NDArray[np.float32])
assert_type(np.square(_c64_0d), np.complex64)
assert_type(np.square(_c64_1d), np.ndarray[tuple[int], np.dtype[np.complex64]])
assert_type(np.square(_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.complex64]])
assert_type(np.square(_c64_nd), npt.NDArray[np.complex64])
assert_type(np.square(_obj_1d), np.ndarray[tuple[int], np.dtype[np.object_]])
assert_type(np.square(_obj_2d), np.ndarray[tuple[int, int], np.dtype[np.object_]])
assert_type(np.square(_obj_nd), npt.NDArray[np.object_])

assert_type(np.square(_py_b_0d, dtype=np.float32), np.float32)
assert_type(np.square(_py_i_1d, dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.square(_i16_2d, dtype=np.float32), np.ndarray[tuple[int, int], np.dtype[np.float32]])

assert_type(np.square(_py_b_0d, dtype="f4"), Any)
assert_type(np.square(_py_i_1d, dtype="f4"), np.ndarray)
assert_type(np.square(_i16_2d, dtype="f4"), np.ndarray[tuple[int, int]])

assert_type(np.square(_py_b_0d, out=_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.square(_py_i_1d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.square(_py_f_2d, out=_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.complex64]])

# _ufunc_11_ifcmo_ifco
# (sign)

assert_type(np.sign(_py_i_0d), np.int_)
assert_type(np.sign(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.sign(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])
assert_type(np.sign(_py_f_0d), np.float64 | Any)  # `complex` overlaps with `int`
assert_type(np.sign(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.sign(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.sign(_py_c_0d), np.complex128 | Any)  # `complex` overlaps with `float`
assert_type(np.sign(_py_c_1d), np.ndarray[tuple[int], np.dtype[np.complex128]])
assert_type(np.sign(_py_c_2d), np.ndarray[tuple[int, int], np.dtype[np.complex128]])

assert_type(np.sign(_i16_0d), np.int16)
assert_type(np.sign(_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.sign(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.int16]])
assert_type(np.sign(_i16_nd), npt.NDArray[np.int16])
assert_type(np.sign(_f32_0d), np.float32)
assert_type(np.sign(_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(np.sign(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.sign(_f32_nd), npt.NDArray[np.float32])
assert_type(np.sign(_c64_0d), np.complex64)
assert_type(np.sign(_c64_1d), np.ndarray[tuple[int], np.dtype[np.complex64]])
assert_type(np.sign(_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.complex64]])
assert_type(np.sign(_c64_nd), npt.NDArray[np.complex64])
assert_type(np.sign(_td_ns_0d), np.float64)
assert_type(np.sign(_td_ns_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.sign(_td_ns_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.sign(_td_ns_nd), npt.NDArray[np.float64])
assert_type(np.sign(_obj_1d), np.ndarray[tuple[int], np.dtype[np.object_]])
assert_type(np.sign(_obj_2d), np.ndarray[tuple[int, int], np.dtype[np.object_]])
assert_type(np.sign(_obj_nd), npt.NDArray[np.object_])

assert_type(np.sign(_py_i_0d, dtype=np.float32), np.float32)
assert_type(np.sign(_py_f_1d, dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.sign(_i16_2d, dtype=np.float32), np.ndarray[tuple[int, int], np.dtype[np.float32]])

assert_type(np.sign(_py_i_0d, dtype="f4"), Any)
assert_type(np.sign(_py_f_1d, dtype="f4"), np.ndarray)
assert_type(np.sign(_i16_2d, dtype="f4"), np.ndarray[tuple[int, int]])

assert_type(np.sign(_py_i_0d, out=_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.sign(_py_f_1d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.sign(_py_c_2d, out=_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.complex64]])

# _ufunc_11_ifcmo
# (positive, negative)

assert_type(np.negative(_py_i_0d), np.int_)
assert_type(np.negative(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.negative(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])
assert_type(np.negative(_py_f_0d), np.float64 | Any)  # `complex` overlaps with `int`
assert_type(np.negative(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.negative(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.negative(_py_c_0d), np.complex128 | Any)  # `complex` overlaps with `float`
assert_type(np.negative(_py_c_1d), np.ndarray[tuple[int], np.dtype[np.complex128]])
assert_type(np.negative(_py_c_2d), np.ndarray[tuple[int, int], np.dtype[np.complex128]])

assert_type(np.negative(_i16_0d), np.int16)
assert_type(np.negative(_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.negative(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.int16]])
assert_type(np.negative(_i16_nd), npt.NDArray[np.int16])
assert_type(np.negative(_f32_0d), np.float32)
assert_type(np.negative(_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(np.negative(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.negative(_f32_nd), npt.NDArray[np.float32])
assert_type(np.negative(_c64_0d), np.complex64)
assert_type(np.negative(_c64_1d), np.ndarray[tuple[int], np.dtype[np.complex64]])
assert_type(np.negative(_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.complex64]])
assert_type(np.negative(_c64_nd), npt.NDArray[np.complex64])
assert_type(np.negative(_td_ns_0d), np.timedelta64[int])
assert_type(np.negative(_td_ns_1d), np.ndarray[tuple[int], np.dtype[np.timedelta64[int]]])
assert_type(np.negative(_td_ns_2d), np.ndarray[tuple[int, int], np.dtype[np.timedelta64[int]]])
assert_type(np.negative(_td_ns_nd), npt.NDArray[np.timedelta64[int]])
assert_type(np.negative(_obj_1d), np.ndarray[tuple[int], np.dtype[np.object_]])
assert_type(np.negative(_obj_2d), np.ndarray[tuple[int, int], np.dtype[np.object_]])
assert_type(np.negative(_obj_nd), npt.NDArray[np.object_])

assert_type(np.negative(_py_i_0d, dtype=np.float32), np.float32)
assert_type(np.negative(_py_f_1d, dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.negative(_i16_2d, dtype=np.float32), np.ndarray[tuple[int, int], np.dtype[np.float32]])

assert_type(np.negative(_py_i_0d, dtype="f4"), Any)
assert_type(np.negative(_py_f_1d, dtype="f4"), np.ndarray)
assert_type(np.negative(_i16_2d, dtype="f4"), np.ndarray[tuple[int, int]])

assert_type(np.negative(_py_i_0d, out=_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.negative(_py_f_1d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.negative(_py_c_2d, out=_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.complex64]])

# _ufunc_11_bio
# (invert)

assert_type(np.invert(_py_b_0d), np.bool)
assert_type(np.invert(_py_b_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.invert(_py_b_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.invert(_py_i_0d), np.int_ | Any)
assert_type(np.invert(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.invert(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])

assert_type(np.invert(_bool_0d), np.bool)
assert_type(np.invert(_bool_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.invert(_bool_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.invert(_bool_nd), npt.NDArray[np.bool])
assert_type(np.invert(_u8_0d), np.uint8)
assert_type(np.invert(_u8_1d), np.ndarray[tuple[int], np.dtype[np.uint8]])
assert_type(np.invert(_u8_2d), np.ndarray[tuple[int, int], np.dtype[np.uint8]])
assert_type(np.invert(_u8_nd), npt.NDArray[np.uint8])
assert_type(np.invert(_i16_0d), np.int16)
assert_type(np.invert(_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.invert(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.int16]])
assert_type(np.invert(_i16_nd), npt.NDArray[np.int16])
assert_type(np.invert(_obj_1d), np.ndarray[tuple[int], np.dtype[np.object_]])
assert_type(np.invert(_obj_2d), np.ndarray[tuple[int, int], np.dtype[np.object_]])
assert_type(np.invert(_obj_nd), npt.NDArray[np.object_])

assert_type(np.invert(_py_b_0d, dtype=np.uint8), np.uint8)
assert_type(np.invert(_py_i_1d, dtype=np.int16), npt.NDArray[np.int16])
assert_type(np.invert(_u8_2d, dtype=np.bool), np.ndarray[tuple[int, int], np.dtype[np.bool]])

assert_type(np.invert(_py_b_0d, dtype="i4"), Any)
assert_type(np.invert(_py_i_1d, dtype="i4"), np.ndarray)
assert_type(np.invert(_u8_2d, dtype="i4"), np.ndarray[tuple[int, int]])

assert_type(np.invert(_py_b_1d, out=_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])

# _ufunc_11_bifo
# (ceil, floor, trunc)

assert_type(np.ceil(_py_b_0d), np.bool)
assert_type(np.ceil(_py_b_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.ceil(_py_b_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.ceil(_py_i_0d), np.int_ | Any)
assert_type(np.ceil(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.ceil(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])
assert_type(np.ceil(_py_f_0d), np.float64 | Any)
assert_type(np.ceil(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.ceil(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])

assert_type(np.ceil(_bool_0d), np.bool)
assert_type(np.ceil(_bool_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.ceil(_bool_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.ceil(_bool_nd), npt.NDArray[np.bool])
assert_type(np.ceil(_u8_0d), np.uint8)
assert_type(np.ceil(_u8_1d), np.ndarray[tuple[int], np.dtype[np.uint8]])
assert_type(np.ceil(_u8_2d), np.ndarray[tuple[int, int], np.dtype[np.uint8]])
assert_type(np.ceil(_u8_nd), npt.NDArray[np.uint8])
assert_type(np.ceil(_i16_0d), np.int16)
assert_type(np.ceil(_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.ceil(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.int16]])
assert_type(np.ceil(_i16_nd), npt.NDArray[np.int16])
assert_type(np.ceil(_f32_0d), np.float32)
assert_type(np.ceil(_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(np.ceil(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.ceil(_f32_nd), npt.NDArray[np.float32])
assert_type(np.ceil(_obj_1d), np.ndarray[tuple[int], np.dtype[np.object_]])
assert_type(np.ceil(_obj_2d), np.ndarray[tuple[int, int], np.dtype[np.object_]])
assert_type(np.ceil(_obj_nd), npt.NDArray[np.object_])

assert_type(np.ceil(_py_b_0d, dtype=np.uint8), np.uint8)
assert_type(np.ceil(_py_i_1d, dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.ceil(_u8_2d, dtype=np.bool), np.ndarray[tuple[int, int], np.dtype[np.bool]])

assert_type(np.ceil(_py_b_0d, dtype="f4"), Any)
assert_type(np.ceil(_py_i_1d, dtype="f4"), np.ndarray)
assert_type(np.ceil(_u8_2d, dtype="f4"), np.ndarray[tuple[int, int]])

assert_type(np.ceil(_py_b_1d, out=_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.ceil(_py_i_2d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# _ufunc_11_bifcmo
# (abs[olute])

assert_type(np.abs(_py_b_0d), np.bool)
assert_type(np.abs(_py_b_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.abs(_py_b_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.abs(_py_i_0d), np.int_ | Any)
assert_type(np.abs(_py_i_1d), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.abs(_py_i_2d), np.ndarray[tuple[int, int], np.dtype[np.int_]])
assert_type(np.abs(_py_f_0d), np.float64 | Any)
assert_type(np.abs(_py_f_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.abs(_py_f_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.abs(_py_c_0d), np.float64 | Any)
assert_type(np.abs(_py_c_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.abs(_py_c_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])

assert_type(np.abs(_bool_0d), np.bool)
assert_type(np.abs(_bool_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.abs(_bool_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.abs(_bool_nd), npt.NDArray[np.bool])
assert_type(np.abs(_u8_0d), np.uint8)
assert_type(np.abs(_u8_1d), np.ndarray[tuple[int], np.dtype[np.uint8]])
assert_type(np.abs(_u8_2d), np.ndarray[tuple[int, int], np.dtype[np.uint8]])
assert_type(np.abs(_u8_nd), npt.NDArray[np.uint8])
assert_type(np.abs(_i16_0d), np.int16)
assert_type(np.abs(_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.abs(_i16_2d), np.ndarray[tuple[int, int], np.dtype[np.int16]])
assert_type(np.abs(_i16_nd), npt.NDArray[np.int16])
assert_type(np.abs(_f32_0d), np.float32)
assert_type(np.abs(_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(np.abs(_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.abs(_f32_nd), npt.NDArray[np.float32])
assert_type(np.abs(_c64_0d), np.float32)
assert_type(np.abs(_c64_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(np.abs(_c64_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(np.abs(_c64_nd), npt.NDArray[np.float32])
assert_type(np.abs(_td_ns_0d), np.timedelta64[int])
assert_type(np.abs(_td_ns_1d), np.ndarray[tuple[int], np.dtype[np.timedelta64[int]]])
assert_type(np.abs(_td_ns_2d), np.ndarray[tuple[int, int], np.dtype[np.timedelta64[int]]])
assert_type(np.abs(_td_ns_nd), npt.NDArray[np.timedelta64[int]])
assert_type(np.abs(_obj_1d), np.ndarray[tuple[int], np.dtype[np.object_]])
assert_type(np.abs(_obj_2d), np.ndarray[tuple[int, int], np.dtype[np.object_]])
assert_type(np.abs(_obj_nd), npt.NDArray[np.object_])

assert_type(np.abs(_py_b_0d, dtype=np.uint8), np.uint8)
assert_type(np.abs(_py_c_1d, dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.abs(_u8_2d, dtype=np.bool), np.ndarray[tuple[int, int], np.dtype[np.bool]])

assert_type(np.abs(_py_b_0d, dtype="f4"), Any)
assert_type(np.abs(_py_i_1d, dtype="f4"), np.ndarray)
assert_type(np.abs(_u8_2d, dtype="f4"), np.ndarray[tuple[int, int]])

assert_type(np.abs(_py_b_1d, out=_i16_1d), np.ndarray[tuple[int], np.dtype[np.int16]])
assert_type(np.abs(_py_i_2d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# _ufunc_12_frexp

assert_type(np.frexp(_py_i_0d), tuple[np.float64, np.int32])
assert_type(
    np.frexp(_py_i_1d),
    tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
    ],
)
assert_type(
    np.frexp(_py_i_2d),
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.int32]],
    ],
)
assert_type(np.frexp(_py_f_0d), tuple[np.float64, np.int32])
assert_type(
    np.frexp(_py_f_1d),
    tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
    ],
)
assert_type(
    np.frexp(_py_f_2d),
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.int32]],
    ],
)

assert_type(np.frexp(_i16_0d), tuple[np.float64, np.int32])
assert_type(
    np.frexp(_i16_1d),
    tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
    ],
)
assert_type(
    np.frexp(_i16_2d),
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.int32]],
    ],
)
assert_type(np.frexp(_i16_nd), tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]])
assert_type(np.frexp(_f32_0d), tuple[np.float32, np.int32])
assert_type(
    np.frexp(_f32_1d),
    tuple[
        np.ndarray[tuple[int], np.dtype[np.float32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
    ],
)
assert_type(
    np.frexp(_f32_2d),
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int], np.dtype[np.int32]],
    ],
)
assert_type(np.frexp(_f32_nd), tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]])

assert_type(
    np.frexp(_py_i_2d, out=(_f32_2d, _i16_2d)),
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int], np.dtype[np.int16]],
    ]
)

# _ufunc_12_modf

type _tuple2[T] = tuple[T, T]

assert_type(np.modf(_py_i_0d), _tuple2[np.float64])
assert_type(np.modf(_py_i_1d), _tuple2[np.ndarray[tuple[int], np.dtype[np.float64]]])
assert_type(np.modf(_py_i_2d), _tuple2[np.ndarray[tuple[int, int], np.dtype[np.float64]]])
assert_type(np.modf(_py_f_0d), _tuple2[np.float64])
assert_type(np.modf(_py_f_1d), _tuple2[np.ndarray[tuple[int], np.dtype[np.float64]]])
assert_type(np.modf(_py_f_2d), _tuple2[np.ndarray[tuple[int, int], np.dtype[np.float64]]])

assert_type(np.modf(_i16_0d), _tuple2[np.float64])
assert_type(np.modf(_i16_1d), _tuple2[np.ndarray[tuple[int], np.dtype[np.float64]]])
assert_type(np.modf(_i16_2d), _tuple2[np.ndarray[tuple[int, int], np.dtype[np.float64]]])
assert_type(np.modf(_i16_nd), _tuple2[npt.NDArray[np.float64]])
assert_type(np.modf(_f32_0d), _tuple2[np.float32])
assert_type(np.modf(_f32_1d), _tuple2[np.ndarray[tuple[int], np.dtype[np.float32]]])
assert_type(np.modf(_f32_2d), _tuple2[np.ndarray[tuple[int, int], np.dtype[np.float32]]])
assert_type(np.modf(_f32_nd), _tuple2[npt.NDArray[np.float32]])

assert_type(np.modf(_py_i_0d, dtype=np.float32), _tuple2[np.float32])
assert_type(np.modf(_py_i_1d, dtype=np.float32), _tuple2[npt.NDArray[np.float32]])
assert_type(np.modf(_i16_2d, dtype=np.float32), _tuple2[np.ndarray[tuple[int, int], np.dtype[np.float32]]])

assert_type(np.modf(_py_i_0d, dtype="f4"), _tuple2[Any])
assert_type(np.modf(_py_i_1d, dtype="f4"), _tuple2[np.ndarray])
assert_type(np.modf(_i16_2d, dtype="f4"), _tuple2[np.ndarray[tuple[int, int]]])

assert_type(
    np.modf(_py_i_2d, out=(_f32_2d, _i16_2d)),
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int], np.dtype[np.int16]],
    ]
)

# _ufunc_21_cmp
# (equal, greater, greater_equal, less, less_equal, not_equal)

assert_type(np.less(_py_b_0d, _py_b_0d), np.bool)
assert_type(np.less(_py_b_1d, _py_b_1d), npt.NDArray[np.bool])
assert_type(np.less(_py_b_2d, _py_b_2d), npt.NDArray[np.bool])
assert_type(np.less(_py_i_0d, _py_i_0d), np.bool)
assert_type(np.less(_py_i_1d, _py_i_1d), npt.NDArray[np.bool])
assert_type(np.less(_py_i_2d, _py_i_2d), npt.NDArray[np.bool])
assert_type(np.less(_py_f_0d, _py_f_0d), np.bool)
assert_type(np.less(_py_f_1d, _py_f_1d), npt.NDArray[np.bool])
assert_type(np.less(_py_f_2d, _py_f_2d), npt.NDArray[np.bool])
assert_type(np.less(_py_c_0d, _py_c_0d), np.bool)
assert_type(np.less(_py_c_1d, _py_c_1d), npt.NDArray[np.bool])
assert_type(np.less(_py_c_2d, _py_c_2d), npt.NDArray[np.bool])

assert_type(np.less(_bool_0d, _bool_0d), np.bool)
assert_type(np.less(_bool_1d, _bool_1d), npt.NDArray[np.bool])
assert_type(np.less(_bool_2d, _bool_2d), npt.NDArray[np.bool])
assert_type(np.less(_bool_nd, _bool_nd), npt.NDArray[np.bool])
assert_type(np.less(_i16_0d, _i16_0d), np.bool)
assert_type(np.less(_i16_1d, _i16_1d), npt.NDArray[np.bool])
assert_type(np.less(_i16_2d, _i16_2d), npt.NDArray[np.bool])
assert_type(np.less(_i16_nd, _i16_nd), npt.NDArray[np.bool])
assert_type(np.less(_f32_0d, _f32_0d), np.bool)
assert_type(np.less(_f32_1d, _f32_1d), npt.NDArray[np.bool])
assert_type(np.less(_f32_2d, _f32_2d), npt.NDArray[np.bool])
assert_type(np.less(_f32_nd, _f32_nd), npt.NDArray[np.bool])
assert_type(np.less(_c64_0d, _c64_0d), np.bool)
assert_type(np.less(_c64_1d, _c64_1d), npt.NDArray[np.bool])
assert_type(np.less(_c64_2d, _c64_2d), npt.NDArray[np.bool])
assert_type(np.less(_c64_nd, _c64_nd), npt.NDArray[np.bool])
assert_type(np.less(_dt_ns_0d, _dt_ns_0d), np.bool)
assert_type(np.less(_dt_ns_1d, _dt_ns_1d), npt.NDArray[np.bool])
assert_type(np.less(_dt_ns_2d, _dt_ns_2d), npt.NDArray[np.bool])
assert_type(np.less(_dt_ns_nd, _dt_ns_nd), npt.NDArray[np.bool])
assert_type(np.less(_td_ns_0d, _td_ns_0d), np.bool)
assert_type(np.less(_td_ns_1d, _td_ns_1d), npt.NDArray[np.bool])
assert_type(np.less(_td_ns_2d, _td_ns_2d), npt.NDArray[np.bool])
assert_type(np.less(_td_ns_nd, _td_ns_nd), npt.NDArray[np.bool])

assert_type(np.less(_py_i_2d, _py_i_2d, out=_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])

assert_type(np.less.outer(_py_c_0d, _py_c_0d), np.bool)
assert_type(np.less.outer(_py_c_1d, _py_c_0d), npt.NDArray[np.bool])
assert_type(np.less.outer(_py_c_0d, _py_c_1d), npt.NDArray[np.bool])
assert_type(np.less.outer(_py_c_1d, _py_c_1d), npt.NDArray[np.bool])

assert_type(np.less.at(_c64_1d, 1, _py_c_1d), None)

assert_type(np.less.accumulate(_bool_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.less.accumulate(_bool_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.less.accumulate(_py_b_1d), npt.NDArray[np.bool])
assert_type(np.less.accumulate(_py_b_1d, out=_bool_1d), np.ndarray[tuple[int], np.dtype[np.bool]])

assert_type(np.less.reduce(_bool_1d), npt.NDArray[np.bool] | np.bool)
assert_type(np.less.reduce(_bool_2d), npt.NDArray[np.bool] | np.bool)
assert_type(np.less.reduce(_bool_1d, axis=None), np.bool)
assert_type(np.less.reduce(_bool_1d, keepdims=True), npt.NDArray[np.bool])
assert_type(np.less.reduce(_bool_1d, out=...), npt.NDArray[np.bool])
assert_type(np.less.reduce(_bool_1d, out=_bool_1d), np.ndarray[tuple[int], np.dtype[np.bool]])

assert_type(np.less.reduceat(_bool_1d, (0,)), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.less.reduceat(_bool_2d, (0,)), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.less.reduceat(_py_b_1d, (0,)), npt.NDArray[np.bool])
assert_type(np.less.reduceat(_py_b_1d, (0,), out=_bool_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
