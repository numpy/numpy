import numpy as np
import numpy.typing as npt
import numpy._typing as np_t

from typing_extensions import assert_type

AR_U: npt.NDArray[np.str_]
AR_B: npt.NDArray[np.bytes_]
AR_S: np.ndarray[np_t._Shape, np.dtypes.StringDType]


assert_type(np.strings.equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.equal(AR_B, AR_B), npt.NDArray[np.bool])
assert_type(np.strings.equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.not_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.not_equal(AR_B, AR_B), npt.NDArray[np.bool])
assert_type(np.strings.not_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.greater_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.greater_equal(AR_B, AR_B), npt.NDArray[np.bool])
assert_type(np.strings.greater_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.less_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.less_equal(AR_B, AR_B), npt.NDArray[np.bool])
assert_type(np.strings.less_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.greater(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.greater(AR_B, AR_B), npt.NDArray[np.bool])
assert_type(np.strings.greater(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.less(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.less(AR_B, AR_B), npt.NDArray[np.bool])
assert_type(np.strings.less(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.add(AR_U, AR_U), npt.NDArray[np.str_])
assert_type(np.strings.add(AR_B, AR_B), npt.NDArray[np.bytes_])
assert_type(np.strings.add(AR_S, AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.multiply(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.multiply(AR_B, [5, 4, 3]), npt.NDArray[np.bytes_])
assert_type(np.strings.multiply(AR_S, 5), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.mod(AR_U, "test"), npt.NDArray[np.str_])
assert_type(np.strings.mod(AR_B, "test"), npt.NDArray[np.bytes_])
assert_type(np.strings.mod(AR_S, "test"), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.capitalize(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.capitalize(AR_B), npt.NDArray[np.bytes_])
assert_type(np.strings.capitalize(AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.center(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.center(AR_B, [2, 3, 4], b"a"), npt.NDArray[np.bytes_])
assert_type(np.strings.center(AR_S, 5), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.encode(AR_U), npt.NDArray[np.bytes_])
assert_type(np.strings.encode(AR_S), npt.NDArray[np.bytes_])
assert_type(np.strings.decode(AR_B), npt.NDArray[np.str_])

assert_type(np.strings.expandtabs(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.expandtabs(AR_B, tabsize=4), npt.NDArray[np.bytes_])
assert_type(np.strings.expandtabs(AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.join(AR_U, "_"), npt.NDArray[np.str_])
assert_type(np.strings.join(AR_B, [b"_", b""]), npt.NDArray[np.bytes_])
assert_type(np.strings.join(AR_S, AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.ljust(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.ljust(AR_B, [4, 3, 1], fillchar=[b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.strings.ljust(AR_S, 5), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.rjust(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.rjust(AR_B, [4, 3, 1], fillchar=[b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.strings.rjust(AR_S, 5), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.lstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.lstrip(AR_B, b"_"), npt.NDArray[np.bytes_])
assert_type(np.strings.lstrip(AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.rstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.rstrip(AR_B, b"_"), npt.NDArray[np.bytes_])
assert_type(np.strings.rstrip(AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.strip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.strip(AR_B, b"_"), npt.NDArray[np.bytes_])
assert_type(np.strings.strip(AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.count(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.count(AR_B, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
# TODO: why can't we use "a" as an argument here like for unicode above?
assert_type(np.strings.count(AR_S, AR_S, start=[1, 2, 3]), npt.NDArray[np.int_])

assert_type(np.strings.partition(AR_U, "\n"), npt.NDArray[np.str_])
assert_type(np.strings.partition(AR_B, [b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.strings.partition(AR_S, AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.rpartition(AR_U, "\n"), npt.NDArray[np.str_])
assert_type(np.strings.rpartition(AR_B, [b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.strings.rpartition(AR_S, AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.replace(AR_U, "_", "-"), npt.NDArray[np.str_])
assert_type(np.strings.replace(AR_B, [b"_", b""], [b"a", b"b"]), npt.NDArray[np.bytes_])
assert_type(np.strings.replace(AR_S, AR_S, AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.split(AR_U, "_"), npt.NDArray[np.object_])
assert_type(np.strings.split(AR_B, maxsplit=[1, 2, 3]), npt.NDArray[np.object_])
assert_type(np.strings.split(AR_S, AR_S), npt.NDArray[np.object_])

assert_type(np.strings.rsplit(AR_U, "_"), npt.NDArray[np.object_])
assert_type(np.strings.rsplit(AR_B, maxsplit=[1, 2, 3]), npt.NDArray[np.object_])
assert_type(np.strings.rsplit(AR_S, AR_S), npt.NDArray[np.object_])

assert_type(np.strings.splitlines(AR_U), npt.NDArray[np.object_])
assert_type(np.strings.splitlines(AR_B, keepends=[True, True, False]), npt.NDArray[np.object_])
assert_type(np.strings.splitlines(AR_S), npt.NDArray[np.object_])

assert_type(np.strings.lower(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.lower(AR_B), npt.NDArray[np.bytes_])
assert_type(np.strings.lower(AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.upper(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.upper(AR_B), npt.NDArray[np.bytes_])
assert_type(np.strings.upper(AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.swapcase(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.swapcase(AR_B), npt.NDArray[np.bytes_])
assert_type(np.strings.swapcase(AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.title(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.title(AR_B), npt.NDArray[np.bytes_])
assert_type(np.strings.title(AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.zfill(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.zfill(AR_B, [2, 3, 4]), npt.NDArray[np.bytes_])
assert_type(np.strings.zfill(AR_S, 5), np.ndarray[np_t._Shape, np.dtypes.StringDType])

assert_type(np.strings.endswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool])
assert_type(np.strings.endswith(AR_B, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])
assert_type(np.strings.endswith(AR_S, AR_S, start=[1, 2, 3]), npt.NDArray[np.bool])

assert_type(np.strings.startswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool])
assert_type(np.strings.startswith(AR_B, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])
assert_type(np.strings.startswith(AR_S, AR_S, start=[1, 2, 3]), npt.NDArray[np.bool])

assert_type(np.strings.find(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.find(AR_B, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.find(AR_S, AR_S, start=[1, 2, 3]), npt.NDArray[np.int_])

assert_type(np.strings.rfind(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.rfind(AR_B, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.rfind(AR_S, AR_S, start=[1, 2, 3]), npt.NDArray[np.int_])

assert_type(np.strings.index(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.index(AR_B, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.index(AR_S, AR_S, start=[1, 2, 3]), npt.NDArray[np.int_])

assert_type(np.strings.rindex(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.rindex(AR_B, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.rindex(AR_S, AR_S, start=[1, 2, 3]), npt.NDArray[np.int_])

assert_type(np.strings.isalpha(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isalpha(AR_B), npt.NDArray[np.bool])
assert_type(np.strings.isalpha(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isalnum(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isalnum(AR_B), npt.NDArray[np.bool])
assert_type(np.strings.isalnum(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isdecimal(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isdecimal(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isdigit(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isdigit(AR_B), npt.NDArray[np.bool])
assert_type(np.strings.isdigit(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.islower(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.islower(AR_B), npt.NDArray[np.bool])
assert_type(np.strings.islower(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isnumeric(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isnumeric(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isspace(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isspace(AR_B), npt.NDArray[np.bool])
assert_type(np.strings.isspace(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.istitle(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.istitle(AR_B), npt.NDArray[np.bool])
assert_type(np.strings.istitle(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isupper(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isupper(AR_B), npt.NDArray[np.bool])
assert_type(np.strings.isupper(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.str_len(AR_U), npt.NDArray[np.int_])
assert_type(np.strings.str_len(AR_B), npt.NDArray[np.int_])
assert_type(np.strings.str_len(AR_S), npt.NDArray[np.int_])

assert_type(np.strings.translate(AR_U, AR_U), npt.NDArray[np.str_])
assert_type(np.strings.translate(AR_B, AR_B), npt.NDArray[np.bytes_])
assert_type(np.strings.translate(AR_S, AR_S), np.ndarray[np_t._Shape, np.dtypes.StringDType])