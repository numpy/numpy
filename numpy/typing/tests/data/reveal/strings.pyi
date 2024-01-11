import sys

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

AR_U: npt.NDArray[np.str_]
AR_S: npt.NDArray[np.bytes_]

assert_type(np.strings.equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.not_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.not_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.greater_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.greater_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.less_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.less_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.greater(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.greater(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.less(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.less(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.lstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.lstrip(AR_S, b"_"), npt.NDArray[np.bytes_])
assert_type(np.strings.rstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.rstrip(AR_S, b"_"), npt.NDArray[np.bytes_])
assert_type(np.strings.strip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.strip(AR_S, b"_"), npt.NDArray[np.bytes_])

assert_type(np.strings.count(AR_U, "a", [1, 2, 3], [1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.count(AR_S, [b"a", b"b", b"c"], 0, 9), npt.NDArray[np.int_])

assert_type(np.strings.endswith(AR_U, "a", [1, 2, 3], [1, 2, 3]), npt.NDArray[np.bool])
assert_type(np.strings.endswith(AR_S, [b"a", b"b", b"c"], 0, 9), npt.NDArray[np.bool])
assert_type(np.strings.startswith(AR_U, "a", [1, 2, 3], [1, 2, 3]), npt.NDArray[np.bool])
assert_type(np.strings.startswith(AR_S, [b"a", b"b", b"c"], 0, 9), npt.NDArray[np.bool])

assert_type(np.strings.find(AR_U, "a", [1, 2, 3], [1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.find(AR_S, [b"a", b"b", b"c"], 0, 9), npt.NDArray[np.int_])
assert_type(np.strings.rfind(AR_U, "a", [1, 2, 3], [1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.rfind(AR_S, [b"a", b"b", b"c"], 0, 9), npt.NDArray[np.int_])

assert_type(np.strings.isalpha(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isalpha(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isdecimal(AR_U), npt.NDArray[np.bool])

assert_type(np.strings.isdigit(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isdigit(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isnumeric(AR_U), npt.NDArray[np.bool])

assert_type(np.strings.isspace(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isspace(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.str_len(AR_U), npt.NDArray[np.int_])
assert_type(np.strings.str_len(AR_S), npt.NDArray[np.int_])

assert_type(np.strings.replace(AR_U, "_", "-", 1), npt.NDArray[np.str_])
assert_type(np.strings.replace(AR_S, [b"_", b""], [b"a", b"b"], 10), npt.NDArray[np.bytes_])
