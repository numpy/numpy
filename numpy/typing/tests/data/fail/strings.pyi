import numpy as np
import numpy.typing as npt

AR_U: npt.NDArray[np.str_]
AR_S: npt.NDArray[np.bytes_]

np.strings.equal(AR_U, AR_S)  # E: incompatible type

np.strings.not_equal(AR_U, AR_S)  # E: incompatible type

np.strings.greater_equal(AR_U, AR_S)  # E: incompatible type

np.strings.less_equal(AR_U, AR_S)  # E: incompatible type

np.strings.greater(AR_U, AR_S)  # E: incompatible type

np.strings.less(AR_U, AR_S)  # E: incompatible type

np.strings.lstrip(AR_U, x2=b"a")  # E: incompatible type
np.strings.lstrip(AR_S, x2="a")  # E: incompatible type
np.strings.strip(AR_U, x2=b"a")  # E: incompatible type
np.strings.strip(AR_S, x2="a")  # E: incompatible type
np.strings.rstrip(AR_U, x2=b"a")  # E: incompatible type
np.strings.rstrip(AR_S, x2="a")  # E: incompatible type

np.strings.count(AR_U, b"a", x3=[1, 2, 3])  # E: incompatible type
np.strings.count(AR_S, "a", x4=9)  # E: incompatible type

np.strings.endswith(AR_U, b"a", x3=[1, 2, 3])  # E: incompatible type
np.strings.endswith(AR_S, "a", x4=9)  # E: incompatible type
np.strings.startswith(AR_U, b"a", x3=[1, 2, 3])  # E: incompatible type
np.strings.startswith(AR_S, "a", x4=9)  # E: incompatible type

np.strings.find(AR_U, b"a", x3=[1, 2, 3])  # E: incompatible type
np.strings.find(AR_S, "a", x4=9)  # E: incompatible type
np.strings.rfind(AR_U, b"a", x3=[1, 2, 3])  # E: incompatible type
np.strings.rfind(AR_S, "a", x4=9)  # E: incompatible type

np.strings.isdecimal(AR_S)  # E: incompatible type
np.strings.isnumeric(AR_S)  # E: incompatible type

np.strings.replace(AR_U, b"_", b"-")  # E: incompatible type
np.strings.replace(AR_S, "_", "-")  # E: incompatible type
