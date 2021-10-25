import numpy as np
from typing import Any

AR_U: np.chararray[Any, np.dtype[np.str_]]
AR_S: np.chararray[Any, np.dtype[np.bytes_]]

reveal_type(AR_U == AR_U)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S == AR_S)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U != AR_U)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S != AR_S)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U >= AR_U)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S >= AR_S)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U <= AR_U)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S <= AR_S)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U > AR_U)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S > AR_S)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U < AR_U)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S < AR_S)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U * 5)  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S * [5])  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U % "test")  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S % b"test")  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.capitalize())  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.capitalize())  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.center(5))  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.center([2, 3, 4], b"a"))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.encode())  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]
reveal_type(AR_S.decode())  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]

reveal_type(AR_U.expandtabs())  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.expandtabs(tabsize=4))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.join("_"))  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.join([b"_", b""]))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.ljust(5))  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.ljust([4, 3, 1], fillchar=[b"a", b"b", b"c"]))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]
reveal_type(AR_U.rjust(5))  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.rjust([4, 3, 1], fillchar=[b"a", b"b", b"c"]))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.lstrip())  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.lstrip(chars=b"_"))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]
reveal_type(AR_U.rstrip())  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.rstrip(chars=b"_"))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]
reveal_type(AR_U.strip())  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.strip(chars=b"_"))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.partition("\n"))  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.partition([b"a", b"b", b"c"]))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]
reveal_type(AR_U.rpartition("\n"))  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.rpartition([b"a", b"b", b"c"]))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.replace("_", "-"))  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.replace([b"_", b""], [b"a", b"b"]))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.split("_"))  # E: numpy.ndarray[Any, numpy.dtype[numpy.object_]]
reveal_type(AR_S.split(maxsplit=[1, 2, 3]))  # E: numpy.ndarray[Any, numpy.dtype[numpy.object_]]
reveal_type(AR_U.rsplit("_"))  # E: numpy.ndarray[Any, numpy.dtype[numpy.object_]]
reveal_type(AR_S.rsplit(maxsplit=[1, 2, 3]))  # E: numpy.ndarray[Any, numpy.dtype[numpy.object_]]

reveal_type(AR_U.splitlines())  # E: numpy.ndarray[Any, numpy.dtype[numpy.object_]]
reveal_type(AR_S.splitlines(keepends=[True, True, False]))  # E: numpy.ndarray[Any, numpy.dtype[numpy.object_]]

reveal_type(AR_U.swapcase())  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.swapcase())  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.title())  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.title())  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.upper())  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.upper())  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.zfill(5))  # E: numpy.chararray[Any, numpy.dtype[numpy.str_]]
reveal_type(AR_S.zfill([2, 3, 4]))  # E: numpy.chararray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(AR_U.count("a", start=[1, 2, 3]))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(AR_S.count([b"a", b"b", b"c"], end=9))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]

reveal_type(AR_U.endswith("a", start=[1, 2, 3]))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.endswith([b"a", b"b", b"c"], end=9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_U.startswith("a", start=[1, 2, 3]))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.startswith([b"a", b"b", b"c"], end=9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U.find("a", start=[1, 2, 3]))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(AR_S.find([b"a", b"b", b"c"], end=9))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(AR_U.rfind("a", start=[1, 2, 3]))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(AR_S.rfind([b"a", b"b", b"c"], end=9))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]

reveal_type(AR_U.index("a", start=[1, 2, 3]))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(AR_S.index([b"a", b"b", b"c"], end=9))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(AR_U.rindex("a", start=[1, 2, 3]))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(AR_S.rindex([b"a", b"b", b"c"], end=9))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]

reveal_type(AR_U.isalpha())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.isalpha())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U.isalnum())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.isalnum())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U.isdecimal())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.isdecimal())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U.isdigit())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.isdigit())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U.islower())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.islower())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U.isnumeric())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.isnumeric())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U.isspace())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.isspace())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U.istitle())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.istitle())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(AR_U.isupper())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR_S.isupper())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
