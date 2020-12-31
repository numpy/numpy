import importlib.util
import itertools
import os
import re
from collections import defaultdict
from typing import Optional, IO, Dict, List

import pytest
import numpy as np
from numpy.typing.mypy_plugin import _PRECISION_DICT

try:
    from mypy import api
except ImportError:
    NO_MYPY = True
else:
    NO_MYPY = False


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PASS_DIR = os.path.join(DATA_DIR, "pass")
FAIL_DIR = os.path.join(DATA_DIR, "fail")
REVEAL_DIR = os.path.join(DATA_DIR, "reveal")
MYPY_INI = os.path.join(DATA_DIR, "mypy.ini")
CACHE_DIR = os.path.join(DATA_DIR, ".mypy_cache")


def get_test_cases(directory):
    for root, _, files in os.walk(directory):
        for fname in files:
            if os.path.splitext(fname)[-1] == ".py":
                fullpath = os.path.join(root, fname)
                # Use relative path for nice py.test name
                relpath = os.path.relpath(fullpath, start=directory)

                yield pytest.param(
                    fullpath,
                    # Manually specify a name for the test
                    id=relpath,
                )


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(PASS_DIR))
def test_success(path):
    stdout, stderr, exitcode = api.run([
        "--config-file",
        MYPY_INI,
        "--cache-dir",
        CACHE_DIR,
        path,
    ])
    assert exitcode == 0, stdout
    assert re.match(r"Success: no issues found in \d+ source files?", stdout.strip())


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(FAIL_DIR))
def test_fail(path):
    __tracebackhide__ = True

    stdout, stderr, exitcode = api.run([
        "--config-file",
        MYPY_INI,
        "--cache-dir",
        CACHE_DIR,
        path,
    ])
    assert exitcode != 0

    with open(path) as fin:
        lines = fin.readlines()

    errors = defaultdict(lambda: "")
    error_lines = stdout.rstrip("\n").split("\n")
    assert re.match(
        r"Found \d+ errors? in \d+ files? \(checked \d+ source files?\)",
        error_lines[-1].strip(),
    )
    for error_line in error_lines[:-1]:
        error_line = error_line.strip()
        if not error_line:
            continue

        match = re.match(
            r"^.+\.py:(?P<lineno>\d+): (error|note): .+$",
            error_line,
        )
        if match is None:
            raise ValueError(f"Unexpected error line format: {error_line}")
        lineno = int(match.group('lineno'))
        errors[lineno] += error_line

    for i, line in enumerate(lines):
        lineno = i + 1
        if line.startswith('#') or (" E:" not in line and lineno not in errors):
            continue

        target_line = lines[lineno - 1]
        if "# E:" in target_line:
            marker = target_line.split("# E:")[-1].strip()
            expected_error = errors.get(lineno)
            _test_fail(path, marker, expected_error, lineno)
        else:
            pytest.fail(f"Error {repr(errors[lineno])} not found")


_FAIL_MSG1 = """Extra error at line {}

Extra error: {!r}
"""

_FAIL_MSG2 = """Error mismatch at line {}

Expected error: {!r}
Observed error: {!r}
"""


def _test_fail(path: str, error: str, expected_error: Optional[str], lineno: int) -> None:
    if expected_error is None:
        raise AssertionError(_FAIL_MSG1.format(lineno, error))
    elif error not in expected_error:
        raise AssertionError(_FAIL_MSG2.format(lineno, expected_error, error))


def _construct_format_dict():
    dct = {k.split(".")[-1]: v.replace("numpy", "numpy.typing") for
           k, v in _PRECISION_DICT.items()}

    return {
        "uint8": "numpy.unsignedinteger[numpy.typing._8Bit]",
        "uint16": "numpy.unsignedinteger[numpy.typing._16Bit]",
        "uint32": "numpy.unsignedinteger[numpy.typing._32Bit]",
        "uint64": "numpy.unsignedinteger[numpy.typing._64Bit]",
        "int8": "numpy.signedinteger[numpy.typing._8Bit]",
        "int16": "numpy.signedinteger[numpy.typing._16Bit]",
        "int32": "numpy.signedinteger[numpy.typing._32Bit]",
        "int64": "numpy.signedinteger[numpy.typing._64Bit]",
        "float16": "numpy.floating[numpy.typing._16Bit]",
        "float32": "numpy.floating[numpy.typing._32Bit]",
        "float64": "numpy.floating[numpy.typing._64Bit]",
        "complex64": "numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]",
        "complex128": "numpy.complexfloating[numpy.typing._64Bit, numpy.typing._64Bit]",

        "ubyte": f"numpy.unsignedinteger[{dct['_NBitByte']}]",
        "ushort": f"numpy.unsignedinteger[{dct['_NBitShort']}]",
        "uintc": f"numpy.unsignedinteger[{dct['_NBitIntC']}]",
        "uintp": f"numpy.unsignedinteger[{dct['_NBitIntP']}]",
        "uint": f"numpy.unsignedinteger[{dct['_NBitInt']}]",
        "ulonglong": f"numpy.unsignedinteger[{dct['_NBitLongLong']}]",
        "byte": f"numpy.signedinteger[{dct['_NBitByte']}]",
        "short": f"numpy.signedinteger[{dct['_NBitShort']}]",
        "intc": f"numpy.signedinteger[{dct['_NBitIntC']}]",
        "intp": f"numpy.signedinteger[{dct['_NBitIntP']}]",
        "int_": f"numpy.signedinteger[{dct['_NBitInt']}]",
        "longlong": f"numpy.signedinteger[{dct['_NBitLongLong']}]",

        "half": f"numpy.floating[{dct['_NBitHalf']}]",
        "single": f"numpy.floating[{dct['_NBitSingle']}]",
        "double": f"numpy.floating[{dct['_NBitDouble']}]",
        "longdouble": f"numpy.floating[{dct['_NBitLongDouble']}]",
        "csingle": f"numpy.complexfloating[{dct['_NBitSingle']}, {dct['_NBitSingle']}]",
        "cdouble": f"numpy.complexfloating[{dct['_NBitDouble']}, {dct['_NBitDouble']}]",
        "clongdouble": f"numpy.complexfloating[{dct['_NBitLongDouble']}, {dct['_NBitLongDouble']}]",

        # numpy.typing
        "_NBitInt": dct['_NBitInt'],
    }


#: A dictionary with all supported format keys (as keys)
#: and matching values
FORMAT_DICT: Dict[str, str] = _construct_format_dict()


def _parse_reveals(file: IO[str]) -> List[str]:
    """Extract and parse all ``"  # E: "`` comments from the passed file-like object.

    All format keys will be substituted for their respective value from `FORMAT_DICT`,
    *e.g.* ``"{float64}"`` becomes ``"numpy.floating[numpy.typing._64Bit]"``.
    """
    string = file.read().replace("*", "")

    # Grab all `# E:`-based comments
    comments_array = np.char.partition(string.split("\n"), sep="  # E: ")[:, 2]
    comments = "/n".join(comments_array)

    # Only search for the `{*}` pattern within comments,
    # otherwise there is the risk of accidently grabbing dictionaries and sets
    key_set = set(re.findall(r"\{(.*?)\}", comments))
    kwargs = {
        k: FORMAT_DICT.get(k, f"<UNRECOGNIZED FORMAT KEY {k!r}>") for k in key_set
    }
    fmt_str = comments.format(**kwargs)

    return fmt_str.split("/n")


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(REVEAL_DIR))
def test_reveal(path):
    __tracebackhide__ = True

    stdout, stderr, exitcode = api.run([
        "--config-file",
        MYPY_INI,
        "--cache-dir",
        CACHE_DIR,
        path,
    ])

    with open(path) as fin:
        lines = _parse_reveals(fin)

    stdout_list = stdout.replace('*', '').split("\n")
    for error_line in stdout_list:
        error_line = error_line.strip()
        if not error_line:
            continue

        match = re.match(
            r"^.+\.py:(?P<lineno>\d+): note: .+$",
            error_line,
        )
        if match is None:
            raise ValueError(f"Unexpected reveal line format: {error_line}")
        lineno = int(match.group('lineno')) - 1
        assert "Revealed type is" in error_line

        marker = lines[lineno]
        _test_reveal(path, marker, error_line, 1 + lineno)


_REVEAL_MSG = """Reveal mismatch at line {}

Expected reveal: {!r}
Observed reveal: {!r}
"""


def _test_reveal(path: str, reveal: str, expected_reveal: str, lineno: int) -> None:
    if reveal not in expected_reveal:
        raise AssertionError(_REVEAL_MSG.format(lineno, expected_reveal, reveal))


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(PASS_DIR))
def test_code_runs(path):
    path_without_extension, _ = os.path.splitext(path)
    dirname, filename = path.split(os.sep)[-2:]
    spec = importlib.util.spec_from_file_location(f"{dirname}.{filename}", path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
