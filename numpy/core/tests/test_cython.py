import os
import shutil
import subprocess
import sys
import pytest

import numpy as np

# This import is copied from random.tests.test_extending
try:
    import cython
    from Cython.Compiler.Version import version as cython_version
except ImportError:
    cython = None
else:
    from distutils.version import LooseVersion

    # Cython 0.29.21 is required for Python 3.9 and there are
    # other fixes in the 0.29 series that are needed even for earlier
    # Python versions.
    # Note: keep in sync with the one in pyproject.toml
    required_version = LooseVersion("0.29.21")
    if LooseVersion(cython_version) < required_version:
        # too old or wrong cython, skip the test
        cython = None

pytestmark = pytest.mark.skipif(cython is None, reason="requires cython")


@pytest.fixture
def install_temp(request, tmp_path):
    # Based in part on test_cython from random.tests.test_extending

    here = os.path.dirname(__file__)
    ext_dir = os.path.join(here, "examples")

    tmp_path = tmp_path._str
    cytest = os.path.join(tmp_path, "cytest")

    shutil.copytree(ext_dir, cytest)
    # build the examples and "install" them into a temporary directory

    install_log = os.path.join(tmp_path, "tmp_install_log.txt")
    subprocess.check_call(
        [
            sys.executable,
            "setup.py",
            "build",
            "install",
            "--prefix",
            os.path.join(tmp_path, "installdir"),
            "--single-version-externally-managed",
            "--record",
            install_log,
        ],
        cwd=cytest,
    )

    # In order to import the built module, we need its path to sys.path
    # so parse that out of the record
    with open(install_log) as fid:
        for line in fid:
            if "checks" in line:
                sys.path.append(os.path.dirname(line))
                break
        else:
            raise RuntimeError(f'could not parse "{install_log}"')


def test_is_integer_object(install_temp):
    import checks

    assert not checks.is_integer(np.timedelta64(1234))
    assert not checks.is_integer(np.datetime64(1234, "D"))

    assert not checks.is_integer("a")

    assert not checks.is_integer(float("nan"))
    assert not checks.is_integer(np.nan)
    assert not checks.is_integer(2.0)
    assert not checks.is_integer(np.float64(2.0))

    assert checks.is_integer(1)
    assert checks.is_integer(np.int64(2))
    assert checks.is_integer(np.uint32(4))

    # Only scalars, not integer-dtyped ndarrays
    assert not checks.is_integer(np.array([1]))


def test_is_float_object(install_temp):
    import checks

    assert not checks.is_float(np.timedelta64(1234))
    assert not checks.is_float(np.datetime64(1234, "D"))

    assert not checks.is_float("a")

    assert checks.is_float(float("nan"))
    assert checks.is_float(np.nan)
    assert checks.is_float(2.0)
    assert checks.is_float(np.float64(2.0))

    assert not checks.is_float(1)
    assert not checks.is_float(np.int64(2))
    assert not checks.is_float(np.uint32(4))

    # Only scalars, not float-dtyped ndarrays
    assert not checks.is_float(np.array([1.0]))


def test_is_complex_object(install_temp):
    import checks

    assert not checks.is_complex(np.timedelta64(1234))
    assert not checks.is_complex(np.datetime64(1234, "D"))

    assert not checks.is_complex("a")

    assert not checks.is_complex(float("nan"))
    assert not checks.is_complex(np.nan)
    assert not checks.is_complex(2.0)
    assert not checks.is_complex(np.float64(2.0))

    assert not checks.is_complex(1)
    assert not checks.is_complex(np.int64(2))
    assert not checks.is_complex(np.uint32(4))

    assert checks.is_complex(2.0j)
    assert checks.is_complex(np.complex64(2.0))
    assert checks.is_complex(np.complex64(np.nan))

    # Only scalars, not float-dtyped ndarrays
    assert not checks.is_complex(np.array([1.0], dtype="c8"))


def test_is_bool_object(install_temp):
    import checks

    assert not checks.is_bool(np.timedelta64(1234))
    assert not checks.is_bool(np.datetime64(1234, "D"))

    assert not checks.is_bool("a")
    assert not checks.is_bool("True")

    assert not checks.is_bool(float("nan"))
    assert not checks.is_bool(np.nan)
    assert not checks.is_bool(2.0)
    assert not checks.is_bool(np.float64(2.0))

    assert not checks.is_bool(1)
    assert not checks.is_bool(np.int64(2))
    assert not checks.is_bool(np.uint32(4))

    assert not checks.is_bool(2.0j)
    assert not checks.is_bool(np.complex64(2.0))
    assert not checks.is_bool(np.complex64(np.nan))

    assert checks.is_bool(True)
    assert checks.is_bool(np.bool_(False))


    # Only scalars, not bool-dtyped ndarrays
    assert not checks.is_bool(np.array([True], dtype=bool))


def test_is_timedelta64_object(install_temp):
    import checks

    assert checks.is_td64(np.timedelta64(1234))
    assert checks.is_td64(np.timedelta64(1234, "ns"))
    assert checks.is_td64(np.timedelta64("NaT", "ns"))

    assert not checks.is_td64(1)
    assert not checks.is_td64(None)
    assert not checks.is_td64("foo")
    assert not checks.is_td64(np.datetime64("now", "s"))


def test_is_datetime64_object(install_temp):
    import checks

    assert checks.is_dt64(np.datetime64(1234, "ns"))
    assert checks.is_dt64(np.datetime64("NaT", "ns"))

    assert not checks.is_dt64(1)
    assert not checks.is_dt64(None)
    assert not checks.is_dt64("foo")
    assert not checks.is_dt64(np.timedelta64(1234))


def test_get_datetime64_value(install_temp):
    import checks

    dt64 = np.datetime64("2016-01-01", "ns")

    result = checks.get_dt64_value(dt64)
    expected = dt64.view("i8")

    assert result == expected


def test_get_timedelta64_value(install_temp):
    import checks

    td64 = np.timedelta64(12345, "h")

    result = checks.get_td64_value(td64)
    expected = td64.view("i8")

    assert result == expected


def test_get_datetime64_unit(install_temp):
    import checks

    dt64 = np.datetime64("2016-01-01", "ns")
    result = checks.get_dt64_unit(dt64)
    expected = 10
    assert result == expected

    td64 = np.timedelta64(12345, "h")
    result = checks.get_dt64_unit(td64)
    expected = 5
    assert result == expected
