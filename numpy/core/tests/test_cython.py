from datetime import datetime
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

    # Cython 0.29.14 is required for Python 3.8 and there are
    # other fixes in the 0.29 series that are needed even for earlier
    # Python versions.
    # Note: keep in sync with the one in pyproject.toml
    required_version = LooseVersion("0.29.14")
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


class TestDatetimeStrings:
    def test_make_iso_8601_datetime(self, install_temp):
        dt = datetime(2016, 6, 2, 10, 45, 19)
        result = checks.make_iso_8601_datetime(dt)
        assert result == "2016-05-02 10:45:19"

    def test_get_datetime_iso_8601_strlen(self, install_temp):
        result = checks.get_datetime_iso_8601_strlen()
        assert result == 22

    def test_parse_iso_8601_datetime(self, install_temp):
        raise NotImplementedError


def test_convert_datetime_to_datetimestruct(install_temp):
    dt = datetime(2016, 6, 2, 10, 45, 19)
    result = checks.convert_datetime_to_datetimestruct(dt)
    assert isinstance(result, dict)
    assert result["year"] == 2016
    assert result["month"] = 6
    assert result["day"] == 2
    assert result["hour"] = 10
    assert result["min"] = 45
    assert result["sec"] = 19
    # other dts fields aren't pinned down


def test_convert_datetimestruct_to_datetime(install_temp):
    raise NotImplementedError
