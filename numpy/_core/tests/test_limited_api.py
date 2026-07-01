import os
import shutil
import subprocess
import sys
import sysconfig

import pytest

import numpy as np
from numpy.testing import IS_EDITABLE, IS_WASM, NOGIL_BUILD
from numpy.testing._private.utils import run_subprocess

# This import is copied from random.tests.test_extending
try:
    import cython
    from Cython.Compiler.Version import version as cython_version
except ImportError:
    cython = None
else:
    from numpy._utils import _pep440

    # Note: keep in sync with the one in pyproject.toml
    required_version = "3.0.6"
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # too old or wrong cython, skip the test
        cython = None

pytestmark = pytest.mark.skipif(cython is None, reason="requires cython")


if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )


@pytest.fixture(scope='module')
def install_temp(tmpdir_factory):
    # Based in part on test_cython from random.tests.test_extending
    if IS_WASM:
        pytest.skip("No subprocess")

    # Build against a copy of the sources placed next to the build dir:
    # meson refers to sources via paths relative to the build dir, and on
    # Windows the unnormalized cwd + `..` chain joining the deeply nested
    # pytest tmp dir and site-packages can exceed MAX_PATH, failing the
    # compile with "Cannot open source file".
    tmp_root = tmpdir_factory.mktemp("limited_api")
    srcdir = str(tmp_root / "src")
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), 'examples', 'limited_api'),
        srcdir)
    build_dir = tmp_root / "build"
    os.makedirs(build_dir, exist_ok=True)
    # Ensure we use the correct Python interpreter even when `meson` is
    # installed in a different Python environment (see gh-24956)
    native_file = str(build_dir / 'interpreter-native-file.ini')
    with open(native_file, 'w') as f:
        f.write("[binaries]\n")
        f.write(f"python = '{sys.executable}'\n")
        f.write(f"python3 = '{sys.executable}'")

    try:
        subprocess.check_call(["meson", "--version"])
    except FileNotFoundError:
        pytest.skip("No usable 'meson' found")
    if sysconfig.get_platform() == "win-arm64":
        pytest.skip("Meson unable to find MSVC linker on win-arm64")
    if sys.platform == "win32":
        run_subprocess(["meson", "setup",
                        "--werror",
                        "--buildtype=release",
                        "--vsenv", "--native-file", native_file,
                        str(srcdir)],
                       build_dir)
    else:
        run_subprocess(["meson", "setup", "--werror",
                        "--native-file", native_file, str(srcdir)],
                       build_dir)
    run_subprocess(["meson", "compile", "-vv"], build_dir)

    sys.path.append(str(build_dir))


def _check_c_api_module(mod):
    arr = np.ones((200, 200))
    assert mod.nonzero(arr) == 200 * 200

    # Legacy single-array iterator: PyArray_ITER_NEXT / _DATA / _NOTDONE.
    arr = np.arange(12.0).reshape(3, 4)
    assert mod.iter_next(arr) == 66.0
    assert mod.iter_goto1d(arr, 5) == 5.0
    assert mod.iter_goto1d(arr, -1) == 11.0
    assert mod.iter_reset(arr) == 66.0
    assert mod.iter_goto(arr, (1, 2)) == 6.0
    assert mod.iter_goto(arr, (2, 3)) == 11.0

    # Broadcasting multi-iterator.
    a = np.arange(3.0).reshape(3, 1)
    b = np.arange(4.0).reshape(1, 4)
    assert mod.multi_iter_next(a, b) == float(np.sum(a + b))
    va, vb = mod.multi_iter_goto(a, b, (1, 2))
    assert va == 1.0 and vb == 2.0
    va, vb = mod.multi_iter_goto1d(a, b, 6)
    assert va == 1.0 and vb == 2.0

    a = np.arange(6.0).reshape(2, 3)
    b = np.zeros((2, 3))
    assert mod.multi_iter_nexti(a, b, 3) == 3.0

    # PyDataType_FLAGS / PyDataType_C_METADATA on datetime descriptors.
    dt = np.array(["2021-01-01"], dtype="datetime64[D]")
    flags, base, num = mod.datetime_metadata(dt)
    assert flags == dt.dtype.flags
    assert base == 4    # NPY_FR_D
    assert num == 1

    # A plain seconds timedelta: base NPY_FR_s, unit multiplier 1.
    td = np.array([5], dtype="timedelta64[s]")
    flags, base, num = mod.datetime_metadata(td)
    assert flags == td.dtype.flags
    assert base == 7    # NPY_FR_s
    assert num == 1

    # A non-unit multiplier exercises the metadata `num` field.
    td = np.array([1000], dtype="timedelta64[ms]").astype("timedelta64[10ms]")
    flags, base, num = mod.datetime_metadata(td)
    assert flags == td.dtype.flags
    assert base == 8    # NPY_FR_ms
    assert num == 10

    # Non-datetime descriptors have no c_metadata and are rejected.
    with pytest.raises(RuntimeError):
        mod.datetime_metadata(np.arange(3))


def _check_cython_module(mod):
    arr = np.ones((200, 200))
    assert mod.nonzero(arr) == 200 * 200

    # Legacy single-array iterator.
    arr = np.arange(12.0).reshape(3, 4)
    assert mod.iter_next(arr) == 66.0
    assert mod.iter_goto1d(arr, 5) == 5.0
    assert mod.iter_goto1d(arr, -1) == 11.0
    assert mod.iter_reset(arr) == 66.0
    assert mod.iter_goto(arr, (1, 2)) == 6.0
    assert mod.iter_goto(arr, (2, 3)) == 11.0

    # Broadcasting multi-iterator.
    a = np.arange(3.0).reshape(3, 1)
    b = np.arange(4.0).reshape(1, 4)
    assert mod.multi_iter_next(a, b) == float(np.sum(a + b))
    va, vb = mod.multi_iter_goto1d(a, b, 6)
    assert va == 1.0 and vb == 2.0

    a = np.arange(6.0).reshape(2, 3)
    b = np.zeros((2, 3))
    assert mod.multi_iter_nexti(a, b, 3) == 3.0

    # Datetime / timedelta scalar accessors (.pxd helpers).
    dt = np.datetime64("2021-01-01", "D")
    value, base, num = mod.datetime_metadata(dt)
    assert value == dt.astype("int64")
    assert base == 4    # NPY_FR_D
    assert num == 1
    assert mod.get_datetime_value(dt) == dt.astype("int64")
    assert mod.get_datetime_unit(dt) == 4
    assert mod.is_datetime64(dt)
    assert not mod.is_timedelta64(dt)

    # A plain seconds timedelta: base NPY_FR_s, unit multiplier 1.
    td = np.timedelta64(5, "s")
    value, base, num = mod.datetime_metadata(td)
    assert value == 5
    assert base == 7    # NPY_FR_s
    assert num == 1
    assert mod.get_timedelta_value(td) == 5
    assert mod.is_timedelta64(td)
    assert not mod.is_datetime64(td)

    # A non-unit multiplier exercises the metadata `num` field.
    td = np.timedelta64(1000, "ms").astype("timedelta64[10ms]")
    value, base, num = mod.datetime_metadata(td)
    assert value == td.astype("int64")
    assert base == 8    # NPY_FR_ms
    assert num == 10
    assert mod.get_timedelta_value(td) == td.astype("int64")

    # Non-datetime scalars are rejected.
    with pytest.raises(TypeError):
        mod.datetime_metadata(np.int64(5))


_PY_ABI3_VERSIONS = [(3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15)]
_NPY_TARGET_VERSIONS = ["2_0", "2_1", "2_2", "2_3", "2_4", "2_5"]

def _module_names(prefix, abi3_versions):
    names = []
    for major, minor in abi3_versions:
        for npy_key in _NPY_TARGET_VERSIONS:
            names.append(f"{prefix}_{major}_{minor}_npy{npy_key}")
    return names


def limited_api_module_names():
    return _module_names("limited_api", _PY_ABI3_VERSIONS)


def limited_api_cython_module_names():
    return _module_names("limited_api_cython", _PY_ABI3_VERSIONS)


@pytest.mark.skipif(IS_WASM, reason="Can't start subprocess")
@pytest.mark.xfail(
    sysconfig.get_config_var("Py_DEBUG"),
    reason=(
        "Py_LIMITED_API is incompatible with Py_DEBUG, Py_TRACE_REFS, "
        "and Py_REF_DEBUG"
    ),
)
@pytest.mark.xfail(
    NOGIL_BUILD,
    reason="Py_GIL_DISABLED builds do not currently support abi3",
)
@pytest.mark.parametrize("module_name", limited_api_module_names())
def test_limited_api_abi3(install_temp, module_name):
    """Exercise the NumPy C-API through each limited-API matrix module."""
    mod = pytest.importorskip(module_name)
    _check_c_api_module(mod)


@pytest.mark.skipif(IS_WASM, reason="Can't start subprocess")
@pytest.mark.xfail(
    sysconfig.get_config_var("Py_DEBUG"),
    reason=(
        "Py_LIMITED_API is incompatible with Py_DEBUG, Py_TRACE_REFS, "
        "and Py_REF_DEBUG"
    ),
)
@pytest.mark.xfail(
    NOGIL_BUILD,
    reason="Py_GIL_DISABLED builds do not currently support abi3",
)
@pytest.mark.parametrize("module_name", limited_api_cython_module_names())
def test_limited_api_cython(install_temp, module_name):
    mod = pytest.importorskip(module_name)
    _check_cython_module(mod)


@pytest.mark.skipif(
    sys.version_info < (3, 15), reason="opaque PyObject requires Python 3.15+"
)
@pytest.mark.skipif(
    sys.platform == "win32" and not sysconfig.get_config_var('Py_GIL_DISABLED'),
    reason=("Meson does not yet support building abi3t extensions on the "
            "GIL-enabled build")
)
def test_limited_opaque(install_temp):
    import limited_api_opaque

    _check_c_api_module(limited_api_opaque)
