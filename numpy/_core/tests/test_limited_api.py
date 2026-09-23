import importlib
import os
import shutil
import subprocess
import sys
import sysconfig

import pytest

import numpy as np
from numpy.testing import HAS_SUBPROCESSES, IS_EDITABLE, NOGIL_BUILD
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
    required_version = "3.1.0"
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # too old or wrong cython, skip the test
        cython = None

pytestmark = pytest.mark.skipif(cython is None, reason="requires cython")


if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )


def build_limited_api_modules(tmpdir_factory):
    # Based in part on test_cython from random.tests.test_extending
    # This runs meson only once, but that single build compiles a separate
    # extension module for every (Python, NumPy) version combination; see
    # the comment above _PY_ABI3_VERSIONS below.
    if not HAS_SUBPROCESSES:
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


def _check_api_module(mod, cython=False):
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
    if not cython:
        va, vb = mod.multi_iter_goto(a, b, (1, 2))
        assert va == 1.0 and vb == 2.0
    va, vb = mod.multi_iter_goto1d(a, b, 6)
    assert va == 1.0 and vb == 2.0

    a = np.arange(6.0).reshape(2, 3)
    b = np.zeros((2, 3))
    assert mod.multi_iter_nexti(a, b, 3) == 3.0

    if cython:
        # Datetime / timedelta scalar accessors (.pxd helpers).
        dt = np.datetime64("2021-01-01", "D")
        assert mod.get_datetime_value(dt) == dt.astype("int64")
        assert mod.get_datetime_unit(dt) == 4
        assert mod.is_datetime64(dt)
        assert not mod.is_timedelta64(dt)

        # A plain seconds timedelta: base NPY_FR_s, unit multiplier 1.
        td = np.timedelta64(5, "s")
        assert mod.get_timedelta_value(td) == 5
        assert mod.is_timedelta64(td)
        assert not mod.is_datetime64(td)

        # A non-unit multiplier exercises the metadata `num` field.
        td = np.timedelta64(1000, "ms").astype("timedelta64[10ms]")
        assert mod.get_timedelta_value(td) == td.astype("int64")
    else:
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
        td = np.array([1000], dtype="timedelta64[ms]").astype(
            "timedelta64[10ms]")
        flags, base, num = mod.datetime_metadata(td)
        assert flags == td.dtype.flags
        assert base == 8    # NPY_FR_ms
        assert num == 10

        # Non-datetime descriptors have no c_metadata and are rejected.
        with pytest.raises(RuntimeError):
            mod.datetime_metadata(np.arange(3))

        # NpyString allocator API; under abi3t PyArray_StringDTypeObject is
        # opaque and only the descriptor object pointer is passed. Absent
        # when the module targets NumPy < 2.0 (the "default" target).
        if hasattr(mod, "stringdtype_load"):
            arr = np.array(["hello", "world"], dtype=np.dtypes.StringDType())
            assert mod.stringdtype_load(arr) == "hello"
            # A long string is stored on the heap, so loading it
            # dereferences the allocator acquired from the descriptor.
            text = "numpy" * 20
            arr = np.array([text, "world"], dtype=np.dtypes.StringDType())
            assert mod.stringdtype_load(arr) == text
            arr = np.array([None, "world"],
                           dtype=np.dtypes.StringDType(na_object=None))
            assert mod.stringdtype_load(arr) is None


# Test limited API extension modules for all supported Python and NumPy versions.
#
# The single meson build run by build_limited_api_modules compiles one C and one
# Cython extension module for every combination of abi3 Python version (up to
# the running interpreter) and NumPy target version listed below, e.g.
# limited_api_3_11_npy2_2 and limited_api_cython_3_11_npy2_2 (see meson.build
# in examples/limited_api). The test first runs the build, then imports every
# one of those modules by name and tests it with the _check_api_module helper.
# So every iteration tests a different extension module built with a different
# combination of Python and NumPy target versions.
#
# Normally a single Python package uses a single Python and a single NumPy version
# as the target (i.e., lowest-supported) version, so this test isn't realistic in
# that respect. However, it *is* realistic for cross-package issues between
# extension modules. This can happen in practice, e.g. when Cython extensions load
# types/objects from `sys.modules` instead of from the local copy embedded in each
# extension (cython#7914).
#
# The _PY_ABI3_VERSIONS and _NPY_TARGET_VERSIONS lists should be kept in sync
# with the lists defined in meson.build, and the test should be updated
# if new versions are added here.
# The special "default" entry builds without defining NPY_TARGET_VERSION at all,
# which exercises the path where numpyconfig.h picks the current API version.
_PY_ABI3_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14", "3.15"]
_NPY_TARGET_VERSIONS = ["2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "default"]


def _module_names(prefix):
    names = []
    for py_ver in _PY_ABI3_VERSIONS:
        if sys.version_info < tuple(map(int, py_ver.split('.'))):
            continue
        py = py_ver.replace('.', '_')
        for npy_ver in _NPY_TARGET_VERSIONS:
            npy = npy_ver.replace('.', '_')
            names.append(f"{prefix}_{py}_npy{npy}")
    return names


@pytest.mark.skipif(
    not HAS_SUBPROCESSES, reason="platform cannot start subprocesses"
)
def test_limited_api(tmpdir_factory, subtests):
    # Keep these conditions in sync with the ones in meson.build: the abi3
    # modules are only built on GIL-enabled interpreters (and Py_LIMITED_API
    # is incompatible with Py_DEBUG), while the opaque abi3t module requires
    # Python 3.15+ and, on Windows, a free-threaded build.
    test_abi3 = not NOGIL_BUILD and not sysconfig.get_config_var("Py_DEBUG")
    test_opaque = sys.version_info >= (3, 15) and (
        sys.platform != "win32" or sysconfig.get_config_var("Py_GIL_DISABLED")
    )
    if not test_abi3 and not test_opaque:
        pytest.skip("no limited API modules are built for this interpreter")

    build_limited_api_modules(tmpdir_factory)

    if test_abi3:
        for module_name in _module_names("limited_api"):
            with subtests.test(module=module_name):
                mod = importlib.import_module(module_name)
                _check_api_module(mod)

        # see https://github.com/cython/cython/issues/7914
        skip_3_11 = _pep440.parse(cython_version) == _pep440.Version("3.3.0")
        for module_name in _module_names("limited_api_cython"):
            if skip_3_11 and "_3_11_" in module_name:
                # abi3 3.11 module is unimportable with Cython 3.3.0
                continue
            with subtests.test(module=module_name):
                mod = importlib.import_module(module_name)
                _check_api_module(mod, cython=True)

    if test_opaque:
        with subtests.test(module="limited_api_opaque"):
            import limited_api_opaque

            _check_api_module(limited_api_opaque)
