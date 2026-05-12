import os
import subprocess
import sys
import sysconfig

import pytest

from numpy.testing import IS_EDITABLE, IS_WASM, NOGIL_BUILD

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

    srcdir = os.path.join(os.path.dirname(__file__), 'examples', 'limited_api')
    build_dir = tmpdir_factory.mktemp("limited_api") / "build"
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
        subprocess.check_call(["meson", "setup",
                               "--werror",
                               "--buildtype=release",
                               "--vsenv", "--native-file", native_file,
                               str(srcdir)],
                              cwd=build_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup", "--werror",
                               "--native-file", native_file, str(srcdir)],
                              cwd=build_dir
                              )
    try:
        subprocess.check_call(
            ["meson", "compile", "-vv"], cwd=build_dir)
    except subprocess.CalledProcessError as p:
        print(f"{p.stdout=}")
        print(f"{p.stderr=}")
        raise

    sys.path.append(str(build_dir))


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
    reason="Py_GIL_DISABLED builds do not currently support the limited API",
)
@pytest.mark.slow
def test_limited_api(install_temp):
    """Test building a third-party C extension with the limited API
    and building a cython extension with the limited API
    """

    import limited_api1  # Earliest (3.6)  # noqa: F401
    import limited_api2  # cython  # noqa: F401
    import limited_api_latest  # Latest version (current Python)  # noqa: F401

@pytest.mark.skipif(
    sys.version_info < (3, 15), reason="opaque PyObject requires Python 3.15+"
)
def test_limited_opaque(install_temp):
    import limited_api_opaque

    import numpy as np
    arr = np.ones((200, 200))
    assert limited_api_opaque.nonzero(arr) == 200 * 200

    # Test PyArray_ITER_NEXT / PyArray_ITER_DATA / PyArray_ITER_NOTDONE
    arr = np.arange(12.0).reshape(3, 4)
    assert limited_api_opaque.iter_next(arr) == 66.0

    # Test PyArray_ITER_GOTO1D
    assert limited_api_opaque.iter_goto1d(arr, 5) == 5.0
    assert limited_api_opaque.iter_goto1d(arr, -1) == 11.0

    # Test PyArray_ITER_RESET
    assert limited_api_opaque.iter_reset(arr) == 66.0

    # Test PyArray_MultiIter_NEXT / RESET / DATA with broadcasting
    a = np.arange(3.0).reshape(3, 1)   # shape (3, 1)
    b = np.arange(4.0).reshape(1, 4)   # shape (1, 4)
    # Each broadcast element is a[i] + b[j], total sum:
    expected = float(np.sum(a + b))
    assert limited_api_opaque.multi_iter_next(a, b) == expected

    # Test PyArray_ITER_GOTO
    arr = np.arange(12.0).reshape(3, 4)
    assert limited_api_opaque.iter_goto(arr, (1, 2)) == 6.0
    assert limited_api_opaque.iter_goto(arr, (2, 3)) == 11.0

    # Test PyArray_MultiIter_GOTO
    a = np.arange(3.0).reshape(3, 1)
    b = np.arange(4.0).reshape(1, 4)
    va, vb = limited_api_opaque.multi_iter_goto(a, b, (1, 2))
    assert va == 1.0 and vb == 2.0

    # Test PyArray_MultiIter_GOTO1D
    # flat index 6 in (3,4) broadcast → row 1, col 2
    va, vb = limited_api_opaque.multi_iter_goto1d(a, b, 6)
    assert va == 1.0 and vb == 2.0

    # Test PyArray_MultiIter_NEXTi
    a = np.arange(6.0).reshape(2, 3)
    b = np.zeros((2, 3))
    # Advance iter 0 by 3 steps → flat index 3 → value 3.0
    assert limited_api_opaque.multi_iter_nexti(a, b, 3) == 3.0
