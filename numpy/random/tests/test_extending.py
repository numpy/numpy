from importlib.util import spec_from_file_location, module_from_spec
import os
import pathlib
import pytest
import shutil
import subprocess
import sys
import textwrap
import warnings

import numpy as np
from numpy.testing import IS_WASM


try:
    import cffi
except ImportError:
    cffi = None

if sys.flags.optimize > 1:
    # no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1
    # cffi cannot succeed
    cffi = None

try:
    with warnings.catch_warnings(record=True) as w:
        # numba issue gh-4733
        warnings.filterwarnings('always', '', DeprecationWarning)
        import numba
except (ImportError, SystemError):
    # Certain numpy/numba versions trigger a SystemError due to a numba bug
    numba = None

try:
    import cython
    from Cython.Compiler.Version import version as cython_version
except ImportError:
    cython = None
else:
    from numpy._utils import _pep440
    # Cython 0.29.30 is required for Python 3.11 and there are
    # other fixes in the 0.29 series that are needed even for earlier
    # Python versions.
    # Note: keep in sync with the one in pyproject.toml
    required_version = '0.29.35'
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # too old or wrong cython, skip the test
        cython = None


@pytest.mark.skipif(IS_WASM, reason="Can't start subprocess")
@pytest.mark.skipif(cython is None, reason="requires cython")
@pytest.mark.slow
def test_cython(tmp_path):
    import glob
    # build the examples in a temporary directory
    srcdir = os.path.join(os.path.dirname(__file__), '..')
    shutil.copytree(srcdir, tmp_path / 'random')
    build_dir = tmp_path / 'random' / '_examples' / 'cython'
    # We don't want a wheel build, so do the steps in a controlled way
    # The meson.build file is not copied as part of the build, so generate it
    with open(build_dir / "meson.build", "wt", encoding="utf-8") as fid:
        get_inc = ('import os; os.chdir(".."); import numpy; '
                   'print(os.path.abspath(numpy.get_include() + "../../.."))')
        fid.write(textwrap.dedent(f"""\
            project('random-build-examples', 'c', 'cpp', 'cython')

            # https://mesonbuild.com/Python-module.html
            py_mod = import('python')
            py3 = py_mod.find_installation(pure: false)
            py3_dep = py3.dependency()

            py_mod = import('python')
            py = py_mod.find_installation(pure: false)
            cc = meson.get_compiler('c')
            cy = meson.get_compiler('cython')

            if not cy.version().version_compare('>=0.29.35')
              error('tests requires Cython >= 0.29.35')
            endif

            _numpy_abs = run_command(py3, ['-c', '{get_inc}'],
                                     check: true).stdout().strip()

            npymath_path = _numpy_abs / 'core' / 'lib'
            npy_include_path = _numpy_abs / 'core' / 'include'
            npyrandom_path = _numpy_abs / 'random' / 'lib'
            npymath_lib = cc.find_library('npymath', dirs: npymath_path)
            npyrandom_lib = cc.find_library('npyrandom', dirs: npyrandom_path)

            py.extension_module(
                'extending_distributions',
                'extending_distributions.pyx',
                install: false,
                include_directories: [npy_include_path],
                dependencies: [npyrandom_lib, npymath_lib],
            )
            py.extension_module(
                'extending',
                'extending.pyx',
                install: false,
                include_directories: [npy_include_path],
                dependencies: [npyrandom_lib, npymath_lib],
            )
        """))
    target_dir = build_dir / "build"
    os.makedirs(target_dir, exist_ok=True)
    subprocess.check_call(["meson", "setup", str(build_dir)], cwd=target_dir)
    subprocess.check_call(["meson", "compile"], cwd=target_dir)

    # gh-16162: make sure numpy's __init__.pxd was used for cython
    # not really part of this test, but it is a convenient place to check

    g = glob.glob(str(target_dir / "*" / "extending.pyx.c"))
    with open(g[0]) as fid:
        txt_to_find = 'NumPy API declarations from "numpy/__init__'
        for i, line in enumerate(fid):
            if txt_to_find in line:
                break
        else:
            assert False, ("Could not find '{}' in C file, "
                           "wrong pxd used".format(txt_to_find))
    # import without adding the directory to sys.path
    so1 = sorted(glob.glob(str(target_dir / "extending.*")))[0]
    so2 = sorted(glob.glob(str(target_dir / "extending_distributions.*")))[0]
    spec1 = spec_from_file_location("extending", so1)
    spec2 = spec_from_file_location("extending_distributions", so2)
    extending = module_from_spec(spec1)
    spec1.loader.exec_module(extending)
    extending_distributions = module_from_spec(spec2)
    spec2.loader.exec_module(extending_distributions)
    # actually test the cython c-extension
    from numpy.random import PCG64
    values = extending_distributions.uniforms_ex(PCG64(0), 10, 'd')
    assert values.shape == (10,)
    assert values.dtype == np.float64

@pytest.mark.skipif(numba is None or cffi is None,
                    reason="requires numba and cffi")
def test_numba():
    from numpy.random._examples.numba import extending  # noqa: F401

@pytest.mark.skipif(cffi is None, reason="requires cffi")
def test_cffi():
    from numpy.random._examples.cffi import extending  # noqa: F401
