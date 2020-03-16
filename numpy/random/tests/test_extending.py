import os
import pytest
import shutil
import subprocess
import sys
import warnings
import numpy as np

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
except ImportError:
    numba = None

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
    required_version = LooseVersion('0.29.14')
    if LooseVersion(cython_version) < required_version:
        # too old or wrong cython, skip the test
        cython = None

@pytest.mark.skipif(cython is None, reason="requires cython")
@pytest.mark.slow
def test_cython(tmp_path):
    srcdir = os.path.join(os.path.dirname(__file__), '..')
    shutil.copytree(srcdir, tmp_path / 'random')
    # build the examples and "install" them into a temporary directory
    env = os.environ.copy()
    subprocess.check_call([sys.executable, 'setup.py', 'build', 'install',
                           '--prefix', str(tmp_path / 'installdir'),
                           '--single-version-externally-managed',
                           '--record', str(tmp_path/ 'tmp_install_log.txt'),
                          ],
                          cwd=str(tmp_path / 'random' / '_examples' / 'cython'),
                          env=env)
    # get the path to the so's
    so1 = so2 = None
    with open(tmp_path /'tmp_install_log.txt') as fid:
        for line in fid:
            if 'extending.' in line:
                so1 = line.strip()
            if 'extending_distributions' in line:
                so2 = line.strip()
    assert so1 is not None
    assert so2 is not None
    # import the so's without adding the directory to sys.path
    from importlib.machinery import ExtensionFileLoader 
    extending = ExtensionFileLoader('extending', so1).load_module()
    extending_distributions = ExtensionFileLoader('extending_distributions', so2).load_module()

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
