import os, sys
import pytest
import warnings
import shutil
import subprocess

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
except ImportError:
    cython = None
else:
    cython_ver = cython.__version__.split('.')
    # Cython 0.29.14 is required for Python 3.8 and there are
    # other fixes in the 0.29 series that are needed even for earlier
    # Python versions.
    # Note: keep in sync with the one in pyproject.toml
    if len(cython_ver) < 3 or cython_ver < ['0', '29', '14']:
        # too old or wrong cython, skip the test
        cython = None

@pytest.mark.skipif(cython is None, reason="requires cython")
@pytest.mark.slow
@pytest.mark.skipif(sys.platform == 'win32', reason="cmd too long on CI")
def test_cython(tmp_path):
    examples = os.path.join(os.path.dirname(__file__), '..', '_examples')
    base = os.path.dirname(examples)
    shutil.copytree(examples, tmp_path / '_examples')
    subprocess.check_call([sys.executable, 'setup.py', 'build'],
                          env=os.environ,
                          cwd=str(tmp_path / '_examples' / 'cython'))

@pytest.mark.skipif(numba is None or cffi is None,
                    reason="requires numba and cffi")
def test_numba():
    from numpy.random._examples.numba import extending

@pytest.mark.skipif(cffi is None, reason="requires cffi")
def test_cffi():
    from numpy.random._examples.cffi import extending
