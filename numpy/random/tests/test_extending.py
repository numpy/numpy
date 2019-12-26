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

cython_ver = cython.__version__.split('.')
if len(cython_ver) < 3 or cython_ver < ['0', '29', '14']:
    # too old or wrong cython, skip the test
    cython = None

@pytest.mark.skipif(cython is None, reason="requires cython")
@pytest.mark.slow
@pytest.mark.skipif(sys.platform == 'win32', reason="cmd too long on CI")
def test_cython(tmp_path):
    curdir = os.getcwd()
    argv = sys.argv
    examples = os.path.join(os.path.dirname(__file__), '..', '_examples')
    base = os.path.dirname(examples)
    shutil.copytree(examples, tmp_path / '_examples')
    env = os.environ.copy()
    subprocess.check_call([sys.executable, 'setup.py', 'build'], env=env,
                          cwd=str(tmp_path / '_examples' / 'cython'))

@pytest.mark.skipif(numba is None or cffi is None,
                    reason="requires numba and cffi")
def test_numba():
    from numpy.random._examples.numba import extending

@pytest.mark.skipif(cffi is None, reason="requires cffi")
def test_cffi():
    from numpy.random._examples.cffi import extending
