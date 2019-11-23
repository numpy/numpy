import os, sys
import pytest
import warnings

try:
    with warnings.catch_warnings(record=True) as w:
        # numba issue gh-4733
        warnings.filterwarnings('always', '', DeprecationWarning)
        import numba
    import cffi
except ImportError:
    numba = None

try:
    import cython
except ImportError:
    cython = None

@pytest.mark.skipif(cython is None, reason="requires cython")
def test_cython():
    curdir = os.getcwd()
    argv = sys.argv
    examples = (os.path.dirname(__file__), '..', '_examples')
    try:
        os.chdir(os.path.join(*examples))
        sys.argv = argv[:1] + ['build']
        with warnings.catch_warnings(record=True) as w:
            # setuptools issue gh-1885
            warnings.filterwarnings('always', '', DeprecationWarning)
            from numpy.random._examples.cython import setup
    finally:
        sys.argv = argv
        os.chdir(curdir)

@pytest.mark.skipif(numba is None, reason="requires numba")
def test_numba():
        from numpy.random._examples.numba import extending

