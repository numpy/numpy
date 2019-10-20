import os, sys
import pytest

try:
    import numba
except ImportError:
    numba = None

def test_cython():
    curdir = os.getcwd()
    argv = sys.argv
    try:
        os.chdir('numpy/random/examples')
        sys.argv = argv[:1] + ['build']
        from numpy.random.examples.cython import setup
    finally:
        sys.argv = argv
        os.chdir(curdir)

@pytest.mark.skipif(numba is None, reason="requires numba")
def test_numba():
        from numpy.random.examples.numba import extending
    
