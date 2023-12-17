import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from . import util
from pathlib import Path

def get_docdir():
    # Assumes that spin is used to run tests
    nproot = Path(__file__).resolve().parents[8]
    return  nproot / "doc" / "source" / "f2py" / "code"

pytestmark = pytest.mark.skipif(
    not get_docdir().is_dir(),
    reason=f"Could not find f2py documentation sources"
    f"({get_docdir()} does not exist)",
)

def _path(*args):
    return get_docdir().joinpath(*args)

@pytest.mark.slow
class TestDocAdvanced(util.F2PyTest):
    # options = ['--debug-capi', '--build-dir', '/tmp/build-f2py']
    sources = [_path('asterisk1.f90'), _path('asterisk2.f90'),
               _path('ftype.f')]

    def test_asterisk1(self):
        foo = getattr(self.module, 'foo1')
        assert_equal(foo(), b'123456789A12')

    def test_asterisk2(self):
        foo = getattr(self.module, 'foo2')
        assert_equal(foo(2), b'12')
        assert_equal(foo(12), b'123456789A12')
        assert_equal(foo(20), b'123456789A123456789B')

    def test_ftype(self):
        ftype = self.module
        ftype.foo()
        assert_equal(ftype.data.a, 0)
        ftype.data.a = 3
        ftype.data.x = [1, 2, 3]
        assert_equal(ftype.data.a, 3)
        assert_array_equal(ftype.data.x,
                           np.array([1, 2, 3], dtype=np.float32))
        ftype.data.x[1] = 45
        assert_array_equal(ftype.data.x,
                           np.array([1, 45, 3], dtype=np.float32))

    # TODO: implement test methods for other example Fortran codes
