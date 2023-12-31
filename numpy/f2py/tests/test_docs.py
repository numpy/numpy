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

@pytest.fixture(scope="module")
def doc_advanced_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestDocAdvanced",
        sources=[_path("asterisk1.f90"),
                 _path("asterisk2.f90"),
                 _path("ftype.f")],
    )
    return spec

@pytest.mark.parametrize("_mod", ["doc_advanced_spec"], indirect=True)
def test_asterisk1(_mod):
    foo = getattr(_mod, 'foo1')
    assert_equal(foo(), b'123456789A12')

@pytest.mark.parametrize("_mod", ["doc_advanced_spec"], indirect=True)
def test_asterisk2(_mod):
    foo = getattr(_mod, 'foo2')
    assert_equal(foo(2), b'12')
    assert_equal(foo(12), b'123456789A12')
    assert_equal(foo(20), b'123456789A123456789B')

@pytest.mark.parametrize("_mod", ["doc_advanced_spec"], indirect=True)
def test_ftype(_mod):
    ftype = _mod
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
