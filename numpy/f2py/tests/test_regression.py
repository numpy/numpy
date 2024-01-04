import os
import pytest

import numpy as np

from . import util


def test_include_path():
    incdir = np.f2py.get_include()
    fnames_in_dir = os.listdir(incdir)
    for fname in ("fortranobject.c", "fortranobject.h"):
        assert fname in fnames_in_dir


@pytest.fixture(scope="module")
def inout_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestIntentInOut",
        sources = [util.getpath("tests", "src", "regression", "inout.f90")],
    )
    return spec


@pytest.mark.parametrize("_mod", ["inout_spec"], indirect=True)
def test_inout(_mod):
    # non-contiguous should raise error
    x = np.arange(6, dtype=np.float32)[::2]
    pytest.raises(ValueError, _mod.foo, x)

    # check values with contiguous array
    x = np.arange(3, dtype=np.float32)
    _mod.foo(x)
    assert np.allclose(x, [3, 1, 2])


@pytest.fixture(scope="module")
def negbound_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestNegativeBounds",
        sources = [util.getpath("tests", "src", "negative_bounds", "issue_20853.f90")],
    )
    return spec


@pytest.mark.parametrize("_mod", ["negbound_spec"], indirect=True)
def test_negbound(_mod):
    xvec = np.arange(12)
    xlow = -6
    xhigh = 4
    # Calculate the upper bound,
    # Keeping the 1 index in mind
    def ubound(xl, xh):
        return xh - xl + 1
    rval = _mod.foo(is_=xlow, ie_=xhigh,
                    arr=xvec[:ubound(xlow, xhigh)])
    expval = np.arange(11, dtype = np.float32)
    assert np.allclose(rval, expval)


@pytest.fixture(scope="module")
def npversion_spec():
    # Check that th attribute __f2py_numpy_version__ is present
    # in the compiled module and that has the value np.__version__.
    spec = util.F2PyModuleSpec(
        test_class_name="TestNumpyVersionAttribute",
        sources = [util.getpath("tests", "src", "regression", "inout.f90")],
    )
    return spec


@pytest.mark.parametrize("_mod", ["npversion_spec"], indirect=True)
def test_numpy_version_attribute(_mod):
    # Check that _mod has an attribute named "__f2py_numpy_version__"
    assert hasattr(_mod, "__f2py_numpy_version__")
    # Check that the attribute __f2py_numpy_version__ is a string
    assert isinstance(_mod.__f2py_numpy_version__, str)
    # Check that __f2py_numpy_version__ has the value numpy.__version__
    assert np.__version__ == _mod.__f2py_numpy_version__


@pytest.fixture(scope="module")
def mod_subrout_spec():
    # Check that th attribute __f2py_numpy_version__ is present
    # in the compiled module and that has the value np.__version__.
    spec = util.F2PyModuleSpec(
        test_class_name="TestModuleAndSubroutine",
        sources = [util.getpath("tests", "src", "regression", "gh25337", "data.f90"),
                   util.getpath("tests", "src", "regression", "gh25337", "use_data.f90")],
    )
    return spec


@pytest.mark.parametrize("_mod", ["mod_subrout_spec"], indirect=True)
def test_gh25337(_mod):
    _mod.data.set_shift(3)
    assert "data" in dir(_mod)


@pytest.fixture(scope="module")
def include_spec():
    # Check that th attribute __f2py_numpy_version__ is present
    # in the compiled module and that has the value np.__version__.
    spec = util.F2PyModuleSpec(
        test_class_name="TestIncludeFiles",
        sources = [util.getpath("tests", "src", "regression", "incfile.f90")],
        options = [f"-I{util.getpath('tests', 'src', 'regression')}",
                   f"--include-paths {util.getpath('tests', 'src', 'regression')}"],
    )
    return spec


@pytest.mark.parametrize("_mod", ["include_spec"], indirect=True)
def test_gh25344(_mod):
    exp = 7.0
    res = _mod.add(3.0, 4.0)
    assert  exp == res
