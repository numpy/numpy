from . import util
import pytest
import numpy as np


@pytest.fixture(scope="module")
def f2cmap_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestF2Cmap",
        sources=[
            util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90"),
            util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap"),
        ],
    )
    return spec


@pytest.mark.parametrize("_mod", ["f2cmap_spec"], indirect=True)
def test_gh15095(_mod):
    inp = np.ones(3)
    out = _mod.func1(inp)
    exp_out = 3
    assert out == exp_out
