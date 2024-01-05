import os
import pytest

from . import util


@pytest.fixture(scope="module")
def value_attr_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestValueAttr",
        sources=[util.getpath("tests", "src", "value_attrspec", "gh21665.f90")],
    )
    return spec


# gh-21665
@pytest.mark.parametrize("_mod", ["value_attr_spec"], indirect=True)
def test_gh21665(_mod):
    inp = 2
    out = _mod.fortfuncs.square(inp)
    exp_out = 4
    assert out == exp_out
