from pathlib import Path
import pytest
import textwrap
from . import util
from numpy.f2py import crackfortran


@pytest.fixture(scope="module")
def abstract_interface_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestAbstractInterface",
        sources=[util.getpath("tests", "src", "abstract_interface", "foo.f90")],
        skip=["add1", "add2"],
    )
    return spec


@pytest.mark.parametrize("_mod", ["abstract_interface_spec"], indirect=True)
def test_abstract_interface(_mod):
    assert _mod.ops_module.foo(3, 5) == (8, 13)


def test_parse_abstract_interface():
    # Test gh18403
    fpath = util.getpath("tests", "src", "abstract_interface", "gh18403_mod.f90")
    mod = crackfortran.crackfortran([str(fpath)])
    assert len(mod) == 1
    assert len(mod[0]["body"]) == 1
    assert mod[0]["body"][0]["block"] == "abstract interface"
