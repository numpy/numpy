"""See https://github.com/numpy/numpy/pull/10676.

"""
import sys
import pytest

from . import util


@pytest.fixture(scope="module")
def quotedchar_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestQuotedCharacter",
        sources = [util.getpath("tests", "src", "quoted_character", "foo.f")],
    )
    return spec

@pytest.mark.skipif(sys.platform == "win32",
                    reason="Fails with MinGW64 Gfortran (Issue #9673)")
@pytest.mark.parametrize("_mod", ["quotedchar_spec"], indirect=True)
def test_quoted_character(_mod):
    assert _mod.foo() == (b"'", b'"', b";", b"!", b"(", b")")
