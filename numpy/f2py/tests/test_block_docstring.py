import sys
import pytest
from . import util

from numpy.testing import IS_PYPY, IS_WASM


@pytest.fixture(scope="module")
def block_docstring_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestBlockDocString",
        sources=[util.getpath("tests", "src", "block_docstring", "foo.f")],
    )
    return spec


@pytest.mark.xfail(IS_PYPY, reason="PyPy cannot modify tp_doc after PyType_Ready")
@pytest.mark.parametrize(
    "_mod",
    ["block_docstring_spec"],
    indirect=True,
)
def test_block_docstring(_mod):
    expected = "bar : 'i'-array(2,3)\n"
    assert _mod.block.__doc__ == expected
