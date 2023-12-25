import sys
import pytest
from . import util

from numpy.testing import IS_PYPY, IS_WASM


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
@pytest.mark.slow
@pytest.fixture(scope="module")
def _mod(module_builder_factory):
    spec = util.F2PyModuleSpec(
        test_class_name="TestBlockDocString",
        sources=[util.getpath("tests", "src", "block_docstring", "foo.f")],
    )
    return module_builder_factory(spec)

@pytest.mark.skipif(sys.platform == "win32",
                    reason="Fails with MinGW64 Gfortran (Issue #9673)")
@pytest.mark.xfail(IS_PYPY,
                   reason="PyPy cannot modify tp_doc after PyType_Ready")
def test_block_docstring(_mod):
    expected = "bar : 'i'-array(2,3)\n"
    assert _mod.block.__doc__ == expected
