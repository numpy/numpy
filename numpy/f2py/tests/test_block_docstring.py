import sys
import pytest
from . import util

from numpy.testing import assert_equal, IS_PYPY


class TestBlockDocString(util.F2PyTest):
    sources = [util.getpath("tests", "src", "block_docstring", "foo.f")]

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Fails with MinGW64 Gfortran (Issue #9673)")
    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_block_docstring(self):
        expected = "bar : 'i'-array(2,3)\n"
        assert_equal(self.module.block.__doc__, expected)
