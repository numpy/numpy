import sys

import pytest

from numpy.f2py import _testutils
from numpy.testing import IS_PYPY


@pytest.mark.slow
class TestBlockDocString(_testutils.F2PyTest):
    sources = [_testutils.getpath("tests", "src", "block_docstring", "foo.f")]

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Fails with MinGW64 Gfortran (Issue #9673)")
    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_block_docstring(self):
        expected = "bar : 'i'-array(2,3)\n"
        assert self.module.block.__doc__ == expected
