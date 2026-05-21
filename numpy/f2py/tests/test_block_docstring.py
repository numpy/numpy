import sys

import pytest

from . import util


@pytest.mark.slow
class TestBlockDocString(util.F2PyTest):
    sources = [util.getpath("tests", "src", "block_docstring", "foo.f")]

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Fails with MinGW64 Gfortran (Issue #9673)")
    def test_block_docstring(self):
        expected = "bar : 'i'-array(2,3)\n"
        assert self.module.block.__doc__ == expected
