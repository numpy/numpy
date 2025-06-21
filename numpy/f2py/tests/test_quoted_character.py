"""See https://github.com/numpy/numpy/pull/10676.

"""
import sys

import pytest

from numpy.f2py import testutils


class TestQuotedCharacter(testutils.F2PyTest):
    sources = [testutils.getpath("tests", "src", "quoted_character", "foo.f")]

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Fails with MinGW64 Gfortran (Issue #9673)")
    @pytest.mark.slow
    def test_quoted_character(self):
        assert self.module.foo() == (b"'", b'"', b";", b"!", b"(", b")")
