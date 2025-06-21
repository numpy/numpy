import textwrap

import pytest

from numpy.f2py import testutils
from numpy.testing import IS_PYPY


class TestMixed(testutils.F2PyTest):
    sources = [
        testutils.getpath("tests", "src", "mixed", "foo.f"),
        testutils.getpath("tests", "src", "mixed", "foo_fixed.f90"),
        testutils.getpath("tests", "src", "mixed", "foo_free.f90"),
    ]

    @pytest.mark.slow
    def test_all(self):
        assert self.module.bar11() == 11
        assert self.module.foo_fixed.bar12() == 12
        assert self.module.foo_free.bar13() == 13

    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = bar11()

        Wrapper for ``bar11``.

        Returns
        -------
        a : int
        """)
        assert self.module.bar11.__doc__ == expected
