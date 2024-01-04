import os
import textwrap
import pytest

from numpy.testing import IS_PYPY
from . import util


@pytest.fixture(scope="module")
def mixed_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestMixed",
        sources=[
            util.getpath("tests", "src", "mixed", "foo.f"),
            util.getpath("tests", "src", "mixed", "foo_fixed.f90"),
            util.getpath("tests", "src", "mixed", "foo_free.f90"),
        ],
    )
    return spec


@pytest.mark.parametrize("_mod", ["mixed_spec"], indirect=True)
def test_all(_mod):
    assert _mod.bar11() == 11
    assert _mod.foo_fixed.bar12() == 12
    assert _mod.foo_free.bar13() == 13


@pytest.mark.xfail(IS_PYPY, reason="PyPy cannot modify tp_doc after PyType_Ready")
@pytest.mark.parametrize("_mod", ["mixed_spec"], indirect=True)
def test_docstring(_mod):
    expected = textwrap.dedent(
        """\
    a = bar11()

    Wrapper for ``bar11``.

    Returns
    -------
    a : int
    """
    )
    assert _mod.bar11.__doc__ == expected
