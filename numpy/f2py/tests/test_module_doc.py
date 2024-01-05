import os
import sys
import pytest
import textwrap

from . import util
from numpy.testing import IS_PYPY


@pytest.fixture(scope="module")
def modstring_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestModuleDocString",
        sources = [
            util.getpath("tests", "src", "module_data",
                         "module_data_docstring.f90")
        ],
    )
    return spec

@pytest.mark.xfail(IS_PYPY,
                   reason="PyPy cannot modify tp_doc after PyType_Ready")
@pytest.mark.parametrize("_mod", ["modstring_spec"], indirect=True)
def test_module_docstring(_mod):
    assert _mod.mod.__doc__ == textwrap.dedent("""\
                 i : 'i'-scalar
                 x : 'i'-array(4)
                 a : 'f'-array(2,3)
                 b : 'f'-array(-1,-1), not allocated\x00
                 foo()\n
                 Wrapper for ``foo``.\n\n""")
