import os
from . import util

from numpy.testing import assert_equal


def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))


class TestModuleDocString(util.F2PyTest):
    sources = [_path('src', 'module_data', 'module_data_docstring.f90')]

    def test_module_docstring(self):
        expected = "i : 'i'-scalar\nx : 'i'-array(4)\na : 'f'-array(2,3)\nb : 'f'-array(-1,-1), not allocated\x00\nfoo()\n\nWrapper for ``foo``.\n\n"
        assert_equal(self.module.mod.__doc__, expected)
