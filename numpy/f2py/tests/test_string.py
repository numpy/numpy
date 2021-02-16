import os
import pytest

from numpy.testing import assert_array_equal, assert_raises
import numpy as np
from . import util


def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))

class TestString(util.F2PyTest):
    sources = [_path('src', 'string', 'char.f90')]

    @pytest.mark.slow
    def test_char(self):
        strings = np.array(['ab', 'cd', 'ef'], dtype='c').T
        inp, out = self.module.char_test.change_strings(strings, strings.shape[1])
        assert_array_equal(inp, strings)
        expected = strings.copy()
        expected[1, :] = 'AAA'
        assert_array_equal(out, expected)

    def test_char_bytesize(self):
        a = np.array(b'123')
        b = np.array(b'12345')
        self.module.char_test.string_size(b)
        assert_raises(ValueError, self.module.char_test.string_size, a)
        assert_array_equal(b, np.array(b'A234'))

