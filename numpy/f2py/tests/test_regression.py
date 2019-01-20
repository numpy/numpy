from __future__ import division, absolute_import, print_function

import os
import pytest

import numpy as np
from numpy.testing import assert_raises, assert_equal

from . import util


def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))


class TestIntentInOut(util.F2PyTest):
    # Check that intent(in out) translates as intent(inout)
    sources = [_path('src', 'regression', 'inout.f90')]

    @pytest.mark.slow
    def test_inout(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float32)[::2]
        assert_raises(ValueError, self.module.foo, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float32)
        self.module.foo(x)
        assert_equal(x, [3, 1, 2])

@pytest.mark.parametrize('code', [
        'program test_f2py\nend program test_f2py',
        b'program test_f2py\nend program test_f2py',
    ])
def test_compile(tmpdir, code):
    # Make sure we can compile str and bytes gh-12796
    cwd = os.getcwd()
    try:
        os.chdir(str(tmpdir))
        ret = np.f2py.compile(code, modulename='test1_f2py', extension='.f90')
        assert_equal(ret, 0)
    finally:
        os.chdir(cwd)
