import os
import math

from numpy.testing import *
from numpy import array

import util

def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))

class TestMixed(util.F2PyTest):
    sources = [_path('src', 'mixed', 'foo.f'),
               _path('src', 'mixed', 'foo_fixed.f90'),
               _path('src', 'mixed', 'foo_free.f90')]

    @dec.slow
    def test_all(self):
        assert_( self.module.bar11() == 11)
        assert_( self.module.foo_fixed.bar12() == 12)
        assert_( self.module.foo_free.bar13() == 13)

if __name__ == "__main__":
    import nose
    nose.runmodule()
