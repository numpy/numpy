import os
import math

from numpy.testing import *
from numpy import array

import util

def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))

class TestSizeSumExample(util.F2PyTest):
    sources = [_path('src', 'size', 'foo.f90'),
               ]

    @dec.slow
    def test_all(self):
        r = self.module.foo([[1,2]])
        assert_equal(r, [3],`r`)

        r = self.module.foo([[1,2],[3,4]])
        assert_equal(r, [3,7],`r`)

        r = self.module.foo([[1,2],[3,4],[5,6]])
        assert_equal(r, [3,7,11],`r`)

if __name__ == "__main__":
    import nose
    nose.runmodule()
