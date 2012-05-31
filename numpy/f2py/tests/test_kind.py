import os
import math
import subprocess

from numpy.testing import *
from numpy import array

import util

def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))

from numpy.f2py.crackfortran import _selected_int_kind_func as selected_int_kind
from numpy.f2py.crackfortran import _selected_real_kind_func as selected_real_kind

class TestKind(util.F2PyTest):
    sources = [_path('src', 'kind', 'foo.f90'),
               ]

    @dec.slow
    def test_all(self):
        selectedrealkind = self.module.selectedrealkind
        selectedintkind = self.module.selectedintkind

        for i in range(40):
            assert_(selectedintkind(i) in [selected_int_kind(i),-1],\
                    'selectedintkind(%s): expected %r but got %r' %  (i, selected_int_kind(i), selectedintkind(i)))

        max_real_kind = 15
        try:
            subprocess.check_call(['gfortran', '--version'],
                    stdout=subprocess.PIPE)
            max_real_kind = 20
        except OSError:
            # gfortran does not exist, so it is not the compiler used.
            # Other compilers behave differently for kind numbers > 8.
            # This includes Intel, PGI, Cray, and Pathscale.
            # NAG behaves completely different, using 1, 2, 3 as its "kind"s.
            # Details:
            # https://gist.github.com/2767436
            pass

        for i in range(max_real_kind):
            assert_(selectedrealkind(i) in [selected_real_kind(i),-1],\
                    'selectedrealkind(%s): expected %r but got %r' %  (i, selected_real_kind(i), selectedrealkind(i)))

if __name__ == "__main__":
    import nose
    nose.runmodule()
