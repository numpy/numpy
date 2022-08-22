
from numpy.core._internal import IS_PYPY
import numpy.testing


class TestInternal:

    def test_IS_PYPY(self):
        assert IS_PYPY == numpy.testing.IS_PYPY, \
                'fast check on PYPY is unequal to full check'
        
