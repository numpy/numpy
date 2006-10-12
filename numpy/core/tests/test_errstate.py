#~ import sys
#~ if sys.version_info[:2] >= (2, 5):
    #~ exec """
from __future__ import with_statement
from numpy.core import *
from numpy.random import rand, randint
from numpy.testing import *



class test_errstate(NumpyTestCase):

    
    def test_invalid(self):
        with errstate(all='raise', under='ignore'):
            a = -arange(3)
            # This should work
            with errstate(invalid='ignore'):
                sqrt(a)
            # While this should fail!
            try:
                sqrt(a)
            except FloatingPointError:
                pass
            else:
                self.fail()
                
    def test_divide(self):
        with errstate(all='raise', under='ignore'):
            a = -arange(3)
            # This should work
            with errstate(divide='ignore'):
                a / 0
            # While this should fail!
            try:
                a / 0
            except FloatingPointError:
                pass
            else:
                self.fail()
#~ """

if __name__ == '__main__':
    from numpy.testing import *
    NumpyTest().run()