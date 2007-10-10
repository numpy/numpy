from numpy.testing import *

from numpy.scons_fake.ctypesext import foo

class test_ra(NumpyTestCase):
    def test(self):
        foo()

if __name__ == "__main__":
    NumpyTest('numpy.scons_fake.ctypesext').run()
