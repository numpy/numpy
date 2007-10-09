from numpy.testing import *

from numpy.scons_fake import spam

class test_ra(NumpyTestCase):
    def test(self):
        spam.system('dir')

if __name__ == "__main__":
    NumpyTest('numpy.scons_fake.foo').run()
