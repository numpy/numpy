from numpy.testing import *

from numpy import array
from numpy.core.oldnumeric import put

class test_put(ScipyTestCase):
    def check_bug_r2089(self, level=1):
        a = array([0,0,0])
        put(a,[1],[1.2])
        assert_array_equal(a,[0,1,0])
        put(a,[1],array([2.2]))
        assert_array_equal(a,[0,2,0])

if __name__ == "__main__":
    ScipyTest().run()
