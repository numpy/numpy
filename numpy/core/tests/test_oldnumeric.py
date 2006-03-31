from numpy.testing import *

from numpy import array, ndarray, arange, argmax
from numpy.core.oldnumeric import put

class test_put(ScipyTestCase):
    def check_bug_r2089(self, level=1):
        a = array([0,0,0])
        put(a,[1],[1.2])
        assert_array_equal(a,[0,1,0])
        put(a,[1],array([2.2]))
        assert_array_equal(a,[0,2,0])
        
class test_wrapit(ScipyTestCase):
    def check_array_subclass(self, level=1):
        class subarray(ndarray):
            def get_argmax(self):
                raise AttributeError
            argmax = property(get_argmax)
        a = subarray([3], int, arange(3))
        assert_equal(argmax(a), 2)
        b = subarray([3, 3], int, arange(9))
        bmax = argmax(b)
        assert_array_equal(bmax, [2,2,2])
        assert_equal(type(bmax), subarray)


if __name__ == "__main__":
    ScipyTest().run()
