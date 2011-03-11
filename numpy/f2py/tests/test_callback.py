from numpy.testing import *
from numpy import array
import math
import util

class TestF77Callback(util.F2PyTest):
    code = """
       subroutine t(fun,a)
       integer a
cf2py  intent(out) a
       external fun
       call fun(a)
       end

       subroutine func(a)
cf2py  intent(in,out) a
       integer a
       a = a + 11
       end

       subroutine func0(a)
cf2py  intent(out) a
       integer a
       a = 11
       end

       subroutine t2(a)
cf2py  intent(callback) fun
       integer a
cf2py  intent(out) a
       external fun
       call fun(a)
       end
    """

    @dec.slow
    def test_all(self):
        for name in "t,t2".split(","):
            self.check_function(name)

    def check_function(self, name):
        t = getattr(self.module, name)
        r = t(lambda : 4)
        assert_( r==4,`r`)
        r = t(lambda a:5,fun_extra_args=(6,))
        assert_( r==5,`r`)
        r = t(lambda a:a,fun_extra_args=(6,))
        assert_( r==6,`r`)
        r = t(lambda a:5+a,fun_extra_args=(7,))
        assert_( r==12,`r`)
        r = t(lambda a:math.degrees(a),fun_extra_args=(math.pi,))
        assert_( r==180,`r`)
        r = t(math.degrees,fun_extra_args=(math.pi,))
        assert_( r==180,`r`)

        r = t(self.module.func, fun_extra_args=(6,))
        assert_( r==17,`r`)
        r = t(self.module.func0)
        assert_( r==11,`r`)
        r = t(self.module.func0._cpointer)
        assert_( r==11,`r`)
        class A:
            def __call__(self):
                return 7
            def mth(self):
                return 9
        a = A()
        r = t(a)
        assert_( r==7,`r`)
        r = t(a.mth)
        assert_( r==9,`r`)

if __name__ == "__main__":
    import nose
    nose.runmodule()
