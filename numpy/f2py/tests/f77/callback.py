__usage__ = """
Run:
  python callback.py [<f2py options>]
Examples:
  python callback.py --fcompiler=Gnu --no-wrap-functions
  python callback.py --quiet
"""

import f2py2e
import math
import sys
from Numeric import array

def build(f2py_opts):
    try:
        import f77_ext_callback
    except ImportError:
        assert not f2py2e.compile('''\
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

''','f77_ext_callback',f2py_opts,source_fn='f77_callback.f')

    from f77_ext_callback import t,t2
    test_functions = [t,t2]
    return test_functions

def runtest(t):
    r = t(lambda : 4)
    assert r==4,`r`
    r = t(lambda a:5,fun_extra_args=(6,))
    assert r==5,`r`
    r = t(lambda a:a,fun_extra_args=(6,))
    assert r==6,`r`
    r = t(lambda a:5+a,fun_extra_args=(7,))
    assert r==12,`r`
    if sys.version[:3]>='2.3':
        r = t(lambda a:math.degrees(a),fun_extra_args=(math.pi,))
        assert r==180,`r`
        r = t(math.degrees,fun_extra_args=(math.pi,))
        assert r==180,`r`
    from f77_ext_callback import func,func0
    r = t(func,fun_extra_args=(6,))
    assert r==17,`r`
    r = t(func0)
    assert r==11,`r`
    r = t(func0._cpointer)
    assert r==11,`r`
    class A:
        def __call__(self):
            return 7
        def mth(self):
            return 9
    a = A()
    r = t(a)
    assert r==7,`r`
    r = t(a.mth)
    assert r==9,`r`

if __name__=='__main__':
    #import libwadpy
    status = 1
    try:
        repeat,f2py_opts = f2py2e.f2py_testing.cmdline()
        test_functions = build(f2py_opts)
        f2py2e.f2py_testing.run(runtest,test_functions,repeat)
        print 'ok'
        status = 0
    finally:
        if status:
            print '*'*20
            print 'Running f2py2e.diagnose'
            import f2py2e.diagnose
            f2py2e.diagnose.run()
